"""FlyDSL sparse NSA prefill attention kernel for AMD gfx950 (MI350X).

Fused paged gather + dequant + attention kernel.  No large intermediate tensor.
The raw C4 KV pool buffer is passed directly; page/slot decomposition and
dequantization happen inside the kernel.

C4 KV pool buffer layout (per page):
  bytes [0 .. page_size*576)         : token data (interleaved)
    per token slot s (0..page_size-1):
      nope fp8 : bytes [s*576 .. s*576+448)     (448 × e4m3fnuz bytes)
      rope raw : bytes [s*576+448 .. (s+1)*576) (128 raw bytes = 64 bf16 LE)
  bytes [page_size*576 .. page_size*584) : scale section
    per token slot s:
      scale u8 : 7 × ue8m0 bytes at [page_size*576 + s*8 .. +7)
      pad      : 1 byte padding
  bytes [page_size*584 ..)           : zero-padding to align page to 576 bytes

Dequantization (inside kernel, per KV token k):
  nope_f32[d] = cvt_f32_fp8(nope_fp8[d]) * exp2(scale_u8[d//64] - 127)
  rope_f32[d] = bf16_raw_to_f32(rope_raw[2*d:2*d+2])
  kv_f32 = concat(nope_f32, rope_f32)             # 512 dims
  kv_fp8[d] = cvt_pk_fp8_f32(kv_f32[d] * INV_Q_SCALE)
            where INV_Q_SCALE = fp8_max / CLIP = fp8_max / 8.0

LLVM fp8 type constraints (same as before):
  ❌ memref<Nxf8>           — no fp8 type in LLVM
  ✓ load raw i64, use rocdl.cvt_f32_fp8 / rocdl.cvt_pk_fp8_f32

Kernel structure:
  Gather+Dequant: read pool bytes → dequant+requant → store i64 LDS
  GEMM1: Q[16,512] @ K[32,512]^T  — fp8 MFMA
  Softmax: f32 online softmax
  P→LDS: store f32 attention weights
  GEMM2: P[16,32] @ V[32,512]     — fp8 MFMA
"""

from __future__ import annotations

import functools
import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import (
    arith as _mlir_arith,
    fly as _fly,
    llvm as _llvm,
    memref as _memref,
)

_LOG2E = host_math.log2(host_math.e)
_LN2   = host_math.log(2.0)

# Pool layout constants (must match DeepSeekV4SingleKVPool.get_bytes_per_token)
_SLOT_BYTES  = 576   # 448 nope fp8 + 128 rope raw bytes per token
_NOPE_BYTES  = 448
_ROPE_BYTES  = 128   # 64 bf16 × 2 raw bytes
_SCALE_BYTES = 8     # 7 ue8m0 + 1 pad per token
_CLIP        = 8.0   # fixed quantization clip value


def _bytes_per_page_padded(c4_page_size: int) -> int:
    """Replicate DeepSeekV4SingleKVPool.create_buffer padding formula."""
    non_padded = c4_page_size * (_SLOT_BYTES + _SCALE_BYTES)
    return host_math.ceil(non_padded / _SLOT_BYTES) * _SLOT_BYTES


def _is_available() -> bool:
    try:
        return str(get_rocm_arch()).startswith("gfx950")
    except Exception:
        return False


@functools.lru_cache(maxsize=64)
def build_nsa_prefill_paged_kernel(
    h_q: int,
    head_dim: int = 512,
    topk: int = 2048,
    tile_m: int = 16,
    block_n: int = 32,
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
    c4_page_size: int = 64,
    fp8_max: float = 448.0,
):
    """Build the fused paged gather+dequant+attention kernel.

    c4_page_size: tokens per C4 pool page (= full_page_size // 4, typically 64)
    fp8_max:      maximum representable value of the FP8 format in use
                  (448.0 for e4m3fn, 240.0 for e4m3fnuz)
    """
    assert head_dim == 512
    assert tile_m == 16
    assert block_n == 32
    assert topk % block_n == 0
    assert (c4_page_size & (c4_page_size - 1)) == 0, "c4_page_size must be power-of-2"

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    gpu_arch = get_rocm_arch()

    WARP_SIZE  = 64
    TILE_M     = tile_m
    BLOCK_N    = block_n
    HEAD_DIM   = head_dim
    TOPK       = topk
    BLOCK_SIZE = WARP_SIZE

    MFMA_N      = 16
    MFMA_K      = 32
    K_STEPS_QK  = HEAD_DIM // MFMA_K    # 16
    N_BLKS_S    = BLOCK_N  // MFMA_N    # 2
    D_BLKS      = HEAD_DIM // MFMA_N    # 32
    KV_TILES    = TOPK // BLOCK_N

    D_HALF   = HEAD_DIM // 2            # 256
    D_CHUNKS = D_HALF // 8              # 32

    # LDS layout (same as before):
    #   KV  : i64 memref, LDS_KV_I64 entries (8 FP8 per i64)
    #   P   : f32 memref, LDS_P_F32  entries
    LDS_KV_I64    = BLOCK_N * HEAD_DIM // 8    # 2048
    LDS_KV_BYTES  = LDS_KV_I64 * 8             # 16384
    LDS_P_F32     = TILE_M * BLOCK_N           # 512
    LDS_P_BYTES   = LDS_P_F32 * 4             # 2048
    LDS_TOTAL     = LDS_KV_BYTES + LDS_P_BYTES

    Q_STOK  = h_q * HEAD_DIM
    IDX_STR = TOPK

    # Pool layout compile-time constants
    BYTES_PER_PAGE  = _bytes_per_page_padded(c4_page_size)
    PAGE_SIZE_LOG2  = int(host_math.log2(c4_page_size))
    PAGE_SIZE_MASK  = c4_page_size - 1
    INV_Q_SCALE     = fp8_max / _CLIP            # fp8_max / 8.0

    alloc      = SmemAllocator(None, arch=gpu_arch, global_sym_name="nsa_smem_paged")
    base_off   = alloc._align(alloc.ptr, 16)
    alloc.ptr  = base_off + LDS_TOTAL
    kv_lds_off = base_off
    p_lds_off  = base_off + LDS_KV_BYTES

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def nsa_prefill_paged_kernel(
        Q:        fx.Tensor,   # [total_tokens, h_q, HEAD_DIM]  fp8
        KV_pool:  fx.Tensor,   # [N_pages, BYTES_PER_PAGE]       uint8 (raw pool)
        Indices:  fx.Tensor,   # [total_tokens, TOPK]            int32
        Out:      fx.Tensor,   # [total_tokens, h_q, HEAD_DIM]  bf16
        M_Raw:    fx.Tensor,   # [padded_tokens, h_q]           float32
        L_Raw:    fx.Tensor,   # [padded_tokens, h_q]           float32
        total_tokens: fx.Int32,
    ):
        v4f32_type = Vec.make_type(4, fx.Float32)

        fm = arith.FastMathFlags.fast
        def _fadd(a, b): return arith.addf(_raw(a), _raw(b), fastmath=fm)
        def _fmul(a, b): return arith.mulf(_raw(a), _raw(b), fastmath=fm)
        def _fsub(a, b): return arith.subf(_raw(a), _raw(b), fastmath=fm)
        def _fmax(a, b): return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm).result
        def _fmin(a, b): return arith.MinNumFOp(_raw(a), _raw(b), fastmath=fm).result

        c_neg_inf     = fx.Float32(float("-inf"))
        c_zero_f      = fx.Float32(0.0)
        c_one_f       = fx.Float32(1.0)
        c_sm_log2e    = fx.Float32(sm_scale * _LOG2E)
        c_zero_v4f32  = Vec.filled(4, 0.0, fx.Float32)
        c_inv_q_scale = fx.Float32(INV_Q_SCALE)

        # ── LLVM pointers ─────────────────────────────────────────────────
        def _ptr_ty():  return ir.Type.parse("!llvm.ptr")
        def _as_ptr(t):
            v = t
            if hasattr(v, "ir_value") and not isinstance(v, ir.Value):
                v = v.ir_value()
            return _fly.extract_aligned_pointer_as_index(_ptr_ty(), v)

        q_ptr    = _as_ptr(Q)
        pool_ptr = _as_ptr(KV_pool)
        idx_ptr  = _as_ptr(Indices)
        out_ptr  = _as_ptr(Out)
        m_ptr    = _as_ptr(M_Raw)
        l_ptr    = _as_ptr(L_Raw)

        # ── Load helpers ──────────────────────────────────────────────────
        def load_8bytes_as_i64(ptr, byte_off_index):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(byte_off_index), elem_type=T.i8)
            return _llvm.LoadOp(T.i64, gep).result

        def load_i32_elem(ptr, elem_off_index):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(elem_off_index), elem_type=T.i32)
            return _llvm.LoadOp(T.i32, gep).result

        def load_u8_as_i32(ptr, byte_off_index):
            """Load one unsigned byte from the pool (offset is fx.Index)."""
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(byte_off_index), elem_type=T.i8)
            u8 = _llvm.LoadOp(T.i8, gep).result
            return _mlir_arith.ExtUIOp(T.i32, u8).result

        # ── MFMA ──────────────────────────────────────────────────────────
        def mfma_fp8(acc, a_i64, b_i64):
            return rocdl.mfma_f32_16x16x32_fp8_fp8(
                v4f32_type, [a_i64, b_i64, acc, 0, 0, 0]
            )

        # ── Decode FP8 byte (in i32[7:0]) to f32 via AMD hw ───────────────
        def fp8_byte_to_f32(byte_i32):
            return rocdl.cvt_f32_fp8(T.f32, byte_i32, fx.Int32(0))

        # ── Extract byte j (0..7) from i64 as i32 (zero-extended) ─────────
        def extract_byte_i32(v_i64, j_const):
            sh  = _mlir_arith.ShRUIOp(v_i64, _raw(fx.Int64(j_const * 8))).result
            msk = _mlir_arith.AndIOp(sh, _raw(fx.Int64(0xFF))).result
            return _mlir_arith.TruncIOp(T.i32, msk).result

        # ── Extract bf16 value j (0..3) from i64, convert to f32 ──────────
        # bf16 is stored as 2 raw bytes (little-endian).  Conversion:
        # zero-extend 16-bit bf16 bits to i32, shift-left 16 → upper half of f32.
        def extract_bf16_as_f32(v_i64, j_const):
            sh   = _mlir_arith.ShRUIOp(v_i64, _raw(fx.Int64(j_const * 16))).result
            msk  = _mlir_arith.AndIOp(sh, _raw(fx.Int64(0xFFFF))).result
            bits = _mlir_arith.TruncIOp(T.i32, msk).result
            f32b = _mlir_arith.ShLIOp(bits, _raw(fx.Int32(16))).result
            return _mlir_arith.BitcastOp(T.f32, f32b).result

        # ── ue8m0 scale byte → f32 scale: exp2(u8 - 127) ─────────────────
        def ue8m0_to_f32(u8_i32):
            u8_f32 = _mlir_arith.UIToFPOp(T.f32, u8_i32).result
            delta  = _fsub(u8_f32, fx.Float32(127.0))
            return rocdl.exp2(T.f32, _raw(delta))

        # ── Dequant 8 fp8 bytes via per-tile scale → requantize → i64 ─────
        # combined = tile_scale * INV_Q_SCALE; hw cvt saturates automatically
        def dequant_requant_8fp8(raw_i64, tile_scale_f32):
            combined = _fmul(tile_scale_f32, c_inv_q_scale)
            f32s = []
            for j in range_constexpr(8):
                b   = extract_byte_i32(raw_i64, j)
                fv  = fp8_byte_to_f32(b)
                f32s.append(_fmul(fv, combined))
            return _pack_8f32_to_fp8_i64(f32s)

        # ── Requantize 8 bf16 values (from 2×i64 raw bytes) → fp8 i64 ────
        def requant_8bf16(lo_i64, hi_i64):
            f32s = []
            for j in range_constexpr(4):
                fv = extract_bf16_as_f32(lo_i64, j)
                f32s.append(_fmul(fv, c_inv_q_scale))
            for j in range_constexpr(4):
                fv = extract_bf16_as_f32(hi_i64, j)
                f32s.append(_fmul(fv, c_inv_q_scale))
            return _pack_8f32_to_fp8_i64(f32s)

        # ── Pack 8 f32 values as fp8 bytes into i64 ───────────────────────
        # Uses rocdl.cvt_pk_fp8_f32 which saturates out-of-range values.
        def _pack_8f32_to_fp8_i64(f32s):
            i32_lo = rocdl.cvt_pk_fp8_f32(T.i32, f32s[1], f32s[0], fx.Int32(0), fx.Int32(0))
            i32_lo = rocdl.cvt_pk_fp8_f32(T.i32, f32s[3], f32s[2], i32_lo,       fx.Int32(1))
            i32_hi = rocdl.cvt_pk_fp8_f32(T.i32, f32s[5], f32s[4], fx.Int32(0), fx.Int32(0))
            i32_hi = rocdl.cvt_pk_fp8_f32(T.i32, f32s[7], f32s[6], i32_hi,       fx.Int32(1))
            lo64 = _mlir_arith.ExtUIOp(T.i64, i32_lo).result
            hi64 = _mlir_arith.ShLIOp(
                _mlir_arith.ExtUIOp(T.i64, i32_hi).result, _raw(fx.Int64(32))
            ).result
            return _mlir_arith.OrIOp(lo64, hi64).result

        # ── Thread / block ids ────────────────────────────────────────────
        tid     = fx.Index(gpu.thread_idx.x)
        bid_m   = fx.Index(gpu.block_idx.x)
        bid_h   = fx.Index(gpu.block_idx.y)
        lane    = tid % 16
        k_group = tid // 16
        q_start = bid_m * fx.Index(TILE_M)

        # V-extraction constants (used in GEMM2)
        lane_mod8      = lane % fx.Index(8)
        lane_div8      = lane // fx.Index(8)
        lane_shift_i64 = fx.Int64(lane_mod8) * fx.Int64(8)

        # ── LDS ───────────────────────────────────────────────────────────
        lds_base   = alloc.get_base()
        lds_kv_i64 = SmemPtr(lds_base, kv_lds_off, T.i64, shape=(LDS_KV_I64,)).get()
        lds_p_f32  = SmemPtr(lds_base, p_lds_off,  T.f32, shape=(LDS_P_F32,)).get()

        # ── Pre-load Q into registers ─────────────────────────────────────
        q_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_byte_off = (
                (q_start + lane) * fx.Index(Q_STOK)
                + bid_h * fx.Index(HEAD_DIM)
                + fx.Index(ks * MFMA_K)
                + k_group * fx.Index(8)
            )
            q_packs.append(load_8bytes_as_i64(q_ptr, q_byte_off))

        # ── Online softmax init ───────────────────────────────────────────
        _init = (
            [_raw(c_neg_inf)] * 4
            + [_raw(c_zero_f)]  * 4
            + [_raw(c_zero_v4f32)] * D_BLKS
        )
        anchor_tok = q_start

        # ── Main KV-tile loop ─────────────────────────────────────────────
        loop_results = _init
        for kv_tile, _carry in range(0, KV_TILES, 1, init=_init):
            m_run = [_carry[r]     for r in range_constexpr(4)]
            l_run = [_carry[4 + r] for r in range_constexpr(4)]
            o_acc = [_carry[8 + d] for d in range_constexpr(D_BLKS)]

            kv_pos_base = kv_tile * fx.Index(BLOCK_N)

            # ── Fused gather + dequant + requant → LDS i64 ────────────────
            # Each thread handles one KV row and one half of the head dimension.
            # kv_row:  which of the BLOCK_N=32 KV rows this thread fills
            # d_group: 0 → dims 0..255 (nope[0..255])
            #          1 → dims 256..447 (nope[256..447]) + dims 448..511 (rope[0..63])
            kv_row  = tid % fx.Index(BLOCK_N)
            d_group = tid // fx.Index(BLOCK_N)

            # Load flat pool token index k from the indices tensor
            idx_flat = anchor_tok * fx.Index(IDX_STR) + kv_pos_base + kv_row
            k_i32    = load_i32_elem(idx_ptr, idx_flat)

            # Decompose k into page and slot (power-of-2, uses shifts/masks)
            page_i32 = _mlir_arith.ShRUIOp(k_i32, _raw(fx.Int32(PAGE_SIZE_LOG2))).result
            slot_i32 = _mlir_arith.AndIOp(k_i32, _raw(fx.Int32(PAGE_SIZE_MASK))).result
            page_idx = fx.Index(page_i32)
            slot_idx = fx.Index(slot_i32)

            # Byte offset of this token's data section within the flat pool
            token_base = (
                page_idx * fx.Index(BYTES_PER_PAGE)
                + slot_idx * fx.Index(_SLOT_BYTES)
            )
            # Byte offset of this token's scale section
            scale_base = (
                page_idx * fx.Index(BYTES_PER_PAGE)
                + fx.Index(c4_page_size * _SLOT_BYTES)
                + slot_idx * fx.Index(_SCALE_BYTES)
            )

            for dc in range_constexpr(D_CHUNKS):  # 0..31
                i64_off = (
                    kv_row * fx.Index(HEAD_DIM // 8)
                    + d_group * fx.Index(D_HALF // 8)
                    + fx.Index(dc)
                )

                if dc < 24:
                    # Both d_groups are in the nope region:
                    #   d_group=0: abs_dim = dc*8 ∈ [0, 184],  tiles 0..2
                    #   d_group=1: abs_dim = 256+dc*8 ∈ [256,440], tiles 4..6
                    abs_nope = d_group * fx.Index(D_HALF) + fx.Index(dc * 8)
                    nope_off = token_base + abs_nope
                    tile_off = d_group * fx.Index(4) + fx.Index(dc // 8)
                    sc_off   = scale_base + tile_off

                    raw_i64      = load_8bytes_as_i64(pool_ptr, nope_off)
                    scale_u8_i32 = load_u8_as_i32(pool_ptr, sc_off)
                    tile_scale   = ue8m0_to_f32(scale_u8_i32)
                    result_i64   = dequant_requant_8fp8(raw_i64, tile_scale)

                    _memref.store(result_i64, lds_kv_i64, [_raw(i64_off)])

                else:
                    # dc = 24..31: d_group=0 stays in nope, d_group=1 enters rope.
                    # Avoid a runtime SelectOp (wrong API); instead each thread writes
                    # its own slot unconditionally — stores never collide because
                    # i64_off_d0 and i64_off_d1 are distinct.

                    # d_group=0 path: abs_dim = dc*8 ∈ [192, 248], tile=3
                    nope_off_d0 = token_base + fx.Index(dc * 8)
                    sc_off_3    = scale_base + fx.Index(3)
                    raw_nope_d0 = load_8bytes_as_i64(pool_ptr, nope_off_d0)
                    scale_u8_3  = load_u8_as_i32(pool_ptr, sc_off_3)
                    tile_scale_3 = ue8m0_to_f32(scale_u8_3)
                    nope_i64    = dequant_requant_8fp8(raw_nope_d0, tile_scale_3)

                    # d_group=1 path: rope bf16 at byte offset (dc-24)*16 within rope section
                    _rope_start = (dc - 24) * 16   # bytes: 0,16,...,112
                    rope_lo = load_8bytes_as_i64(
                        pool_ptr,
                        token_base + fx.Index(_NOPE_BYTES + _rope_start),
                    )
                    rope_hi = load_8bytes_as_i64(
                        pool_ptr,
                        token_base + fx.Index(_NOPE_BYTES + _rope_start + 8),
                    )
                    rope_i64 = requant_8bf16(rope_lo, rope_hi)

                    # Write each result to its own slot; no select needed.
                    i64_off_d0 = kv_row * fx.Index(HEAD_DIM // 8) + fx.Index(dc)
                    i64_off_d1 = kv_row * fx.Index(HEAD_DIM // 8) + fx.Index(D_HALF // 8 + dc)
                    _memref.store(nope_i64, lds_kv_i64, [_raw(i64_off_d0)])
                    _memref.store(rope_i64, lds_kv_i64, [_raw(i64_off_d1)])

            gpu.barrier()

            # ── GEMM1: S = Q @ K^T (fp8 MFMA) ────────────────────────────
            s_acc = [_raw(c_zero_v4f32) for _ in range(N_BLKS_S)]
            for ks in range_constexpr(K_STEPS_QK):
                q_a = q_packs[ks]
                for nb in range_constexpr(N_BLKS_S):
                    k_i64_off = (
                        (fx.Index(nb * MFMA_N) + lane) * fx.Index(HEAD_DIM // 8)
                        + fx.Index(ks * 4)
                        + k_group
                    )
                    k_pack    = _memref.load(lds_kv_i64, [_raw(k_i64_off)])
                    s_acc[nb] = mfma_fp8(s_acc[nb], q_a, k_pack)

            # ── Online softmax ────────────────────────────────────────────
            s_scaled = []
            for nb in range_constexpr(N_BLKS_S):
                sv = Vec(s_acc[nb])
                s_scaled.append([_fmul(sv[r], c_sm_log2e) for r in range_constexpr(4)])

            local_max = [_fmax(s_scaled[0][r], s_scaled[1][r]) for r in range_constexpr(4)]

            shfl_w  = fx.Int32(WARP_SIZE)
            row_max = list(local_max)
            for xor_off in [8, 4, 2, 1]:
                so = fx.Int32(xor_off)
                for r in range_constexpr(4):
                    row_max[r] = _fmax(row_max[r], fx.Float32(row_max[r]).shuffle_xor(so, shfl_w))

            m_new = [_fmax(m_run[r], row_max[r]) for r in range_constexpr(4)]
            corr  = [rocdl.exp2(T.f32, _raw(_fsub(m_run[r], m_new[r]))) for r in range_constexpr(4)]

            p_vals   = [[None] * 4 for _ in range(N_BLKS_S)]
            tile_sum = [_raw(c_zero_f) for _ in range(4)]
            for nb in range_constexpr(N_BLKS_S):
                for r in range_constexpr(4):
                    p_val = rocdl.exp2(T.f32, _raw(_fsub(s_scaled[nb][r], m_new[r])))
                    p_vals[nb][r] = p_val
                    tile_sum[r]   = _fadd(tile_sum[r], p_val)

            for xor_off in [8, 4, 2, 1]:
                so = fx.Int32(xor_off)
                for r in range_constexpr(4):
                    tile_sum[r] = _fadd(tile_sum[r], fx.Float32(tile_sum[r]).shuffle_xor(so, shfl_w))

            l_new = [_fadd(_fmul(corr[r], l_run[r]), tile_sum[r]) for r in range_constexpr(4)]

            for d in range_constexpr(D_BLKS):
                ov = Vec(o_acc[d])
                o_acc[d] = Vec.from_elements(
                    [_fmul(ov[r], corr[r]) for r in range_constexpr(4)], fx.Float32
                ).ir_value()

            m_run = m_new
            l_run = l_new

            # ── Store P → f32 LDS (transposed) ───────────────────────────
            for nb in range_constexpr(N_BLKS_S):
                for r in range_constexpr(4):
                    p_row = k_group * fx.Index(4) + fx.Index(r)
                    p_col = fx.Index(nb * MFMA_N) + lane
                    p_off = p_row * fx.Index(BLOCK_N) + p_col
                    _memref.store(p_vals[nb][r], lds_p_f32, [_raw(p_off)])

            gpu.barrier()

            # ── GEMM2: O += P @ V (fp8 MFMA) ─────────────────────────────
            p_base_off = lane * fx.Index(BLOCK_N) + k_group * fx.Index(8)
            p_f32_vals = [
                _memref.load(lds_p_f32, [_raw(p_base_off + fx.Index(j))])
                for j in range(8)
            ]

            i32_lo = rocdl.cvt_pk_fp8_f32(T.i32, p_f32_vals[1], p_f32_vals[0], fx.Int32(0), fx.Int32(0))
            i32_lo = rocdl.cvt_pk_fp8_f32(T.i32, p_f32_vals[3], p_f32_vals[2], i32_lo,       fx.Int32(1))
            i32_hi = rocdl.cvt_pk_fp8_f32(T.i32, p_f32_vals[5], p_f32_vals[4], fx.Int32(0), fx.Int32(0))
            i32_hi = rocdl.cvt_pk_fp8_f32(T.i32, p_f32_vals[7], p_f32_vals[6], i32_hi,       fx.Int32(1))

            lo64    = _mlir_arith.ExtUIOp(T.i64, i32_lo).result
            hi64    = _mlir_arith.ShLIOp(
                _mlir_arith.ExtUIOp(T.i64, i32_hi).result, _raw(fx.Int64(32))
            ).result
            p_a_i64 = _mlir_arith.OrIOp(lo64, hi64).result

            for d in range_constexpr(D_BLKS):
                hd_off_f    = fx.Index(d * 2) + lane_div8
                v_bytes_i32 = []
                for j in range_constexpr(8):
                    kv_i64_off = (
                        (k_group * fx.Index(8) + fx.Index(j)) * fx.Index(HEAD_DIM // 8)
                        + hd_off_f
                    )
                    v_i64  = _memref.load(lds_kv_i64, [_raw(kv_i64_off)])
                    v_sh   = _mlir_arith.ShRUIOp(_raw(v_i64), _raw(lane_shift_i64)).result
                    v_b64  = _mlir_arith.AndIOp(v_sh, _raw(fx.Int64(0xFF))).result
                    v_bytes_i32.append(_mlir_arith.TruncIOp(T.i32, v_b64).result)

                v_b_i64 = _mlir_arith.ExtUIOp(T.i64, v_bytes_i32[0]).result
                for j in range_constexpr(7):
                    bj = _mlir_arith.ShLIOp(
                        _mlir_arith.ExtUIOp(T.i64, v_bytes_i32[j + 1]).result,
                        _raw(fx.Int64((j + 1) * 8))
                    ).result
                    v_b_i64 = _mlir_arith.OrIOp(v_b_i64, bj).result

                o_acc[d] = mfma_fp8(o_acc[d], p_a_i64, v_b_i64)

            gpu.barrier()
            loop_results = yield list(m_run) + list(l_run) + list(o_acc)

        # ── Normalize and write output ─────────────────────────────────────
        m_final_vals = [loop_results[r]     for r in range_constexpr(4)]
        l_final      = [loop_results[4 + r] for r in range_constexpr(4)]
        o_final      = [loop_results[8 + d] for d in range_constexpr(D_BLKS)]

        for d in range_constexpr(D_BLKS):
            ov = Vec(o_final[d])
            for r in range_constexpr(4):
                inv_l  = arith.divf(_raw(c_one_f), _raw(l_final[r]), fastmath=fm)
                o_norm = _fmul(ov[r], inv_l)
                o_bf16 = fx.Float32(o_norm).to(fx.BFloat16).ir_value()
                out_row = k_group * fx.Index(4) + fx.Index(r)
                out_col = fx.Index(d * MFMA_N) + lane
                out_idx = fx.Int64(
                    (q_start + out_row) * fx.Index(Q_STOK)
                    + bid_h * fx.Index(HEAD_DIM)
                    + out_col
                )
                out_gep = buffer_ops.get_element_ptr(out_ptr, out_idx, elem_type=T.bf16)
                _llvm.StoreOp(o_bf16, out_gep)

        # ── Write m_raw and l_raw ──────────────────────────────────────────
        _out_row = k_group * fx.Index(4)
        for r in range_constexpr(4):
            _lse_tok = q_start + _out_row + fx.Index(r)
            _lse_idx = fx.Int64(_lse_tok * fx.Index(h_q) + bid_h)
            _llvm.StoreOp(
                _raw(m_final_vals[r]),
                buffer_ops.get_element_ptr(m_ptr, _lse_idx, elem_type=T.f32),
            )
            _llvm.StoreOp(
                _raw(l_final[r]),
                buffer_ops.get_element_ptr(l_ptr, _lse_idx, elem_type=T.f32),
            )

    # ── JIT launcher ─────────────────────────────────────────────────────
    @flyc.jit
    def launch_nsa_prefill_paged(
        Q: fx.Tensor, KV_pool: fx.Tensor, Indices: fx.Tensor,
        Out: fx.Tensor, M_Raw: fx.Tensor, L_Raw: fx.Tensor,
        total_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalize()

        tokens_idx = fx.Index(total_tokens)
        grid_m     = (tokens_idx + TILE_M - 1) // TILE_M
        launcher   = nsa_prefill_paged_kernel(
            Q, KV_pool, Indices, Out, M_Raw, L_Raw, total_tokens
        )

        passthrough = []
        for pair in [
            ("denormal-fp-math-f32", "preserve-sign,preserve-sign"),
            ("no-nans-fp-math",      "true"),
            ("unsafe-fp-math",       "true"),
        ]:
            passthrough.append(
                ir.ArrayAttr.get([ir.StringAttr.get(pair[0]), ir.StringAttr.get(pair[1])])
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"]        = ir.ArrayAttr.get(passthrough)
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(waves_per_eu))

        launcher.launch(
            grid=(grid_m, fx.Index(h_q), 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_nsa_prefill_paged(*args, **kwargs)

    return _launch


def flydsl_nsa_prefill_paged_with_m_l(
    q:            "torch.Tensor",   # [T, h_q, 512] fp8
    kv_pool:      "torch.Tensor",   # [N_pages, BYTES_PER_PAGE] uint8 (raw C4 pool)
    indices:      "torch.Tensor",   # [T, TOPK] int32
    sm_scale:     float,
    c4_page_size: int   = 64,
    fp8_max:      float = 448.0,
) -> "tuple":
    """Fused paged gather + dequant + attention, returns raw online-softmax stats.

    kv_pool is the raw C4 pool buffer from token_to_kv_pool.get_extra_key_buffer().
    No intermediate tensors are allocated; gather and dequantization happen inside
    the FlyDSL kernel.

    Returns:
        out:   [T, h_q, 512]  bfloat16
        m_raw: [T, h_q]       float32  (running max, log2 space)
        l_raw: [T, h_q]       float32  (sum of exp2 values)
    """
    import torch

    total_tokens, h_q, head_dim = q.shape
    topk = indices.shape[1]

    # Pad to TILE_M=16 so each CTA can safely read/write its full 16-row tile
    _pad    = ((total_tokens + 15) // 16) * 16
    q_pad   = torch.zeros((_pad, h_q, head_dim), dtype=q.dtype, device=q.device)
    q_pad[:total_tokens] = q
    idx_pad = torch.zeros((_pad, topk), dtype=indices.dtype, device=indices.device)
    idx_pad[:total_tokens] = indices

    out   = torch.empty((_pad, h_q, head_dim), dtype=torch.bfloat16, device=q.device)
    out_m = torch.empty((_pad, h_q), dtype=torch.float32, device=q.device)
    out_l = torch.empty((_pad, h_q), dtype=torch.float32, device=q.device)

    kernel = build_nsa_prefill_paged_kernel(
        h_q=h_q,
        head_dim=head_dim,
        topk=topk,
        sm_scale=sm_scale,
        c4_page_size=c4_page_size,
        fp8_max=fp8_max,
    )
    stream = torch.cuda.current_stream()
    kernel(
        q_pad, kv_pool, idx_pad, out, out_m, out_l,
        total_tokens=total_tokens,
        stream=fx.Stream(stream.cuda_stream),
    )
    return out[:total_tokens], out_m[:total_tokens], out_l[:total_tokens]


# ──────────────────────────────────────────────────────────────────────────────
# Legacy API — kept for backward compatibility with tilelang_kernel.py paths.
# These wrappers use a flat pre-quantized KV tensor (old format, not paged pool).
# They are NOT used in the production backend path.
# ──────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=64)
def build_nsa_prefill_kernel(
    h_q: int,
    head_dim: int = 512,
    topk: int = 2048,
    tile_m: int = 16,
    block_n: int = 32,
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
):
    """Legacy builder: flat [num_pages, head_dim] fp8 KV tensor.

    NOT used in the production backend.  Kept for compatibility with test paths
    in tilelang_kernel.py that pass a flat pre-quantized KV tensor.
    """
    assert head_dim == 512
    assert tile_m == 16
    assert block_n == 32
    assert topk % block_n == 0

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    gpu_arch = get_rocm_arch()

    WARP_SIZE  = 64
    TILE_M     = tile_m
    BLOCK_N    = block_n
    HEAD_DIM   = head_dim
    TOPK       = topk
    BLOCK_SIZE = WARP_SIZE

    MFMA_N      = 16
    MFMA_K      = 32
    K_STEPS_QK  = HEAD_DIM // MFMA_K
    N_BLKS_S    = BLOCK_N  // MFMA_N
    D_BLKS      = HEAD_DIM // MFMA_N
    KV_TILES    = TOPK // BLOCK_N

    D_HALF   = HEAD_DIM // 2
    D_CHUNKS = D_HALF // 8

    LDS_KV_I64    = BLOCK_N * HEAD_DIM // 8
    LDS_KV_BYTES  = LDS_KV_I64 * 8
    LDS_P_F32     = TILE_M * BLOCK_N
    LDS_P_BYTES   = LDS_P_F32 * 4
    LDS_TOTAL     = LDS_KV_BYTES + LDS_P_BYTES

    Q_STOK  = h_q * HEAD_DIM
    IDX_STR = TOPK

    alloc      = SmemAllocator(None, arch=gpu_arch, global_sym_name="nsa_smem_flat")
    base_off   = alloc._align(alloc.ptr, 16)
    alloc.ptr  = base_off + LDS_TOTAL
    kv_lds_off = base_off
    p_lds_off  = base_off + LDS_KV_BYTES

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def nsa_prefill_kernel(
        Q:       fx.Tensor,
        KV:      fx.Tensor,
        Indices: fx.Tensor,
        Out:     fx.Tensor,
        M_Raw:   fx.Tensor,
        L_Raw:   fx.Tensor,
        total_tokens: fx.Int32,
    ):
        v4f32_type = Vec.make_type(4, fx.Float32)
        fm = arith.FastMathFlags.fast
        def _fadd(a, b): return arith.addf(_raw(a), _raw(b), fastmath=fm)
        def _fmul(a, b): return arith.mulf(_raw(a), _raw(b), fastmath=fm)
        def _fsub(a, b): return arith.subf(_raw(a), _raw(b), fastmath=fm)
        def _fmax(a, b): return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm).result

        c_neg_inf    = fx.Float32(float("-inf"))
        c_zero_f     = fx.Float32(0.0)
        c_one_f      = fx.Float32(1.0)
        c_sm_log2e   = fx.Float32(sm_scale * _LOG2E)
        c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)

        def _ptr_ty():  return ir.Type.parse("!llvm.ptr")
        def _as_ptr(t):
            v = t
            if hasattr(v, "ir_value") and not isinstance(v, ir.Value):
                v = v.ir_value()
            return _fly.extract_aligned_pointer_as_index(_ptr_ty(), v)

        q_ptr   = _as_ptr(Q)
        kv_ptr  = _as_ptr(KV)
        idx_ptr = _as_ptr(Indices)
        out_ptr = _as_ptr(Out)
        m_ptr   = _as_ptr(M_Raw)
        l_ptr   = _as_ptr(L_Raw)

        def load_8bytes_as_i64(ptr, byte_off_index):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(byte_off_index), elem_type=T.i8)
            return _llvm.LoadOp(T.i64, gep).result

        def load_i32_elem(ptr, elem_off_index):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(elem_off_index), elem_type=T.i32)
            return _llvm.LoadOp(T.i32, gep).result

        def mfma_fp8(acc, a_i64, b_i64):
            return rocdl.mfma_f32_16x16x32_fp8_fp8(v4f32_type, [a_i64, b_i64, acc, 0, 0, 0])

        def fp8_byte_to_f32(byte_i32):
            return rocdl.cvt_f32_fp8(T.f32, byte_i32, fx.Int32(0))

        tid     = fx.Index(gpu.thread_idx.x)
        bid_m   = fx.Index(gpu.block_idx.x)
        bid_h   = fx.Index(gpu.block_idx.y)
        lane    = tid % 16
        k_group = tid // 16
        q_start = bid_m * fx.Index(TILE_M)

        lane_mod8      = lane % fx.Index(8)
        lane_div8      = lane // fx.Index(8)
        lane_shift_i64 = fx.Int64(lane_mod8) * fx.Int64(8)

        lds_base   = alloc.get_base()
        lds_kv_i64 = SmemPtr(lds_base, kv_lds_off, T.i64, shape=(LDS_KV_I64,)).get()
        lds_p_f32  = SmemPtr(lds_base, p_lds_off,  T.f32, shape=(LDS_P_F32,)).get()

        q_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_byte_off = (
                (q_start + lane) * fx.Index(Q_STOK)
                + bid_h * fx.Index(HEAD_DIM)
                + fx.Index(ks * MFMA_K)
                + k_group * fx.Index(8)
            )
            q_packs.append(load_8bytes_as_i64(q_ptr, q_byte_off))

        _init = (
            [_raw(c_neg_inf)] * 4
            + [_raw(c_zero_f)]  * 4
            + [_raw(c_zero_v4f32)] * D_BLKS
        )
        anchor_tok = q_start

        loop_results = _init
        for kv_tile, _carry in range(0, KV_TILES, 1, init=_init):
            m_run = [_carry[r]     for r in range_constexpr(4)]
            l_run = [_carry[4 + r] for r in range_constexpr(4)]
            o_acc = [_carry[8 + d] for d in range_constexpr(D_BLKS)]

            kv_pos_base = kv_tile * fx.Index(BLOCK_N)
            kv_row  = tid % fx.Index(BLOCK_N)
            d_group = tid // fx.Index(BLOCK_N)

            idx_flat = anchor_tok * fx.Index(IDX_STR) + kv_pos_base + kv_row
            page_i32 = load_i32_elem(idx_ptr, idx_flat)
            page_idx = fx.Index(page_i32)

            for dc in range_constexpr(D_CHUNKS):
                d_byte_in_half = fx.Index(dc * 8)
                d_byte_abs     = d_group * fx.Index(D_HALF) + d_byte_in_half
                kv_byte_off    = page_idx * fx.Index(HEAD_DIM) + d_byte_abs
                raw_i64        = load_8bytes_as_i64(kv_ptr, kv_byte_off)
                i64_off = (
                    kv_row * fx.Index(HEAD_DIM // 8)
                    + d_group * fx.Index(D_HALF // 8)
                    + fx.Index(dc)
                )
                _memref.store(raw_i64, lds_kv_i64, [_raw(i64_off)])

            gpu.barrier()

            s_acc = [_raw(c_zero_v4f32) for _ in range(N_BLKS_S)]
            for ks in range_constexpr(K_STEPS_QK):
                q_a = q_packs[ks]
                for nb in range_constexpr(N_BLKS_S):
                    k_i64_off = (
                        (fx.Index(nb * MFMA_N) + lane) * fx.Index(HEAD_DIM // 8)
                        + fx.Index(ks * 4)
                        + k_group
                    )
                    k_pack    = _memref.load(lds_kv_i64, [_raw(k_i64_off)])
                    s_acc[nb] = mfma_fp8(s_acc[nb], q_a, k_pack)

            s_scaled = []
            for nb in range_constexpr(N_BLKS_S):
                sv = Vec(s_acc[nb])
                s_scaled.append([_fmul(sv[r], c_sm_log2e) for r in range_constexpr(4)])

            local_max = [_fmax(s_scaled[0][r], s_scaled[1][r]) for r in range_constexpr(4)]
            shfl_w  = fx.Int32(WARP_SIZE)
            row_max = list(local_max)
            for xor_off in [8, 4, 2, 1]:
                so = fx.Int32(xor_off)
                for r in range_constexpr(4):
                    row_max[r] = _fmax(row_max[r], fx.Float32(row_max[r]).shuffle_xor(so, shfl_w))

            m_new = [_fmax(m_run[r], row_max[r]) for r in range_constexpr(4)]
            corr  = [rocdl.exp2(T.f32, _raw(_fsub(m_run[r], m_new[r]))) for r in range_constexpr(4)]

            p_vals   = [[None] * 4 for _ in range(N_BLKS_S)]
            tile_sum = [_raw(c_zero_f) for _ in range(4)]
            for nb in range_constexpr(N_BLKS_S):
                for r in range_constexpr(4):
                    p_val = rocdl.exp2(T.f32, _raw(_fsub(s_scaled[nb][r], m_new[r])))
                    p_vals[nb][r] = p_val
                    tile_sum[r]   = _fadd(tile_sum[r], p_val)

            for xor_off in [8, 4, 2, 1]:
                so = fx.Int32(xor_off)
                for r in range_constexpr(4):
                    tile_sum[r] = _fadd(tile_sum[r], fx.Float32(tile_sum[r]).shuffle_xor(so, shfl_w))

            l_new = [_fadd(_fmul(corr[r], l_run[r]), tile_sum[r]) for r in range_constexpr(4)]

            for d in range_constexpr(D_BLKS):
                ov = Vec(o_acc[d])
                o_acc[d] = Vec.from_elements(
                    [_fmul(ov[r], corr[r]) for r in range_constexpr(4)], fx.Float32
                ).ir_value()

            m_run = m_new
            l_run = l_new

            for nb in range_constexpr(N_BLKS_S):
                for r in range_constexpr(4):
                    p_row  = k_group * fx.Index(4) + fx.Index(r)
                    p_col  = fx.Index(nb * MFMA_N) + lane
                    p_off  = p_row * fx.Index(BLOCK_N) + p_col
                    _memref.store(p_vals[nb][r], lds_p_f32, [_raw(p_off)])

            gpu.barrier()

            p_base_off = lane * fx.Index(BLOCK_N) + k_group * fx.Index(8)
            p_f32_vals = [
                _memref.load(lds_p_f32, [_raw(p_base_off + fx.Index(j))])
                for j in range(8)
            ]

            i32_lo = rocdl.cvt_pk_fp8_f32(T.i32, p_f32_vals[1], p_f32_vals[0], fx.Int32(0), fx.Int32(0))
            i32_lo = rocdl.cvt_pk_fp8_f32(T.i32, p_f32_vals[3], p_f32_vals[2], i32_lo,       fx.Int32(1))
            i32_hi = rocdl.cvt_pk_fp8_f32(T.i32, p_f32_vals[5], p_f32_vals[4], fx.Int32(0), fx.Int32(0))
            i32_hi = rocdl.cvt_pk_fp8_f32(T.i32, p_f32_vals[7], p_f32_vals[6], i32_hi,       fx.Int32(1))

            lo64    = _mlir_arith.ExtUIOp(T.i64, i32_lo).result
            hi64    = _mlir_arith.ShLIOp(
                _mlir_arith.ExtUIOp(T.i64, i32_hi).result, _raw(fx.Int64(32))
            ).result
            p_a_i64 = _mlir_arith.OrIOp(lo64, hi64).result

            for d in range_constexpr(D_BLKS):
                hd_off_f    = fx.Index(d * 2) + lane_div8
                v_bytes_i32 = []
                for j in range_constexpr(8):
                    kv_i64_off = (
                        (k_group * fx.Index(8) + fx.Index(j)) * fx.Index(HEAD_DIM // 8)
                        + hd_off_f
                    )
                    v_i64  = _memref.load(lds_kv_i64, [_raw(kv_i64_off)])
                    v_sh   = _mlir_arith.ShRUIOp(_raw(v_i64), _raw(lane_shift_i64)).result
                    v_b64  = _mlir_arith.AndIOp(v_sh, _raw(fx.Int64(0xFF))).result
                    v_bytes_i32.append(_mlir_arith.TruncIOp(T.i32, v_b64).result)

                v_b_i64 = _mlir_arith.ExtUIOp(T.i64, v_bytes_i32[0]).result
                for j in range_constexpr(7):
                    bj = _mlir_arith.ShLIOp(
                        _mlir_arith.ExtUIOp(T.i64, v_bytes_i32[j + 1]).result,
                        _raw(fx.Int64((j + 1) * 8))
                    ).result
                    v_b_i64 = _mlir_arith.OrIOp(v_b_i64, bj).result

                o_acc[d] = mfma_fp8(o_acc[d], p_a_i64, v_b_i64)

            gpu.barrier()
            loop_results = yield list(m_run) + list(l_run) + list(o_acc)

        m_final_vals = [loop_results[r]     for r in range_constexpr(4)]
        l_final      = [loop_results[4 + r] for r in range_constexpr(4)]
        o_final      = [loop_results[8 + d] for d in range_constexpr(D_BLKS)]

        for d in range_constexpr(D_BLKS):
            ov = Vec(o_final[d])
            for r in range_constexpr(4):
                inv_l  = arith.divf(_raw(c_one_f), _raw(l_final[r]), fastmath=fm)
                o_norm = _fmul(ov[r], inv_l)
                o_bf16 = fx.Float32(o_norm).to(fx.BFloat16).ir_value()
                out_row = k_group * fx.Index(4) + fx.Index(r)
                out_col = fx.Index(d * MFMA_N) + lane
                out_idx = fx.Int64(
                    (q_start + out_row) * fx.Index(Q_STOK)
                    + bid_h * fx.Index(HEAD_DIM)
                    + out_col
                )
                out_gep = buffer_ops.get_element_ptr(out_ptr, out_idx, elem_type=T.bf16)
                _llvm.StoreOp(o_bf16, out_gep)

        _out_row = k_group * fx.Index(4)
        for r in range_constexpr(4):
            _lse_tok = q_start + _out_row + fx.Index(r)
            _lse_idx = fx.Int64(_lse_tok * fx.Index(h_q) + bid_h)
            _llvm.StoreOp(
                _raw(m_final_vals[r]),
                buffer_ops.get_element_ptr(m_ptr, _lse_idx, elem_type=T.f32),
            )
            _llvm.StoreOp(
                _raw(l_final[r]),
                buffer_ops.get_element_ptr(l_ptr, _lse_idx, elem_type=T.f32),
            )

    @flyc.jit
    def launch_nsa_prefill(
        Q: fx.Tensor, KV: fx.Tensor, Indices: fx.Tensor, Out: fx.Tensor,
        M_Raw: fx.Tensor, L_Raw: fx.Tensor,
        total_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalize()

        tokens_idx = fx.Index(total_tokens)
        grid_m     = (tokens_idx + TILE_M - 1) // TILE_M
        launcher   = nsa_prefill_kernel(Q, KV, Indices, Out, M_Raw, L_Raw, total_tokens)

        passthrough = []
        for pair in [
            ("denormal-fp-math-f32", "preserve-sign,preserve-sign"),
            ("no-nans-fp-math",      "true"),
            ("unsafe-fp-math",       "true"),
        ]:
            passthrough.append(
                ir.ArrayAttr.get([ir.StringAttr.get(pair[0]), ir.StringAttr.get(pair[1])])
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"]        = ir.ArrayAttr.get(passthrough)
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(waves_per_eu))

        launcher.launch(
            grid=(grid_m, fx.Index(h_q), 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_nsa_prefill(*args, **kwargs)

    return _launch


def flydsl_nsa_prefill(
    q:       "torch.Tensor",
    kv:      "torch.Tensor",
    indices: "torch.Tensor",
    sm_scale: float,
) -> "torch.Tensor":
    """Legacy: flat [num_pages, 512] fp8 KV tensor.  NOT for production use."""
    import torch

    total_tokens, h_q, head_dim = q.shape
    topk = indices.shape[1]
    _fp8_types = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    assert q.dtype in _fp8_types
    assert kv.dtype == q.dtype
    assert indices.dtype == torch.int32

    _pad    = ((total_tokens + 15) // 16) * 16
    q_pad   = torch.zeros((_pad, h_q, head_dim), dtype=q.dtype, device=q.device)
    q_pad[:total_tokens] = q
    idx_pad = torch.zeros((_pad, topk), dtype=indices.dtype, device=indices.device)
    idx_pad[:total_tokens] = indices
    out    = torch.empty((_pad, h_q, head_dim), dtype=torch.bfloat16, device=q.device)
    out_m  = torch.empty((_pad, h_q), dtype=torch.float32, device=q.device)
    out_l  = torch.empty((_pad, h_q), dtype=torch.float32, device=q.device)
    kernel = build_nsa_prefill_kernel(h_q=h_q, head_dim=head_dim, topk=topk, sm_scale=sm_scale)
    stream = torch.cuda.current_stream()
    kernel(q_pad, kv, idx_pad, out, out_m, out_l, total_tokens=total_tokens,
           stream=fx.Stream(stream.cuda_stream))
    return out[:total_tokens]


def flydsl_nsa_prefill_with_lse(
    q:        "torch.Tensor",
    kv:       "torch.Tensor",
    indices:  "torch.Tensor",
    sm_scale: float,
) -> "tuple":
    """Legacy: flat KV tensor, returns (out, lse).  NOT for production use."""
    import torch

    total_tokens, h_q, head_dim = q.shape
    topk = indices.shape[1]
    _fp8_types = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    assert q.dtype in _fp8_types
    assert kv.dtype == q.dtype
    assert indices.dtype == torch.int32

    _pad    = ((total_tokens + 15) // 16) * 16
    q_pad   = torch.zeros((_pad, h_q, head_dim), dtype=q.dtype, device=q.device)
    q_pad[:total_tokens] = q
    idx_pad = torch.zeros((_pad, topk), dtype=indices.dtype, device=indices.device)
    idx_pad[:total_tokens] = indices
    out    = torch.empty((_pad, h_q, head_dim), dtype=torch.bfloat16, device=q.device)
    out_m  = torch.empty((_pad, h_q), dtype=torch.float32, device=q.device)
    out_l  = torch.empty((_pad, h_q), dtype=torch.float32, device=q.device)
    kernel = build_nsa_prefill_kernel(h_q=h_q, head_dim=head_dim, topk=topk, sm_scale=sm_scale)
    stream = torch.cuda.current_stream()
    kernel(q_pad, kv, idx_pad, out, out_m, out_l, total_tokens=total_tokens,
           stream=fx.Stream(stream.cuda_stream))
    m   = out_m[:total_tokens]
    l   = out_l[:total_tokens]
    lse = m * _LN2 + torch.log(l)
    return out[:total_tokens], lse


def flydsl_nsa_prefill_with_m_l(
    q:        "torch.Tensor",
    kv:       "torch.Tensor",
    indices:  "torch.Tensor",
    sm_scale: float,
) -> "tuple":
    """Legacy: flat KV tensor, returns (out, m_raw, l_raw).  NOT for production use."""
    import torch

    total_tokens, h_q, head_dim = q.shape
    topk = indices.shape[1]
    _fp8_types = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    assert q.dtype in _fp8_types
    assert kv.dtype == q.dtype
    assert indices.dtype == torch.int32

    _pad    = ((total_tokens + 15) // 16) * 16
    q_pad   = torch.zeros((_pad, h_q, head_dim), dtype=q.dtype, device=q.device)
    q_pad[:total_tokens] = q
    idx_pad = torch.zeros((_pad, topk), dtype=indices.dtype, device=indices.device)
    idx_pad[:total_tokens] = indices
    out    = torch.empty((_pad, h_q, head_dim), dtype=torch.bfloat16, device=q.device)
    out_m  = torch.empty((_pad, h_q), dtype=torch.float32, device=q.device)
    out_l  = torch.empty((_pad, h_q), dtype=torch.float32, device=q.device)
    kernel = build_nsa_prefill_kernel(h_q=h_q, head_dim=head_dim, topk=topk, sm_scale=sm_scale)
    stream = torch.cuda.current_stream()
    kernel(q_pad, kv, idx_pad, out, out_m, out_l, total_tokens=total_tokens,
           stream=fx.Stream(stream.cuda_stream))
    return out[:total_tokens], out_m[:total_tokens], out_l[:total_tokens]
