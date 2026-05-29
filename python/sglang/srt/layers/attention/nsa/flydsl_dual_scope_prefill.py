"""Fused dual-scope sparse-attention prefill for DeepSeek V4 / NSA (gfx950).

This module replaces the broken two-kernel + Python-LSE-merge prefill path with
a SINGLE fused dual-scope kernel that attends over both the sliding-window (SWA,
"main") scope and the compressed (C4 / C128, "extra") scope in one shared
online-softmax pass.  It is the FlyDSL analogue of the authoritative Triton
reference ``_fused_gather_attn_dsv4_dual_scope_kernel`` in
``nsa/triton_decode/triton_mla_kernels_decode_fused.py``.

Components:
  * ``flydsl_dual_scope_prefill``      -- public entry matching the backend hook;
                                          the live prefill path on gfx950.
  * ``_build_dual_scope_kernel``       -- the FlyDSL kernel: gather + e4m3fn fp8
                                          dequant + QK + exp2 online softmax + PV
                                          over both scopes in one fused pass.
  * ``_flydsl_dual_scope_kernel_impl`` -- thin launcher that normalizes inputs,
                                          builds/caches the kernel, and launches it.
  * ``_torch_reference_dual_scope``    -- faithful pure-PyTorch correctness ORACLE
                                          used by the tests (NOT a live fallback).

The FlyDSL kernel is the live path (``_USE_FLYDSL_KERNEL = True``) and matches
both the PyTorch oracle and the production Triton kernel on synthetic, scaled,
and real captured tensors.  The FlyDSL import is lazy/guarded so this module
still imports cleanly on a CPU host where FlyDSL (a remote-container dependency)
is absent; the PyTorch oracle and the test harness run without FlyDSL present.

----------------------------------------------------------------------------
Paged KV pool byte layout (per block of ``block_size`` token slots), mirrored
from ``DeepSeekV4SingleKVPool.get_bytes_per_token`` and the Triton kernel:

    bytes [0 .. block_size*576)                : token DATA section (data-major)
        per slot s (0 .. block_size-1) -> 576 bytes at [s*576 .. (s+1)*576):
            nope : bytes [s*576       .. s*576+448)  -> 448 fp8 (e4m3) values
            rope : bytes [s*576+448   .. s*576+576)  -> 64 bf16 values (128 bytes, LE)
    bytes [block_size*576 .. block_size*584)   : token SCALE section
        per slot s -> 8 bytes at [block_size*576 + s*8 .. +8):
            7 x ue8m0 scale bytes (one per 64-dim nope tile) + 1 byte pad

Dequant (per gathered key token):
    nope_f32[d] = fp8_to_f32(nope_byte[d]) * exp2(scale_byte[d // 64] - 127)
    rope_f32[d] = bf16_to_f32(rope value d)
    kv[0:512]   = concat(nope_f32[0:448], rope_f32[0:64])          # K == V (MLA)
----------------------------------------------------------------------------
"""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import torch

# ----------------------------------------------------------------------------
# FlyDSL is a remote-container (gfx950) dependency.  Import its API surface at
# MODULE level -- guarded so the module still imports on a CPU host where FlyDSL
# is absent (the PyTorch reference path + CPU harness need no FlyDSL).
#
# IMPORTANT: these MUST be module globals (not imported inside the kernel
# builder).  When the FlyDSL names live in the builder's local scope they become
# closure free-vars of the @flyc.kernel function (and its nested _as_ptr helper),
# which trips FlyDSL's ASTRewriter freevar-count check
# ("requires N free vars, not N-1").  The known-good artefact kernels
# (flydsl_nsa_prefill.py / flydsl_swa_prefill.py) import at module level for
# exactly this reason; we mirror that import surface here.
# ----------------------------------------------------------------------------
try:  # pragma: no cover - exercised only on the gfx950 box
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

    _HAS_FLYDSL = True
except Exception:  # FlyDSL absent (CPU host): reference path + harness still run
    _HAS_FLYDSL = False

# ----------------------------------------------------------------------------
# Layout / dim constants (must stay in sync with the pool + Triton reference).
# ----------------------------------------------------------------------------
D_QK: int = 512                 # query / key head dim (448 nope + 64 rope)
D_V: int = 512                  # value (output) head dim
D_NOPE: int = 448               # fp8 nope dims
D_ROPE: int = 64                # bf16 rope dims
QUANT_BLOCK: int = 64           # nope dims per ue8m0 scale tile
N_NOPE_TILES: int = D_NOPE // QUANT_BLOCK            # 7
BYTES_PER_TOKEN_DATA: int = D_NOPE + D_ROPE * 2      # 448 + 128 = 576
BYTES_PER_TOKEN_SCALE: int = 8                       # 7 ue8m0 + 1 pad

_LOG2E: float = 1.4426950408889634

# Internal flag: when True (the production default) the public entry routes
# prefill through the live FlyDSL dual-scope kernel
# (_flydsl_dual_scope_kernel_impl). When False the public entry uses the
# pure-PyTorch oracle (_torch_reference_dual_scope) -- this branch exists for
# the tests/oracle path only, not as a runtime fallback.
_USE_FLYDSL_KERNEL: bool = True


# ============================================================================
# Shape / dtype normalization helpers
# ============================================================================
def _normalize_q(q: torch.Tensor) -> torch.Tensor:
    """Backend passes q as [T, 1, H, D_QK] (after unsqueeze(1)) or [T, H, D_QK].

    Returns a contiguous [T, H, D_QK] tensor.
    """
    if q.ndim == 4:
        # [T, s_q=1, H, D] -> squeeze the singleton seq dim.
        assert q.shape[1] == 1, f"unexpected q seq dim: {q.shape}"
        q = q.squeeze(1)
    assert q.ndim == 3, f"q must be 3D after normalize, got {q.shape}"
    assert q.shape[-1] == D_QK, f"q head dim {q.shape[-1]} != {D_QK}"
    return q.contiguous()


def _normalize_indices(indices: torch.Tensor) -> torch.Tensor:
    """Backend passes indices as [T, 1, topk] (after unsqueeze(1)) or [T, topk]."""
    if indices.ndim == 3:
        assert indices.shape[1] == 1, f"unexpected indices seq dim: {indices.shape}"
        indices = indices.squeeze(1)
    assert indices.ndim == 2, f"indices must be 2D after normalize, got {indices.shape}"
    return indices.contiguous()


def _block_layout(kv_cache: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Reinterpret a paged KV cache as a flat data-major uint8 tensor.

    ``kv_cache`` is the 4D view the backend builds:
        [num_blocks, block_size, 1, kv_cache_total_dim]  (uint8, total_dim == 584)
    The backend already sliced off page padding before viewing, so the per-block
    byte run is exactly ``block_size * 584`` and data-major.

    Returns ``(u8_flat[num_blocks, block_size*584], num_blocks, block_size)``.
    """
    num_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    u8 = kv_cache.contiguous().view(torch.uint8).reshape(num_blocks, -1)
    per_block_bytes = u8.shape[1]
    expected = block_size * (BYTES_PER_TOKEN_DATA + BYTES_PER_TOKEN_SCALE)
    assert per_block_bytes == expected, (
        f"per-block bytes {per_block_bytes} != expected {expected} "
        f"(block_size={block_size}); pool layout assumption violated"
    )
    return u8, num_blocks, block_size


# ============================================================================
# Pure-PyTorch dequant of a gathered scope (byte-unpacking path == oracle)
# ============================================================================
def _dequant_scope(
    kv_cache: torch.Tensor,
    indices: torch.Tensor,          # [T, topk] int
    topk_length: Optional[torch.Tensor],  # [batch] int, or None
    s_q: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather + dequant one scope into dense bf16 KV and a validity mask.

    Returns:
        kv:    [T, topk, D_QK] float32  (dequantized key == value; invalid -> 0)
        valid: [T, topk]       bool     (False for idx==-1 or beyond topk_length)
    """
    device = kv_cache.device
    T, topk = indices.shape
    u8, num_blocks, block_size = _block_layout(kv_cache)
    u8_flat = u8.reshape(-1)  # [num_blocks * per_block_bytes]
    per_block_bytes = u8.shape[1]

    idx = indices.to(torch.int64)

    # Validity: idx == -1 sentinel, plus optional per-batch topk_length cutoff.
    valid = idx != -1
    if topk_length is not None:
        n_batch = topk_length.shape[0]
        sq = max(1, s_q)
        batch_of_t = torch.div(
            torch.arange(T, device=device), sq, rounding_mode="floor"
        ).clamp_(max=n_batch - 1)
        # per-token cutoff length, broadcast over the topk axis
        cutoff = topk_length.to(torch.int64)[batch_of_t]  # [T]
        col = torch.arange(topk, device=device).unsqueeze(0)  # [1, topk]
        valid = valid & (col < cutoff.unsqueeze(1))

    idx_c = idx.clamp_(min=0)
    block = torch.div(idx_c, block_size, rounding_mode="floor")  # [T, topk]
    off = idx_c - block * block_size

    # Absolute byte base of each gathered token's DATA and SCALE sections.
    block_base = block * per_block_bytes
    data_base = block_base + off * BYTES_PER_TOKEN_DATA              # [T, topk]
    scale_base = block_base + block_size * BYTES_PER_TOKEN_DATA + off * BYTES_PER_TOKEN_SCALE

    # --- nope: 448 fp8 bytes -> float32, scaled per 64-dim tile -----------------
    nope_off = data_base.unsqueeze(-1) + torch.arange(D_NOPE, device=device)  # [T,topk,448]
    nope_u8 = u8_flat[nope_off.reshape(-1)].reshape(T, topk, D_NOPE)
    nope_f = nope_u8.view(torch.float8_e4m3fn).to(torch.float32)

    scale_idx = scale_base.unsqueeze(-1) + torch.arange(N_NOPE_TILES, device=device)
    scale_u8 = u8_flat[scale_idx.reshape(-1)].reshape(T, topk, N_NOPE_TILES)
    scale_f = torch.exp2(scale_u8.to(torch.float32) - 127.0)  # ue8m0 -> f32
    # broadcast each tile scale over its 64 dims
    nope_dq = (
        nope_f.reshape(T, topk, N_NOPE_TILES, QUANT_BLOCK)
        * scale_f.unsqueeze(-1)
    ).reshape(T, topk, D_NOPE)

    # --- rope: 128 raw bytes -> 64 bf16 -> float32 -----------------------------
    rope_off = (
        data_base.unsqueeze(-1) + D_NOPE + torch.arange(D_ROPE * 2, device=device)
    )
    rope_u8 = u8_flat[rope_off.reshape(-1)].reshape(T, topk, D_ROPE * 2)
    rope_f = rope_u8.view(torch.bfloat16).to(torch.float32).reshape(T, topk, D_ROPE)

    kv = torch.cat([nope_dq, rope_f], dim=-1)  # [T, topk, 512]
    kv = torch.where(valid.unsqueeze(-1), kv, torch.zeros_like(kv))
    return kv, valid


# ============================================================================
# Faithful pure-PyTorch dual-scope reference (the correctness oracle)
# ============================================================================
def _torch_reference_dual_scope(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_length: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    compress_ratio: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
) -> torch.Tensor:
    """Dense-equivalent of the Triton dual-scope online-softmax kernel.

    Mirrors ``_fused_gather_attn_dsv4_dual_scope_kernel`` exactly (numerically
    identical up to bf16-matmul rounding):

      * gather + dequant SWA (main) and C4/C128 (extra) keys from the paged pool,
      * QK with K == V (MLA latent), score *= softmax_scale,
      * mask invalid / padded slots (idx==-1 or beyond topk_length) to -inf
        BEFORE softmax so zero-padding never pollutes the denominator,
      * one shared softmax over [SWA ++ extra] with the attn_sink folded into the
        denominator (denom = sum exp(score - m) + exp(attn_sink - m)),
      * lonely queries (no valid key) emit a zero row,
      * output is the probability-weighted sum of V, cast to bf16.

    The Triton kernel does this in log2 space with exp2 and a running (m, l, acc)
    state processed SWA-first then extra; an online softmax over a fixed key set
    is mathematically identical to the dense masked softmax computed here, which
    is easier to read and audit.  Returns [T, H, head_dim_v] bf16.
    """
    q3 = _normalize_q(q)                       # [T, H, 512]
    device = q3.device
    T, H, _ = q3.shape
    qf = q3.to(torch.bfloat16).to(torch.float32)

    swa_idx = _normalize_indices(swa_indices)  # [T, topk_m]
    s_q_main = max(1, T // swa_topk_length.shape[0]) if swa_topk_length is not None else 1
    kv_main, valid_main = _dequant_scope(
        swa_k_cache, swa_idx, swa_topk_length, s_q_main
    )

    kv_list = [kv_main]
    valid_list = [valid_main]

    if extra_k_cache is not None and extra_indices is not None:
        extra_idx = _normalize_indices(extra_indices)
        s_q_extra = (
            max(1, T // extra_topk_length.shape[0])
            if extra_topk_length is not None
            else 1
        )
        kv_extra, valid_extra = _dequant_scope(
            extra_k_cache, extra_idx, extra_topk_length, s_q_extra
        )
        kv_list.append(kv_extra)
        valid_list.append(valid_extra)

    kv = torch.cat(kv_list, dim=1)        # [T, K, 512]
    valid = torch.cat(valid_list, dim=1)  # [T, K]

    # QK^T in bf16 (mirrors the bf16 MFMA path), accumulate in float32.
    kv_bf = kv.to(torch.bfloat16).to(torch.float32)
    scores = torch.einsum("thd,tkd->thk", qf, kv_bf) * float(softmax_scale)  # [T,H,K]

    neg_inf = float("-inf")
    valid_h = valid.unsqueeze(1)  # [T, 1, K]
    scores = torch.where(valid_h, scores, torch.full_like(scores, neg_inf))

    m = scores.amax(dim=-1)                          # [T, H]
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    p = torch.where(
        torch.isinf(scores), torch.zeros_like(scores),
        torch.exp(scores - m_safe.unsqueeze(-1)),
    )                                                # [T, H, K]
    l = p.sum(dim=-1)                                # [T, H]
    acc = torch.einsum("thk,tkd->thd", p, kv_bf)     # [T, H, 512]

    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).reshape(1, H)            # [1, H]
        denom = l + torch.exp(sink - m_safe)
    else:
        denom = l

    lonely = l == 0.0
    denom = torch.where(denom == 0.0, torch.ones_like(denom), denom)
    out = acc / denom.unsqueeze(-1)
    out = torch.where(lonely.unsqueeze(-1), torch.zeros_like(out), out)

    out = out[..., :head_dim_v].to(torch.bfloat16).contiguous()  # [T, H, head_dim_v]
    return out


# ============================================================================
# FlyDSL fused dual-scope kernel (live prefill path): gather + fp8 dequant +
# QK + exp2 online softmax + PV over SWA (main) and C4/C128 (extra) in one pass.
# ============================================================================
def _flydsl_available() -> bool:
    """True only on a gfx950 host with FlyDSL importable."""
    if not _HAS_FLYDSL:
        return False
    try:
        return str(get_rocm_arch()).startswith("gfx950")
    except Exception:
        return False


@functools.lru_cache(maxsize=64)
def _build_dual_scope_kernel(
    h_q: int,
    head_dim: int = D_QK,
    head_dim_v: int = D_V,
    topk_main: int = 2048,
    topk_extra: int = 2048,
    block_size_main: int = 64,
    block_size_extra: int = 64,
    tile_m: int = 16,
    block_n: int = 32,
    block_h: int = 16,
    sm_scale: Optional[float] = None,
    has_attn_sink: bool = True,
    has_topk_length_main: bool = True,
    has_topk_length_extra: bool = True,
    waves_per_eu: int = 2,
):
    """Build (and cache) the fused dual-scope FlyDSL kernel.

    The lru_cache key is the full set of compile-time-shaping args (dtype is
    fixed bf16/fp8, dims/topk/block sizes/BLOCK_H/sink/length flags select a
    distinct kernel specialization).  Returns a python launcher.

    The kernel implements the full dual-scope inner math: per-tile gather +
    e4m3fn fp8 dequant into LDS, bf16-MFMA QK, exp2 online softmax with running
    (m, l, acc) state shared across the SWA then EXTRA scopes, and bf16-MFMA PV,
    finalized with attn_sink folding and a lonely-query zero row.

    FlyDSL is imported at MODULE level (guarded); this builder is only reachable
    on the gfx950 box where the import succeeds.
    """
    assert _HAS_FLYDSL, "FlyDSL is not importable (build is gfx950-container only)"
    assert head_dim == D_QK, f"only head_dim={D_QK} is supported"
    assert tile_m == 16 and block_n == 32, "MFMA tiling is fixed at 16x32"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    gpu_arch = get_rocm_arch()

    # ---- tiling -----------------------------------------------------------
    # One CTA per query token (grid.y); BLOCK_H heads tiled in grid.x.  The
    # 16x16x32 bf16 MFMA M-dimension is the BLOCK_H=16 heads of this token
    # (NOT 16 tokens like the artefacts), the N-dimension is BLOCK_N keys, the
    # K-dimension is head_dim.  topk_length is read 1:1 per query token.
    WARP_SIZE = 64
    MFMA_N = 16
    MFMA_K = 32
    BLOCK_SIZE = WARP_SIZE
    K_STEPS_QK = head_dim // MFMA_K     # 16
    N_BLKS_S = block_n // MFMA_N        # 2
    D_BLKS = head_dim // MFMA_N         # 32
    KV_TILES_MAIN = topk_main // block_n
    KV_TILES_EXTRA = topk_extra // block_n

    BLOCK_H = block_h
    assert BLOCK_H == tile_m == MFMA_N, "BLOCK_H must equal MFMA M=16"
    TOK_OUT_STRIDE = h_q * head_dim_v               # Out token stride (elems)
    HEAD_OUT_STRIDE = head_dim_v                    # Out head stride (elems)
    Q_STOK = h_q * head_dim                         # Q token stride (elems)

    # ---- pool byte geometry (data-major: 576 data + 8 scale per slot) -----
    SLOT_DATA_BYTES = D_NOPE + D_ROPE * 2           # 576
    SLOT_SCALE_BYTES = BYTES_PER_TOKEN_SCALE        # 8
    PBB_MAIN = block_size_main * (SLOT_DATA_BYTES + SLOT_SCALE_BYTES)
    PBB_EXTRA = block_size_extra * (SLOT_DATA_BYTES + SLOT_SCALE_BYTES)
    NOPE_HALF = D_NOPE // 2                          # 224
    NOPE_CHUNKS = NOPE_HALF // 8                     # 28
    NOPE_HALF_I64 = NOPE_HALF // 8                    # 28 (i64 per gather half)
    ROPE_HALF = D_ROPE // 2                          # 32
    ROPE_CHUNKS = ROPE_HALF // 8                     # 4

    # Compact-LDS / dequant-on-read split (Phase2 Stage1).  The nope dims live in
    # LDS as raw fp8 packed 8-per-i64; rope dims stay bf16; the per-64-tile ue8m0
    # scales are pre-converted to f32 and cached.  Because the 8x32 MFMA K-step
    # (MFMA_K=32) and 16-wide N/d-block both divide the 448-dim nope boundary,
    # nope-vs-rope is a pure COMPILE-TIME split on the loop index:
    #   QK : ks  < KS_NOPE (=14)        -> nope ; else rope     (D_ROPE//MFMA_K=2)
    #   PV : d   < D_NOPE_BLKS (=28)    -> nope ; else rope      (D_ROPE//MFMA_N=4)
    NOPE_I64 = D_NOPE // 8                            # 56 (i64 per key, nope)
    KS_NOPE = D_NOPE // MFMA_K                        # 14 (QK k-steps that are nope)
    D_NOPE_BLKS = D_NOPE // MFMA_N                    # 28 (PV d-blocks that are nope)

    SM_LOG2E = float(sm_scale) * _LOG2E
    BIG = 1.0e30                                     # finite mask sentinel

    # ---- LDS layout (compact: store KV in its packed pool form) -----------
    # Mirrors the proven-fast single-scope sibling kernel: KV stays fp8 in LDS
    # (8 fp8 per i64) and is DEQUANTIZED ON READ in the QK/PV MFMA loops, instead
    # of dequantizing the whole tile to bf16 up front.  This drops the KV LDS
    # footprint from 32768 B (bf16 [block_n, head_dim]) to ~14.8 KB so the kernel
    # fits ~3 workgroups/CU on gfx950 (64 KB LDS/CU) instead of 1.
    #
    # The packed ue8m0 scales are carried in the SAME i64 buffer as the nope fp8
    # (not a separate f32 memref): a separate small f32 scale memref read inside
    # the runtime scf.for tile loop is mishandled by FlyDSL/MLIR (the load is
    # reordered/hoisted relative to its store across gpu.barrier(), producing a
    # read-before-write RACE).  Reads from the SAME i64 memref as the proven nope
    # path are race-free, so we pack the 7 ue8m0 bytes/key into one extra i64 and
    # decode (exp2) on read into per-thread registers.
    #   KV nope+scale : i64 memref
    #       [0, block_n*56)               raw fp8 nope (8 fp8/i64), 56 i64/key
    #       [block_n*56, block_n*57)      packed 7 ue8m0 bytes/key (read region)
    #       [block_n*57, block_n*58)      g_half==1 scratch half (never read)
    #   KV rope       : bf16 memref, raw [block_n, D_ROPE]
    #   P             : f32 memref, attn weights [BLOCK_H, block_n]
    LDS_NOPE_CHUNKS = block_n * NOPE_I64             # 32*56 = 1792 i64 (nope fp8)
    SCALE_BASE_I64 = LDS_NOPE_CHUNKS                  # scale sub-region offset
    LDS_SCALE_I64 = block_n * 2                       # 64 i64 (real 32 + scratch 32)
    LDS_KV_NOPE_I64 = LDS_NOPE_CHUNKS + LDS_SCALE_I64  # 1856 i64
    LDS_KV_NOPE_BYTES = LDS_KV_NOPE_I64 * 8          # 14848
    LDS_KV_ROPE_BF16 = block_n * D_ROPE              # 32*64 = 2048 bf16
    LDS_KV_ROPE_BYTES = LDS_KV_ROPE_BF16 * 2         # 4096
    LDS_P_F32 = BLOCK_H * block_n                    # 512
    LDS_P_BYTES = LDS_P_F32 * 4                      # 2048
    LDS_TOTAL = (LDS_KV_NOPE_BYTES + LDS_KV_ROPE_BYTES + LDS_P_BYTES)  # 20992

    alloc = SmemAllocator(None, arch=gpu_arch, global_sym_name="dual_scope_smem")
    base_off = alloc._align(alloc.ptr, 16)
    alloc.ptr = base_off + LDS_TOTAL
    kv_nope_lds_off = base_off
    kv_rope_lds_off = kv_nope_lds_off + LDS_KV_NOPE_BYTES
    p_lds_off = kv_rope_lds_off + LDS_KV_ROPE_BYTES

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def dual_scope_prefill_kernel(
        Q: fx.Tensor,             # [T, h_q, head_dim]            bf16
        KV_Main: fx.Tensor,       # [Nb_main, per_block_bytes]    uint8 raw pool
        Indices_Main: fx.Tensor,  # [T, topk_main]                int32
        TopkLen_Main: fx.Tensor,  # [T]                           int32 (1:1 per tok)
        KV_Extra: fx.Tensor,      # [Nb_extra, per_block_bytes]   uint8 raw pool
        Indices_Extra: fx.Tensor, # [T, topk_extra]               int32
        TopkLen_Extra: fx.Tensor, # [T]                           int32 (1:1 per tok)
        AttnSink: fx.Tensor,      # [h_q]                         f32
        Out: fx.Tensor,           # [T, h_q, head_dim_v]          bf16
        total_tokens: fx.Int32,
    ):
        # ----------------------------------------------------------------
        # LLVM pointer helpers (mirror the artefact kernels exactly).
        # ----------------------------------------------------------------
        def _ptr_ty():
            return ir.Type.parse("!llvm.ptr")

        def _as_ptr(t):
            v = t
            if hasattr(v, "ir_value") and not isinstance(v, ir.Value):
                v = v.ir_value()
            return _fly.extract_aligned_pointer_as_index(_ptr_ty(), v)

        q_ptr           = _as_ptr(Q)
        out_ptr         = _as_ptr(Out)
        sink_ptr        = _as_ptr(AttnSink)
        main_pool_ptr   = _as_ptr(KV_Main)
        main_idx_ptr    = _as_ptr(Indices_Main)
        main_tkl_ptr    = _as_ptr(TopkLen_Main)
        extra_pool_ptr  = _as_ptr(KV_Extra)
        extra_idx_ptr   = _as_ptr(Indices_Extra)
        extra_tkl_ptr   = _as_ptr(TopkLen_Extra)

        # ---- typed constants / fast-math wrappers ----------------------
        v4f32_type  = Vec.make_type(4, fx.Float32)
        vec8bf16_ty = Vec.make_type(8, fx.BFloat16)
        fm = arith.FastMathFlags.fast

        def _fadd(a, b): return arith.addf(_raw(a), _raw(b), fastmath=fm)
        def _fmul(a, b): return arith.mulf(_raw(a), _raw(b), fastmath=fm)
        def _fsub(a, b): return arith.subf(_raw(a), _raw(b), fastmath=fm)
        def _fmax(a, b): return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm).result
        def _fmin(a, b): return arith.MinNumFOp(_raw(a), _raw(b), fastmath=fm).result
        def _exp2(x):    return rocdl.exp2(T.f32, _raw(x))

        c_one  = fx.Float32(1.0)
        c_big  = fx.Float32(BIG)
        c_log2e = fx.Float32(_LOG2E)
        c_sm_log2e = fx.Float32(SM_LOG2E)

        # ---- load helpers ----------------------------------------------
        def load_i32(ptr, eoff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(eoff), elem_type=T.i32)
            return _llvm.LoadOp(T.i32, gep).result

        def load_bf16_elem(ptr, eoff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(eoff), elem_type=T.bf16)
            return _llvm.LoadOp(T.bf16, gep).result

        def load_f32_elem(ptr, eoff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(eoff), elem_type=T.f32)
            return _llvm.LoadOp(T.f32, gep).result

        def load_u8_i32(ptr, boff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(boff), elem_type=T.i8)
            u8 = _llvm.LoadOp(T.i8, gep).result
            return _mlir_arith.ExtUIOp(T.i32, u8).result

        def load_i64(ptr, boff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(boff), elem_type=T.i8)
            return _llvm.LoadOp(T.i64, gep).result

        def load_bf16_byte(ptr, boff):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(boff), elem_type=T.i8)
            return _llvm.LoadOp(T.bf16, gep).result

        def extract_byte_i32(v_i64, j):
            sh  = _mlir_arith.ShRUIOp(v_i64, _raw(fx.Int64(j * 8))).result
            msk = _mlir_arith.AndIOp(sh, _raw(fx.Int64(0xFF))).result
            return _mlir_arith.TruncIOp(T.i32, msk).result

        def extract_byte_i32_dyn(v_i64, shift_i64):
            # Runtime byte select (PV reads byte lane%8 of a packed i64).
            sh  = _mlir_arith.ShRUIOp(_raw(v_i64), _raw(shift_i64)).result
            msk = _mlir_arith.AndIOp(sh, _raw(fx.Int64(0xFF))).result
            return _mlir_arith.TruncIOp(T.i32, msk).result

        def fp8_to_f32(byte_i32):
            return rocdl.cvt_f32_fp8(T.f32, byte_i32, fx.Int32(0))

        def ue8m0_to_f32(u8_i32):
            f = _mlir_arith.UIToFPOp(T.f32, u8_i32).result
            return _exp2(_fsub(f, fx.Float32(127.0)))

        def to_bf16(f32val):
            return fx.Float32(f32val).to(fx.BFloat16)

        def mfma_bf16(acc, a, b):
            return rocdl.mfma_f32_16x16x32_bf16(v4f32_type, [a, b, acc, 0, 0, 0])

        # ---- thread / block ids ----------------------------------------
        #   grid = (cdiv(h_q, BLOCK_H), total_tokens); block = (BLOCK_SIZE,1,1).
        tid        = fx.Index(gpu.thread_idx.x)
        bid_h_tile = fx.Index(gpu.block_idx.x)
        pid_t      = fx.Index(gpu.block_idx.y)

        lane     = tid % fx.Index(MFMA_N)        # 0..15 : MFMA row(A)/col(B/C)
        k_group  = tid // fx.Index(MFMA_N)       # 0..3  : MFMA K-subgroup
        head_base = bid_h_tile * fx.Index(BLOCK_H)
        kv_row   = tid % fx.Index(block_n)       # 0..31 : gather key row
        g_half   = tid // fx.Index(block_n)      # 0/1   : gather head-dim half

        # ---- LDS (compact KV: fp8-i64 nope + bf16 rope + f32 scales) ----
        lds_base = alloc.get_base()
        lds_kv_nope = SmemPtr(
            lds_base, kv_nope_lds_off, T.i64, shape=(LDS_KV_NOPE_I64,)
        ).get()
        lds_kv_rope = SmemPtr(
            lds_base, kv_rope_lds_off, T.bf16, shape=(LDS_KV_ROPE_BF16,)
        ).get()
        lds_p  = SmemPtr(lds_base, p_lds_off,  T.f32,  shape=(LDS_P_F32,)).get()

        topklen_main_val  = load_i32(main_tkl_ptr,  pid_t)
        topklen_extra_val = load_i32(extra_tkl_ptr, pid_t)

        # ---- preload Q head tile into K_STEPS_QK x vector<8xbf16> -------
        # A-operand row m = lane -> head (head_base + lane); k_group selects
        # the 8 contraction dims [ks*32 + k_group*8 : +8].
        q_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_elems = []
            for j in range_constexpr(8):
                q_eoff = (
                    pid_t * fx.Index(Q_STOK)
                    + (head_base + lane) * fx.Index(head_dim)
                    + fx.Index(ks * MFMA_K + j) + k_group * fx.Index(8)
                )
                q_elems.append(load_bf16_elem(q_ptr, q_eoff))
            q_packs.append(Vec.from_elements(q_elems, fx.BFloat16))

        # ---- online-softmax carry (finite -BIG sentinel, never -inf) ---
        # Flat loop-carried state for the runtime scf.for tile loops:
        #   [0:4]            m_i   -- per-row running max (4 MFMA rows / lane)
        #   [4:8]            l_i   -- per-row running softmax denominator
        #   [8:8+D_BLKS]     o_acc -- D_BLKS x vector<4xf32> output accumulator
        # All entries are raw IR values (vector<4xf32> for o_acc) so they can be
        # carried directly through scf.for / yield.
        _init = (
            [_raw(fx.Float32(-BIG)) for _ in range_constexpr(4)]
            + [_raw(fx.Float32(0.0)) for _ in range_constexpr(4)]
            + [_raw(Vec.filled(4, 0.0, fx.Float32)) for _ in range_constexpr(D_BLKS)]
        )

        def emit_tile(kv_tile, m_run, l_run, o_acc,
                      pool_ptr, idx_ptr, tkl_val, topk, bs, pbb):
            # kv_tile is a RUNTIME scf.for induction value (fx.Index); the carry
            # (m_run, l_run, o_acc) is passed in explicitly per iteration.
            kv_pos_base = kv_tile * fx.Index(block_n)

            # === gather + dequant one BLOCK_N tile -> LDS (bf16) =========
            idx_i32 = load_i32(idx_ptr, pid_t * fx.Index(topk)
                               + kv_pos_base + kv_row)
            idx_c = _mlir_arith.MaxSIOp(idx_i32, _raw(fx.Int32(0))).result
            blk = _mlir_arith.DivUIOp(idx_c, _raw(fx.Int32(bs))).result
            off = _mlir_arith.RemUIOp(idx_c, _raw(fx.Int32(bs))).result
            blk_i = fx.Index(blk)
            off_i = fx.Index(off)
            data_base = blk_i * fx.Index(pbb) + off_i * fx.Index(SLOT_DATA_BYTES)
            scale_base = (blk_i * fx.Index(pbb)
                          + fx.Index(bs * SLOT_DATA_BYTES)
                          + off_i * fx.Index(SLOT_SCALE_BYTES))

            # phase A: nope (448 fp8) -> store RAW fp8 packed i64 (NO dequant).
            # The same 8-fp8-per-i64 global load as before, but we relocate the
            # value straight into LDS in compact form; the dequant is done later
            # ON READ in the QK/PV loops (math-identical, bf16 MFMA unchanged).
            for dc in range_constexpr(NOPE_CHUNKS):
                abs_dim = g_half * fx.Index(NOPE_HALF) + fx.Index(dc * 8)
                raw_i64 = load_i64(pool_ptr, data_base + abs_dim)
                i64_off = (kv_row * fx.Index(NOPE_I64)
                           + g_half * fx.Index(NOPE_HALF_I64) + fx.Index(dc))
                _memref.store(_raw(raw_i64), lds_kv_nope, [_raw(i64_off)])

            # phase A': pack the 7 ue8m0 scale bytes for this key into ONE i64,
            # stored in the SAME i64 buffer (scale sub-region) so the QK/PV reads
            # go through the proven, race-free i64 path.  Single-writer via the
            # AFFINE index SCALE_BASE_I64 + g_half*block_n + kv_row: g_half==0
            # writes the read region [.., +block_n); g_half==1 writes the scratch
            # half (never read).  exp2 decode happens ON READ (bit-identical).
            sc_i64 = _raw(fx.Int64(0))
            for t in range_constexpr(N_NOPE_TILES):
                bt = load_u8_i32(pool_ptr, scale_base + fx.Index(t))
                bt64 = _mlir_arith.ExtUIOp(T.i64, bt).result
                sh = _mlir_arith.ShLIOp(bt64, _raw(fx.Int64(t * 8))).result
                sc_i64 = _mlir_arith.OrIOp(sc_i64, sh).result
            scale_chunk = (fx.Index(SCALE_BASE_I64)
                           + g_half * fx.Index(block_n) + kv_row)
            _memref.store(sc_i64, lds_kv_nope, [_raw(scale_chunk)])

            # phase B: rope (128 raw bytes -> 64 bf16 LE) -> bf16 (stays bf16)
            for dc in range_constexpr(ROPE_CHUNKS):
                rope_idx = g_half * fx.Index(ROPE_HALF) + fx.Index(dc * 8)
                bvals = []
                for j in range_constexpr(8):
                    boff = (data_base + fx.Index(D_NOPE)
                            + (rope_idx + fx.Index(j)) * fx.Index(2))
                    bvals.append(load_bf16_byte(pool_ptr, boff))
                Vec.from_elements(bvals, fx.BFloat16).store(
                    lds_kv_rope, [kv_row * fx.Index(D_ROPE) + rope_idx]
                )

            gpu.barrier()

            # === QK GEMM : S[BLOCK_H, BLOCK_N] = Q @ K^T (bf16 MFMA) =====
            # K is dequantized ON READ: nope k-steps decode 8 fp8 from the packed
            # i64 LDS (fp8->f32 * cached f32 scale -> bf16); rope k-steps load
            # bf16 directly.  The 8xbf16 fed to mfma_bf16 is bit-identical to the
            # previous up-front-dequantized LDS tile (same ops, relocated).
            # Decode this thread's per-key scales into registers (2 keys it reads
            # in QK: nb*16+lane).  Reads the packed scale i64 from lds_kv_nope and
            # exp2-decodes the 7 tiles; bit-identical to the prior up-front decode.
            qk_scale = []
            for nb in range_constexpr(N_BLKS_S):
                key = fx.Index(nb * MFMA_N) + lane
                sc_i64 = _raw(_memref.load(
                    lds_kv_nope, [_raw(fx.Index(SCALE_BASE_I64) + key)]))
                qk_scale.append(
                    [ue8m0_to_f32(extract_byte_i32(sc_i64, tl))
                     for tl in range_constexpr(N_NOPE_TILES)]
                )

            s_acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(N_BLKS_S)]
            for ks in range_constexpr(K_STEPS_QK):
                q_a = q_packs[ks]
                for nb in range_constexpr(N_BLKS_S):
                    key = fx.Index(nb * MFMA_N) + lane
                    if const_expr(ks < KS_NOPE):
                        # nope: 8 consecutive fp8 dims [ks*32+kg*8 : +8], all in
                        # the single 64-tile ks//2; i64 chunk = ks*4 + k_group.
                        i64_off = (key * fx.Index(NOPE_I64)
                                   + fx.Index(ks * 4) + k_group)
                        raw_i64 = _raw(_memref.load(lds_kv_nope, [_raw(i64_off)]))
                        scale_f = qk_scale[nb][ks // 2]
                        k_b = Vec.from_elements(
                            [to_bf16(_fmul(fp8_to_f32(extract_byte_i32(raw_i64, j)),
                                           scale_f))
                             for j in range_constexpr(8)],
                            fx.BFloat16,
                        )
                    else:
                        # rope: 8 consecutive bf16 dims, loaded directly.
                        rope_off = (key * fx.Index(D_ROPE)
                                    + fx.Index((ks - KS_NOPE) * MFMA_K)
                                    + k_group * fx.Index(8))
                        k_b = Vec.load(vec8bf16_ty, lds_kv_rope, [rope_off])
                    s_acc[nb] = mfma_bf16(s_acc[nb], q_a, k_b)

            # === validity mask (recomputed per key column) ==============
            mask_f = []
            for nb in range_constexpr(N_BLKS_S):
                key_pos = kv_pos_base + fx.Index(nb * MFMA_N) + lane
                ki = load_i32(idx_ptr, pid_t * fx.Index(topk) + key_pos)
                ne = arith.cmpi(arith.CmpIPredicate.ne, ki, fx.Int32(-1))
                kp_i32 = _mlir_arith.IndexCastOp(T.i32, _raw(key_pos)).result
                lt = arith.cmpi(arith.CmpIPredicate.slt, kp_i32, tkl_val)
                good = _mlir_arith.AndIOp(_raw(ne), _raw(lt)).result
                mask_f.append(_mlir_arith.UIToFPOp(T.f32, good).result)

            # === online softmax (log2 space) ===========================
            s_scaled = [
                [_fmul(Vec(s_acc[nb])[r], c_sm_log2e) for r in range_constexpr(4)]
                for nb in range_constexpr(N_BLKS_S)
            ]
            # masked columns: drive far negative for the max (finite, no NaN)
            s_for_max = [
                [_fadd(s_scaled[nb][r], _fmul(_fsub(mask_f[nb], c_one), c_big))
                 for r in range_constexpr(4)]
                for nb in range_constexpr(N_BLKS_S)
            ]
            local_max = [_fmax(s_for_max[0][r], s_for_max[1][r]) for r in range_constexpr(4)]
            row_max = list(local_max)
            for xor_off in [8, 4, 2, 1]:
                so = fx.Int32(xor_off)
                for r in range_constexpr(4):
                    row_max[r] = _fmax(
                        row_max[r],
                        fx.Float32(row_max[r]).shuffle_xor(so, fx.Int32(WARP_SIZE)),
                    )

            m_new = [_fmax(m_run[r], row_max[r]) for r in range_constexpr(4)]
            corr  = [_exp2(_fsub(m_run[r], m_new[r])) for r in range_constexpr(4)]

            p_vals   = [[None] * 4 for _ in range_constexpr(N_BLKS_S)]
            tile_sum = [_raw(fx.Float32(0.0)) for _ in range_constexpr(4)]
            for nb in range_constexpr(N_BLKS_S):
                for r in range_constexpr(4):
                    p = _fmul(_exp2(_fsub(s_scaled[nb][r], m_new[r])), mask_f[nb])
                    p_vals[nb][r] = p
                    tile_sum[r] = _fadd(tile_sum[r], p)
            for xor_off in [8, 4, 2, 1]:
                so = fx.Int32(xor_off)
                for r in range_constexpr(4):
                    tile_sum[r] = _fadd(
                        tile_sum[r],
                        fx.Float32(tile_sum[r]).shuffle_xor(so, fx.Int32(WARP_SIZE)),
                    )

            l_new = [_fadd(_fmul(corr[r], l_run[r]), tile_sum[r]) for r in range_constexpr(4)]
            new_o = []
            for d in range_constexpr(D_BLKS):
                ov = Vec(o_acc[d])
                new_o.append(
                    Vec.from_elements(
                        [_fmul(ov[r], corr[r]) for r in range_constexpr(4)], fx.Float32
                    )
                )

            # === stage P (f32) -> LDS, transposed [head_row][key] =======
            for nb in range_constexpr(N_BLKS_S):
                for r in range_constexpr(4):
                    p_row = k_group * fx.Index(4) + fx.Index(r)
                    p_col = fx.Index(nb * MFMA_N) + lane
                    _memref.store(_raw(p_vals[nb][r]), lds_p,
                                  [_raw(p_row * fx.Index(block_n) + p_col)])
            gpu.barrier()

            # === PV GEMM : O += P @ V (bf16 MFMA) =======================
            # V is the SAME LDS buffer as K (no second gather / no redundant
            # dequant of the tile up front).  Each PV element reads one head-dim
            # value and dequantizes ON READ: nope d-blocks (d<28) decode byte
            # lane%8 of the packed i64 (col = d*16+lane, all within 64-tile d//4);
            # rope d-blocks (d>=28) load bf16 directly.  bf16 V is bit-identical.
            lane_div8 = lane // fx.Index(8)
            lane_shift_i64 = fx.Int64(lane % fx.Index(8)) * fx.Int64(8)
            p_base = lane * fx.Index(block_n) + k_group * fx.Index(8)
            p_a = Vec.from_elements(
                [to_bf16(_memref.load(lds_p, [_raw(p_base + fx.Index(j))]))
                 for j in range_constexpr(8)],
                fx.BFloat16,
            )
            # Decode this thread's per-key scales into registers (8 keys it reads
            # in PV: k_group*8 + j).  Same packed-i64 source as QK; exp2 on read.
            pv_scale = []
            for j in range_constexpr(8):
                key = k_group * fx.Index(8) + fx.Index(j)
                sc_i64 = _raw(_memref.load(
                    lds_kv_nope, [_raw(fx.Index(SCALE_BASE_I64) + key)]))
                pv_scale.append(
                    [ue8m0_to_f32(extract_byte_i32(sc_i64, tl))
                     for tl in range_constexpr(N_NOPE_TILES)]
                )
            for d in range_constexpr(D_BLKS):
                vvals = []
                for j in range_constexpr(8):
                    key = k_group * fx.Index(8) + fx.Index(j)
                    if const_expr(d < D_NOPE_BLKS):
                        # nope: col = d*16+lane; i64 chunk = d*2 + lane//8,
                        # byte = lane%8, 64-tile = d//4 (const per d-block).
                        i64_off = (key * fx.Index(NOPE_I64)
                                   + fx.Index(d * 2) + lane_div8)
                        raw_i64 = _memref.load(lds_kv_nope, [_raw(i64_off)])
                        scale_f = pv_scale[j][d // 4]
                        byte_i32 = extract_byte_i32_dyn(raw_i64, lane_shift_i64)
                        vvals.append(to_bf16(_fmul(fp8_to_f32(byte_i32), scale_f)))
                    else:
                        # rope: col-448 = (d-28)*16 + lane, bf16 loaded directly.
                        rope_off = (key * fx.Index(D_ROPE)
                                    + fx.Index((d - D_NOPE_BLKS) * MFMA_N) + lane)
                        vvals.append(_memref.load(lds_kv_rope, [_raw(rope_off)]))
                v_b = Vec.from_elements(vvals, fx.BFloat16)
                new_o[d] = mfma_bf16(new_o[d], p_a, v_b)
            gpu.barrier()
            return m_new, l_new, new_o

        # ==== dual-scope online softmax: SWA (main) then EXTRA ==========
        # RUNTIME scf.for carry loops: the per-tile body is emitted ONCE (not
        # compile-time unrolled), so code size / JIT time stay bounded even at
        # production topk (e.g. extra topk 2048 / block_n 32 -> 64 tiles).  The
        # trip count KV_TILES_* = ceil(topk / BLOCK_N) is derived from the tensor
        # shapes; the EXTRA loop continues into the SAME shared carried state
        # (m_i, l_i, o_acc) produced by the SWA loop.  Per-key masking beyond
        # topk_length / idx==-1 still zeroes any over-run column exactly as
        # before, so a fixed (padded) trip count is numerically safe.
        loop_results = _init
        for kv_tile, _carry in range(0, KV_TILES_MAIN, 1, init=_init):
            m_run = [_carry[r] for r in range_constexpr(4)]
            l_run = [_carry[4 + r] for r in range_constexpr(4)]
            o_acc = [_carry[8 + d] for d in range_constexpr(D_BLKS)]
            m_new, l_new, new_o = emit_tile(
                kv_tile, m_run, l_run, o_acc,
                main_pool_ptr, main_idx_ptr, topklen_main_val,
                topk_main, block_size_main, PBB_MAIN,
            )
            loop_results = yield list(m_new) + list(l_new) + list(new_o)

        for kv_tile, _carry in range(0, KV_TILES_EXTRA, 1, init=loop_results):
            m_run = [_carry[r] for r in range_constexpr(4)]
            l_run = [_carry[4 + r] for r in range_constexpr(4)]
            o_acc = [_carry[8 + d] for d in range_constexpr(D_BLKS)]
            m_new, l_new, new_o = emit_tile(
                kv_tile, m_run, l_run, o_acc,
                extra_pool_ptr, extra_idx_ptr, topklen_extra_val,
                topk_extra, block_size_extra, PBB_EXTRA,
            )
            loop_results = yield list(m_new) + list(l_new) + list(new_o)

        # ==== finalize : normalize, fold attn_sink, lonely -> zero ======
        m_run = [loop_results[r] for r in range_constexpr(4)]
        l_run = [loop_results[4 + r] for r in range_constexpr(4)]
        o_acc = [loop_results[8 + d] for d in range_constexpr(D_BLKS)]
        for r in range_constexpr(4):
            head = head_base + k_group * fx.Index(4) + fx.Index(r)
            l_r = l_run[r]
            m_r = m_run[r]
            # lonely (l==0): l>=1 whenever any key was valid (max key p=1)
            lonely_f = _fsub(c_one, _fmin(_fmul(l_r, c_big), c_one))
            one_m = _fsub(c_one, lonely_f)
            m_safe = _fmul(m_r, one_m)          # 0 when lonely (mirrors ref)
            if const_expr(has_attn_sink):
                sink = load_f32_elem(sink_ptr, head)
                sink_term = _exp2(_fsub(_fmul(sink, c_log2e), m_safe))
                denom = _fadd(l_r, sink_term)
            else:
                denom = l_r
            safe_denom = _fadd(_fmul(denom, one_m), lonely_f)   # ==1 when lonely
            inv = arith.divf(_raw(c_one), _raw(safe_denom), fastmath=fm)
            for d in range_constexpr(D_BLKS):
                o_norm = _fmul(Vec(o_acc[d])[r], inv)
                out_off = (pid_t * fx.Index(TOK_OUT_STRIDE)
                           + head * fx.Index(HEAD_OUT_STRIDE)
                           + fx.Index(d * MFMA_N) + lane)
                out_gep = buffer_ops.get_element_ptr(
                    out_ptr, fx.Int64(out_off), elem_type=T.bf16
                )
                _llvm.StoreOp(to_bf16(o_norm).ir_value(), out_gep)
        return

    @flyc.jit
    def launch_dual_scope_prefill(
        Q, KV_Main, Indices_Main, TopkLen_Main,
        KV_Extra, Indices_Extra, TopkLen_Extra,
        AttnSink, Out, total_tokens,
        stream: "fx.Stream" = fx.Stream(None),
    ):
        alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalize()

        tokens_idx = fx.Index(total_tokens)
        grid_h = fx.Index((h_q + BLOCK_H - 1) // BLOCK_H)
        launcher = dual_scope_prefill_kernel(
            Q, KV_Main, Indices_Main, TopkLen_Main,
            KV_Extra, Indices_Extra, TopkLen_Extra,
            AttnSink, Out, total_tokens,
        )

        passthrough = []
        for pair in [
            ("denormal-fp-math-f32", "preserve-sign,preserve-sign"),
            ("no-nans-fp-math", "true"),
            ("unsafe-fp-math", "true"),
        ]:
            passthrough.append(
                ir.ArrayAttr.get(
                    [ir.StringAttr.get(pair[0]), ir.StringAttr.get(pair[1])]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough)
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                    T.i32, int(waves_per_eu)
                )

        launcher.launch(
            grid=(grid_h, tokens_idx, 1),
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
            return launch_dual_scope_prefill(*args, **kwargs)

    return _launch


def _flydsl_dual_scope_kernel_impl(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_length: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    compress_ratio: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
) -> torch.Tensor:
    """Thin launcher: normalize inputs, build/cache the kernel, launch, return.

    Allocates the output [T, H, head_dim_v] bf16 and launches the live FlyDSL
    dual-scope kernel, which computes the full attention (gather + fp8 dequant +
    QK + online softmax + PV).  Reached from the public entry whenever
    ``_USE_FLYDSL_KERNEL`` is True and FlyDSL is available on the gfx950 host.
    Q / attn_sink are padded to a multiple of BLOCK_H heads; the extra padding
    heads are sliced off (``[:, :H]``) before returning.
    """
    q3 = _normalize_q(q)                         # [T, H, D_QK]
    device = q3.device
    T_tok, H, _ = q3.shape

    block_h = 16
    h_q_pad = ((H + block_h - 1) // block_h) * block_h

    # --- main (SWA) scope tensors ---------------------------------------
    swa_idx = _normalize_indices(swa_indices).to(torch.int32).contiguous()  # [T, topk_m]
    topk_main = swa_idx.shape[1]
    swa_u8, _nb_main, bs_main = _block_layout(swa_k_cache)                   # [Nb, pbb] u8

    # --- extra scope tensors (dummy, non-indexed, when absent) ----------
    has_extra = extra_k_cache is not None and extra_indices is not None
    if has_extra:
        extra_idx = _normalize_indices(extra_indices).to(torch.int32).contiguous()
        topk_extra = extra_idx.shape[1]
        extra_u8, _nb_extra, bs_extra = _block_layout(extra_k_cache)
    else:
        # No extra scope: feed an all-invalid (-1) dummy so the kernel's extra
        # tiles mask out to zero contribution (matches the reference, which
        # skips the extra scope entirely).
        topk_extra = topk_main
        extra_idx = torch.full((T_tok, topk_extra), -1, dtype=torch.int32, device=device)
        extra_u8 = swa_u8
        bs_extra = bs_main

    # --- per-token topk_length (dummy full-length when absent) ----------
    if swa_topk_length is not None:
        topklen_main = swa_topk_length.to(torch.int32).contiguous()
    else:
        topklen_main = torch.full((T_tok,), topk_main, dtype=torch.int32, device=device)
    if has_extra and extra_topk_length is not None:
        topklen_extra = extra_topk_length.to(torch.int32).contiguous()
    else:
        topklen_extra = torch.full((T_tok,), topk_extra, dtype=torch.int32, device=device)

    # --- attn_sink (padded/dummy to h_q_pad) ----------------------------
    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).reshape(-1).contiguous()
        if sink.shape[0] < h_q_pad:
            sink = torch.cat(
                [sink, torch.zeros(h_q_pad - sink.shape[0], dtype=torch.float32, device=device)]
            )
    else:
        sink = torch.zeros(h_q_pad, dtype=torch.float32, device=device)

    # Q padded to h_q_pad heads (kernel computes a full BLOCK_H tile; extra
    # heads are discarded by the [:, :H] slice on return).
    q_pad = torch.zeros((T_tok, h_q_pad, D_QK), dtype=torch.bfloat16, device=device)
    q_pad[:, :H] = q3.to(torch.bfloat16)

    out = torch.zeros((T_tok, h_q_pad, head_dim_v), dtype=torch.bfloat16, device=device)

    kernel = _build_dual_scope_kernel(
        h_q=h_q_pad,
        head_dim=D_QK,
        head_dim_v=head_dim_v,
        topk_main=topk_main,
        topk_extra=topk_extra,
        block_size_main=bs_main,
        block_size_extra=bs_extra,
        sm_scale=float(softmax_scale),
        has_attn_sink=attn_sink is not None,
        has_topk_length_main=swa_topk_length is not None,
        has_topk_length_extra=(has_extra and extra_topk_length is not None),
    )

    stream = torch.cuda.current_stream()
    kernel(
        q_pad, swa_u8, swa_idx, topklen_main,
        extra_u8, extra_idx, topklen_extra,
        sink, out, total_tokens=int(T_tok),
        stream=fx.Stream(stream.cuda_stream),
    )
    return out[:, :H].contiguous()


# ============================================================================
# Public entry -- matches the backend hook signature exactly.
# ============================================================================
def flydsl_dual_scope_prefill(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_length: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    compress_ratio: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
) -> torch.Tensor:
    """Fused dual-scope sparse-attention prefill (SWA + C4/C128).

    Contract (validated against the FlashMLA baseline in the backend hook):
        q            : [T, 1, H, 512] (or [T, H, 512])     bf16
        swa_k_cache  : [Nb_m, block_size_m, 1, 584]        uint8 raw pool
        swa_indices  : [T, 1, topk_m] (or [T, topk_m])     int32
        swa_topk_length   : [batch] int32 or None
        extra_k_cache: [Nb_e, block_size_e, 1, 584] uint8 or None
        extra_indices: [T, 1, topk_e] (or [T, topk_e]) int32 or None
        extra_topk_length : [batch] int32 or None
        compress_ratio    : 4 or 128 (extra-scope pool compression)
        softmax_scale     : float
        attn_sink         : [H] float or None
        head_dim_v        : 512

    Returns [T, H, head_dim_v] bf16; the hook squeezes a 4D result, so a 3D
    return matches ``flash_mla_with_kvcache_entrypoint(...)[0].squeeze(1)``.
    """
    if _USE_FLYDSL_KERNEL:
        # Live kernel path. If FlyDSL is unavailable here we must NOT silently
        # degrade to the PyTorch oracle -- the kernel is the intended live path,
        # so fail hard (strict mode) and let the server crash.
        if not _flydsl_available():
            raise RuntimeError(
                "[FlyDSL] _USE_FLYDSL_KERNEL=True but FlyDSL runtime is "
                "unavailable -- aborting (strict mode)"
            )
        return _flydsl_dual_scope_kernel_impl(
            q=q,
            swa_k_cache=swa_k_cache,
            swa_indices=swa_indices,
            swa_topk_length=swa_topk_length,
            extra_k_cache=extra_k_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_length,
            compress_ratio=compress_ratio,
            softmax_scale=softmax_scale,
            attn_sink=attn_sink,
            head_dim_v=head_dim_v,
        )

    # Oracle path: only reached when _USE_FLYDSL_KERNEL is False (tests).
    return _torch_reference_dual_scope(
        q=q,
        swa_k_cache=swa_k_cache,
        swa_indices=swa_indices,
        swa_topk_length=swa_topk_length,
        extra_k_cache=extra_k_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        compress_ratio=compress_ratio,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
        head_dim_v=head_dim_v,
    )
