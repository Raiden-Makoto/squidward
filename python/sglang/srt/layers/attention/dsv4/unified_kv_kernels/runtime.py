"""Runtime glue for the unified_kv backend.

Builds unified_kv-style flat ``kv_indices`` / ``kv_indptr`` from SGLang's already-computed
DSV4 metadata, scatters SWA K into the bf16 ``unified_kv`` ring, and dispatches the
vendored paged decode/prefill kernels.

unified_kv[L] layout (page_size 1, bf16, row-major):
  - rows ``[0, swa_pages)``    = SWA ring (``state_slot * win + pos % win``);
  - rows ``[swa_pages, ...)``  = compressed K (``swa_pages + page_index``), where
    SGLang metadata already encodes the compressed slot id:
      HCA (ratio 128): ``c128_page_indices``      (== phys_block, k_per_block=1)
      CSA (ratio   4): ``c4_sparse_page_indices``  (== phys_block*32 + slot)

Index layout: RAGGED-PACKED. Each token's segment is tightly packed
(``kv_indptr`` is a true prefix sum of per-token valid lengths) so the
attention K-loop scans only real entries. The backing buffer is still
allocated at the fixed worst-case capacity ``N * (win + Wc)`` so its shape is
static across CUDA-graph replay; only ``kv_indptr`` values (and the written
prefix) vary per forward. Compressed valid entries are front-packed in the
``*_page_indices`` rows (the same contract the non-unified_kv flashmla path relies
on via ``topk_length``); the per-token compressed count is recovered from the
``kv_indptr`` delta inside the kernel, so no extra length tensor is threaded.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.attention.dsv4.unified_kv_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
)
from sglang.srt.layers.attention.dsv4.unified_kv_kernels.paged_decode_indices import (
    write_v4_paged_decode_indices,
)
from sglang.srt.layers.attention.dsv4.unified_kv_kernels.paged_prefill import (
    sparse_attn_v4_paged_prefill,
)


# ---------------------------------------------------------------------------
# SWA ring scatter
# ---------------------------------------------------------------------------
@triton.jit
def _swa_scatter_kernel(
    kv_ptr,  # [T, D] bf16
    state_slot_ptr,  # [T] int
    positions_ptr,  # [T] int
    final_pos_ptr,  # [T] int
    unified_ptr,  # [pages, D] bf16
    n_rows,
    ring_stride,  # SWA ring per-slot stride
    win: tl.constexpr,
    D: tl.constexpr,
    HAS_FINAL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """BF16 SWA ring scatter. The fp8 (MXFP8) ring store is a separate kernel
    (``_mxfp8_pack_scatter_kernel``) dispatched via ``store_swa_into_unified``."""
    row = tl.program_id(0)
    if row >= n_rows:
        return
    pos = tl.load(positions_ptr + row)
    if HAS_FINAL:
        fp = tl.load(final_pos_ptr + row)
        if pos <= fp - win:
            return
    s = tl.load(state_slot_ptr + row)
    loc = s * ring_stride + (pos % ring_stride)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    vals = tl.load(kv_ptr + row * D + offs, mask=mask, other=0.0)
    tl.store(unified_ptr + loc * D + offs, vals, mask=mask)


# ---------------------------------------------------------------------------
# MXFP8 (E8M0 per-32-block) NoPE pack + BF16 RoPE scatter.
#
# Single source for all three unified-KV fp8 write paths (decode ring store,
# prefill ring store, compressed-K store). Each program packs one [head_dim]
# norm+rope'd row into the MXFP8 NoPE layout (see mxfp8.py) at ``loc`` of the
# fp8 NoPE buffer + writes the RoPE tail bf16 into the parallel rope buffer.
# Rows with ``loc < 0`` are skipped (SWA windowing sentinel).
# ---------------------------------------------------------------------------
@triton.jit
def _mxfp8_pack_scatter_kernel(
    kv_ptr,  # [T, head_dim] norm+rope'd (any float dtype)
    loc_ptr,  # [T] int destination rows (<0 -> skip)
    nope_u8_ptr,  # [pages, NOPE_WIDTH] uint8 (fp8 NoPE buffer viewed as uint8)
    rope_ptr,  # [pages, DIM_ROPE] bf16
    n_rows,
    kv_row_stride,
    kv_col_stride,
    DIM_NOPE: tl.constexpr,  # 448
    DIM_ROPE: tl.constexpr,  # 64
    NUM_BLOCKS: tl.constexpr,  # 14
    FP8_BLK: tl.constexpr,  # 32
    NOPE_WIDTH: tl.constexpr,  # 512
    SCALE_OFF: tl.constexpr,  # 448
    FP8_MAX: tl.constexpr,  # 448.0
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
    loc = tl.load(loc_ptr + rows, mask=rows < n_rows, other=-1).to(tl.int64)
    # Per-row validity: in-range and not the SWA windowing skip sentinel.
    rmask = (rows < n_rows) & (loc >= 0)
    blk_offs = tl.arange(0, FP8_BLK)  # [FP8_BLK]
    for b in tl.static_range(NUM_BLOCKS):
        start = b * FP8_BLK
        cols = start + blk_offs  # [FP8_BLK]
        offs = rows[:, None] * kv_row_stride + cols[None, :] * kv_col_stride
        vals = tl.load(kv_ptr + offs, mask=rmask[:, None], other=0.0).to(tl.float32)
        # E8M0 block exponent: e = ceil(log2(amax / FP8_MAX)); scale = 2^e.
        # Matches mxfp8.pack_nope_mxfp8 / aiter _quantize_nope bit-for-bit.
        amax = tl.max(tl.abs(vals), axis=1)  # [BLOCK_ROWS]
        amax_c = tl.maximum(amax, 1e-30)
        e_unb = tl.ceil(tl.log2(amax_c / FP8_MAX))
        e_unb = tl.where(amax == 0.0, 0.0, e_unb)
        scale = tl.exp2(e_unb)
        q = (vals / scale[:, None]).to(tl.float8e4nv)
        qb = q.to(tl.uint8, bitcast=True)
        tl.store(
            nope_u8_ptr + loc[:, None] * NOPE_WIDTH + cols[None, :],
            qb,
            mask=rmask[:, None],
        )
        e_byte = e_unb.to(tl.int32) + 127
        e_byte = tl.minimum(tl.maximum(e_byte, 0), 255).to(tl.uint8)
        tl.store(
            nope_u8_ptr + loc * NOPE_WIDTH + SCALE_OFF + b,
            e_byte,
            mask=rmask,
        )
    rope_offs = tl.arange(0, DIM_ROPE)  # [DIM_ROPE]
    roffs = rows[:, None] * kv_row_stride + (DIM_NOPE + rope_offs)[None, :] * kv_col_stride
    rvals = tl.load(kv_ptr + roffs, mask=rmask[:, None], other=0.0)
    tl.store(
        rope_ptr + loc[:, None] * DIM_ROPE + rope_offs[None, :],
        rvals.to(rope_ptr.dtype.element_ty),
        mask=rmask[:, None],
    )


@triton.jit
def _mxfp8_pack_dense_kernel(
    kv_ptr,  # [N, head_dim] norm+rope'd (any float dtype)
    nope_u8_ptr,  # [N, NOPE_WIDTH] uint8 (dense fp8 NoPE out viewed as uint8)
    rope_ptr,  # [N, DIM_ROPE] bf16 (dense RoPE out)
    n_rows,
    kv_row_stride,
    kv_col_stride,
    DIM_NOPE: tl.constexpr,  # 448
    DIM_ROPE: tl.constexpr,  # 64
    NUM_BLOCKS: tl.constexpr,  # 14
    FP8_BLK: tl.constexpr,  # 32
    NOPE_WIDTH: tl.constexpr,  # 512
    SCALE_OFF: tl.constexpr,  # 448
    FP8_MAX: tl.constexpr,  # 448.0
    BLOCK_ROWS: tl.constexpr,
):
    """Dense (row i -> row i) variant of ``_mxfp8_pack_scatter_kernel``. Quant
    math is identical bit-for-bit; only the destination addressing differs (no
    ``loc`` indirection / no skip sentinel). Used by ``pack_mxfp8_dense`` to fuse
    the per-step q / extend-kv pack chains in the fp8 prefill path.

    Each program packs a tile of ``BLOCK_ROWS`` rows. The per-32 E8M0 block math
    is vectorized across rows (reduce along the 32-wide axis), so every lane is
    busy and rows are loaded/stored coalesced -- far better HW utilization than a
    one-row-per-program, 32-lane launch."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
    row_mask = rows < n_rows
    blk_offs = tl.arange(0, FP8_BLK)  # [FP8_BLK]
    for b in tl.static_range(NUM_BLOCKS):
        start = b * FP8_BLK
        cols = start + blk_offs  # [FP8_BLK]
        offs = rows[:, None] * kv_row_stride + cols[None, :] * kv_col_stride
        vals = tl.load(kv_ptr + offs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        # E8M0 block exponent: e = ceil(log2(amax / FP8_MAX)); scale = 2^e.
        # Matches mxfp8.pack_nope_mxfp8 / aiter _quantize_nope bit-for-bit.
        amax = tl.max(tl.abs(vals), axis=1)  # [BLOCK_ROWS]
        amax_c = tl.maximum(amax, 1e-30)
        e_unb = tl.ceil(tl.log2(amax_c / FP8_MAX))
        e_unb = tl.where(amax == 0.0, 0.0, e_unb)
        scale = tl.exp2(e_unb)
        q = (vals / scale[:, None]).to(tl.float8e4nv)
        qb = q.to(tl.uint8, bitcast=True)
        tl.store(
            nope_u8_ptr + rows[:, None] * NOPE_WIDTH + cols[None, :],
            qb,
            mask=row_mask[:, None],
        )
        e_byte = e_unb.to(tl.int32) + 127
        e_byte = tl.minimum(tl.maximum(e_byte, 0), 255).to(tl.uint8)
        tl.store(
            nope_u8_ptr + rows * NOPE_WIDTH + SCALE_OFF + b,
            e_byte,
            mask=row_mask,
        )
    rope_offs = tl.arange(0, DIM_ROPE)  # [DIM_ROPE]
    roffs = rows[:, None] * kv_row_stride + (DIM_NOPE + rope_offs)[None, :] * kv_col_stride
    rvals = tl.load(kv_ptr + roffs, mask=row_mask[:, None], other=0.0)
    tl.store(
        rope_ptr + rows[:, None] * DIM_ROPE + rope_offs[None, :],
        rvals.to(rope_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )


def pack_mxfp8_dense(
    x: torch.Tensor,  # [..., head_dim] norm+rope'd (any float dtype)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused MXFP8 dense pack. A single Triton launch produces the packed
    NoPE-fp8 ``[..., 512]`` buffer + its bf16 RoPE ``[..., 64]`` companion,
    byte-identical to ``pack_nope_mxfp8(split_nope_rope(x)[0])`` plus
    ``rope.to(bfloat16)``. Replaces the eager per-step q / extend-kv pack chains
    in the fp8 prefill path (one launch instead of ~10 small eager kernels).

    Arbitrary leading dims are flattened; the packed/rope outputs are fresh,
    contiguous (width-equal row stride) tensors. NoPE pad bytes ``[462:512]`` are
    zeroed via the zero-init alloc (the kernel only writes the 448 data + 14 E8M0
    bytes), matching the reference packer."""
    from sglang.srt.layers.attention.dsv4.unified_kv_kernels import mxfp8

    assert x.shape[-1] == mxfp8.DIM_HEAD, (
        f"pack_mxfp8_dense expects head_dim {mxfp8.DIM_HEAD}, got {x.shape[-1]}"
    )
    lead = x.shape[:-1]
    flat = x.reshape(-1, mxfp8.DIM_HEAD)
    n_rows = flat.shape[0]
    nope = torch.zeros(
        n_rows, mxfp8.NOPE_PACKED_WIDTH, dtype=torch.uint8, device=x.device
    )
    rope = torch.empty(n_rows, mxfp8.DIM_ROPE, dtype=torch.bfloat16, device=x.device)
    if n_rows > 0:
        BLOCK_ROWS = 8
        grid = (triton.cdiv(n_rows, BLOCK_ROWS),)
        _mxfp8_pack_dense_kernel[grid](
            flat,
            nope,
            rope,
            n_rows,
            flat.stride(0),
            flat.stride(1),
            DIM_NOPE=mxfp8.DIM_NOPE,
            DIM_ROPE=mxfp8.DIM_ROPE,
            NUM_BLOCKS=mxfp8.NUM_NOPE_BLOCKS,
            FP8_BLK=mxfp8.FP8_BLOCK,
            NOPE_WIDTH=mxfp8.NOPE_PACKED_WIDTH,
            SCALE_OFF=mxfp8.SCALE_OFFSET,
            FP8_MAX=mxfp8.FP8_MAX,
            BLOCK_ROWS=BLOCK_ROWS,
            num_warps=4,
        )
    nope = nope.view(mxfp8.FP8_DTYPE).reshape(*lead, mxfp8.NOPE_PACKED_WIDTH)
    rope = rope.reshape(*lead, mxfp8.DIM_ROPE)
    return nope, rope


def _launch_mxfp8_pack(
    *,
    kv: torch.Tensor,  # [T, head_dim]
    loc: torch.Tensor,  # [T] int dest rows (<0 = skip)
    unified_kv_nope: torch.Tensor,  # [pages, 512] fp8
    unified_kv_rope: torch.Tensor,  # [pages, 64] bf16
) -> None:
    from sglang.srt.layers.attention.dsv4.unified_kv_kernels import mxfp8

    n_rows, D = kv.shape
    if n_rows == 0:
        return
    assert unified_kv_nope.dtype == mxfp8.FP8_DTYPE
    assert D == mxfp8.DIM_HEAD, f"expected head_dim {mxfp8.DIM_HEAD}, got {D}"
    if not loc.is_contiguous():
        loc = loc.contiguous()
    BLOCK_ROWS = 8
    grid = (triton.cdiv(n_rows, BLOCK_ROWS),)
    _mxfp8_pack_scatter_kernel[grid](
        kv,
        loc,
        unified_kv_nope.view(torch.uint8),
        unified_kv_rope,
        n_rows,
        kv.stride(0),
        kv.stride(1),
        DIM_NOPE=mxfp8.DIM_NOPE,
        DIM_ROPE=mxfp8.DIM_ROPE,
        NUM_BLOCKS=mxfp8.NUM_NOPE_BLOCKS,
        FP8_BLK=mxfp8.FP8_BLOCK,
        NOPE_WIDTH=mxfp8.NOPE_PACKED_WIDTH,
        SCALE_OFF=mxfp8.SCALE_OFFSET,
        FP8_MAX=mxfp8.FP8_MAX,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=4,
    )


@triton.jit
def _mxfp8_copy_scatter_kernel(
    nope_src_u8_ptr,  # [N, NOPE_WIDTH] uint8 (pre-packed dense NoPE)
    rope_src_ptr,  # [N, DIM_ROPE] bf16 (pre-packed dense RoPE)
    loc_ptr,  # [N] int dest rows (<0 -> skip)
    nope_dst_u8_ptr,  # [pages, NOPE_WIDTH] uint8 (pool NoPE viewed as uint8)
    rope_dst_ptr,  # [pages, DIM_ROPE] bf16 (pool RoPE)
    n_rows,
    NOPE_WIDTH: tl.constexpr,  # 512
    DIM_ROPE: tl.constexpr,  # 64
):
    """Copy a pre-packed MXFP8 row (512 fp8/u8 NoPE + 64 bf16 RoPE) into pool row
    ``loc`` (skip when ``loc < 0``). No re-quantization: reuses the bytes already
    produced by ``pack_mxfp8_dense`` for the prefill attention input."""
    row = tl.program_id(0)
    if row >= n_rows:
        return
    loc = tl.load(loc_ptr + row).to(tl.int64)
    if loc < 0:
        return
    noff = tl.arange(0, NOPE_WIDTH)
    nvals = tl.load(nope_src_u8_ptr + row * NOPE_WIDTH + noff)
    tl.store(nope_dst_u8_ptr + loc * NOPE_WIDTH + noff, nvals)
    roff = tl.arange(0, DIM_ROPE)
    rvals = tl.load(rope_src_ptr + row * DIM_ROPE + roff)
    tl.store(rope_dst_ptr + loc * DIM_ROPE + roff, rvals)


def _launch_mxfp8_copy_scatter(
    *,
    nope_src: torch.Tensor,  # [N, 512] fp8 (pre-packed dense)
    rope_src: torch.Tensor,  # [N, 64] bf16 (pre-packed dense)
    loc: torch.Tensor,  # [N] int dest rows (<0 = skip)
    unified_kv_nope: torch.Tensor,  # [pages, 512] fp8
    unified_kv_rope: torch.Tensor,  # [pages, 64] bf16
) -> None:
    from sglang.srt.layers.attention.dsv4.unified_kv_kernels import mxfp8

    n_rows = nope_src.shape[0]
    if n_rows == 0:
        return
    assert unified_kv_nope.dtype == mxfp8.FP8_DTYPE
    assert nope_src.dtype == mxfp8.FP8_DTYPE
    assert nope_src.is_contiguous() and rope_src.is_contiguous()
    if not loc.is_contiguous():
        loc = loc.contiguous()
    _mxfp8_copy_scatter_kernel[(n_rows,)](
        nope_src.view(torch.uint8),
        rope_src,
        loc,
        unified_kv_nope.view(torch.uint8),
        unified_kv_rope,
        n_rows,
        NOPE_WIDTH=mxfp8.NOPE_PACKED_WIDTH,
        DIM_ROPE=mxfp8.DIM_ROPE,
        num_warps=4,
    )


def store_swa_into_unified(
    *,
    kv: torch.Tensor,  # [T, head_dim] bf16 (norm+rope'd)
    state_slot: torch.Tensor,  # [T] int
    positions: torch.Tensor,  # [T] int
    unified_kv: torch.Tensor,  # [pages, head_dim] bf16, or [pages,512] fp8 (MXFP8)
    win: int,  # SWA attention window length
    ring_stride: int,  # SWA ring stride
    final_pos: Optional[torch.Tensor] = None,  # [T] req's last position
    unified_kv_rope: Optional[torch.Tensor] = None,  # [pages,64] bf16 (fp8 mode)
    kv_nope_packed: Optional[torch.Tensor] = None,  # [T,512] fp8 pre-packed NoPE
    kv_rope_packed: Optional[torch.Tensor] = None,  # [T,64] bf16 pre-packed RoPE
) -> None:
    n_rows, D = kv.shape
    if n_rows == 0:
        return

    has_final = final_pos is not None
    fp_arg = final_pos if has_final else positions
    assert kv.is_contiguous()
    assert state_slot.is_contiguous() and positions.is_contiguous()
    assert fp_arg.is_contiguous()

    # fp8 (MXFP8) quant-on-store is gated on the caller passing a RoPE buffer
    # (i.e. the unified pool is fp8). The bf16 path is strictly unchanged.
    if unified_kv_rope is not None:
        # Compute destination ring rows on-device; apply the SWA window sentinel
        # (skip rows where pos <= final_pos - win) by tagging loc = -1.
        ss = state_slot.to(torch.int64)
        pos64 = positions.to(torch.int64)
        loc = ss * ring_stride + (pos64 % ring_stride)
        if has_final:
            keep = pos64 > (final_pos.to(torch.int64) - win)
            loc = torch.where(keep, loc, torch.full_like(loc, -1))
        if kv_nope_packed is not None:
            # Dedup: the prefill attention already packed this chunk's extend kv;
            # reuse those bytes (copy-by-loc) instead of re-quantizing.
            _launch_mxfp8_copy_scatter(
                nope_src=kv_nope_packed,
                rope_src=kv_rope_packed,
                loc=loc,
                unified_kv_nope=unified_kv,
                unified_kv_rope=unified_kv_rope,
            )
        else:
            _launch_mxfp8_pack(
                kv=kv,
                loc=loc,
                unified_kv_nope=unified_kv,
                unified_kv_rope=unified_kv_rope,
            )
        return

    assert kv.dtype == unified_kv.dtype
    _swa_scatter_kernel[(n_rows,)](
        kv,
        state_slot,
        positions,
        fp_arg,
        unified_kv,
        n_rows,
        ring_stride,
        win=win,
        D=D,
        HAS_FINAL=has_final,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=8,
    )


def store_mxfp8_by_loc(
    *,
    kv: torch.Tensor,  # [T, head_dim] norm+rope'd (any float dtype)
    loc: torch.Tensor,  # [T] int destination rows in the unified pool
    unified_kv_nope: torch.Tensor,  # [pages, 512] fp8 (MXFP8-packed NoPE)
    unified_kv_rope: torch.Tensor,  # [pages, 64] bf16
) -> None:
    """MXFP8 (E8M0 per-32-block) NoPE quant + BF16 RoPE scatter to explicit pool
    rows. Same on-disk contract as the SWA ring store (``store_swa_into_unified``)
    and the decode/prefill read kernels. Used by the compressed-K store; HIP-safe
    (pure Triton, no CUDA-only symbol)."""
    _launch_mxfp8_pack(
        kv=kv,
        loc=loc,
        unified_kv_nope=unified_kv_nope,
        unified_kv_rope=unified_kv_rope,
    )


# ---------------------------------------------------------------------------
# Ragged indptr helper (shared by the decode streams + prefill builders)
# ---------------------------------------------------------------------------
def _lengths_to_indptr(lengths: torch.Tensor) -> torch.Tensor:
    """[N] int32 per-token lengths -> [N+1] int32 indptr"""
    return F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))


def decode(
    *,
    q: torch.Tensor,  # [T, H, D] (local heads)
    unified_kv: torch.Tensor,  # [pages, D] bf16, or [pages,512] fp8 (MXFP8 NoPE)
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,  # [H] fp32
    softmax_scale: float,
    unified_kv_rope: Optional[torch.Tensor] = None,  # [pages,64] bf16 (fp8 KV)
) -> torch.Tensor:
    return sparse_attn_v4_paged_decode(
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        softmax_scale,
        unified_kv_rope=unified_kv_rope,
    )


@triton.jit
def _fill_compress_tail_kernel(
    indices_ptr,  # [*] int32 (out)
    indptr_ptr,  # [N+1] int32
    prefix_len_ptr,  # [N] int
    page_idx_ptr,  # [N, Wc] int
    valid_len_ptr,  # [N] int
    swa_pages,
    Wc: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per token: write valid_len compressed slots (swa_pages+page_idx, -1 for empty) into the stream tail at indptr[t]+prefix_len[t]."""
    t = tl.program_id(0)
    cbase = tl.load(indptr_ptr + t) + tl.load(prefix_len_ptr + t).to(tl.int32)
    nc = tl.load(valid_len_ptr + t).to(tl.int32)
    for off in tl.range(0, Wc, BLOCK):
        j = off + tl.arange(0, BLOCK)
        m = j < nc
        j_clamped = tl.minimum(j, Wc - 1)
        pi = tl.load(page_idx_ptr + t * Wc + j_clamped, mask=m, other=-1).to(tl.int32)
        slot = tl.where(pi >= 0, pi + swa_pages, -1)
        tl.store(indices_ptr + cbase + j, slot, mask=m)


def fill_compress_tail(
    *,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    prefix_len: torch.Tensor,
    page_indices: torch.Tensor,  # [N, Wc] int32
    valid_len: torch.Tensor,
    swa_pages: int,
) -> None:
    N, Wc = page_indices.shape
    if N == 0:
        return
    assert prefix_len.is_contiguous() and page_indices.is_contiguous()
    assert valid_len.is_contiguous()
    _fill_compress_tail_kernel[(N,)](
        indices,
        indptr,
        prefix_len,
        page_indices,
        valid_len,
        swa_pages,
        Wc=Wc,
        BLOCK=min(1024, triton.next_power_of_2(max(Wc, 1))),
        num_warps=4,
    )


def build_decode_streams(
    *,
    state_slot: torch.Tensor,  # [N] int
    positions: torch.Tensor,  # [N] int
    swa_len: torch.Tensor,  # [N] int
    hca_len: torch.Tensor,  # [N] int
    csa_len: torch.Tensor,  # [N] int
    hca_page_indices: torch.Tensor,  # [N, hca_width] int32
    csa_width: int,
    win: int,  # SWA attention window length
    ring_stride: int,  # SWA ring per-slot stride
    swa_pages: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    device = state_slot.device
    N = state_slot.shape[0]
    assert state_slot.is_contiguous() and positions.is_contiguous()
    state_slot = state_slot.to(torch.int32)
    positions = positions.to(torch.int32)
    hca_width = hca_page_indices.shape[1]

    swa_p = _lengths_to_indptr(swa_len)
    hca_p = _lengths_to_indptr(swa_len + hca_len)
    csa_p = _lengths_to_indptr(swa_len + csa_len)

    swa_i = torch.empty(N * win, dtype=torch.int32, device=device)
    hca_i = torch.empty(N * (win + hca_width), dtype=torch.int32, device=device)
    csa_i = torch.empty(N * (win + csa_width), dtype=torch.int32, device=device)

    if N > 0:
        batch_id = torch.arange(N, dtype=torch.int32, device=device)
        write_v4_paged_decode_indices(
            state_slot_per_seq=state_slot,
            batch_id_per_token=batch_id,
            positions=positions,
            swa_indptr=swa_p,
            csa_indptr=csa_p,
            hca_indptr=hca_p,
            swa_indices=swa_i,
            csa_indices=csa_i,
            hca_indices=hca_i,
            T=N,
            win=win,
            ring_stride=ring_stride,
        )
        fill_compress_tail(
            indices=hca_i,
            indptr=hca_p,
            prefix_len=swa_len,
            page_indices=hca_page_indices[:N],
            valid_len=hca_len,
            swa_pages=swa_pages,
        )
    return swa_i, swa_p, hca_i, hca_p, csa_i, csa_p


# ---------------------------------------------------------------------------
# Prefill index builder (ragged-packed: paged prefix + flat extend)
# ---------------------------------------------------------------------------
@triton.jit
def _prefill_lengths_kernel(
    positions_ptr,  # [T] int
    chunk_start_ptr,  # [T] int
    page_idx_ptr,  # [T, Wc] int (front-packed, -1 padded)
    prefix_len_ptr,  # [T] int32 out
    extend_len_ptr,  # [T] int32 out
    win: tl.constexpr,
    Wc: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per token: write extend/prefix segment lengths"""
    t = tl.program_id(0)
    pos = tl.load(positions_ptr + t).to(tl.int32)
    cstart = tl.load(chunk_start_ptr + t).to(tl.int32)
    tpic = pos - cstart
    swa_low = tl.maximum(pos - win + 1, 0)
    extend_count = tl.minimum(tpic + 1, win)
    prefix_swa_count = tl.minimum(tl.maximum(cstart - swa_low, 0), win)
    tl.store(extend_len_ptr + t, extend_count)
    if HAS_COMPRESS:
        nc = 0
        for off in tl.range(0, Wc, BLOCK):
            j = off + tl.arange(0, BLOCK)
            m = j < Wc
            j_clamped = tl.minimum(j, Wc - 1)
            pi = tl.load(page_idx_ptr + t * Wc + j_clamped, mask=m, other=-1)
            nc += tl.sum(tl.where(m & (pi >= 0), 1, 0))
        tl.store(prefix_len_ptr + t, prefix_swa_count + nc)
    else:
        tl.store(prefix_len_ptr + t, prefix_swa_count)


@triton.jit
def _build_prefill_indices_kernel(
    positions_ptr,  # [T] int
    chunk_start_ptr,  # [T] int
    cu_q_ptr,  # [T] int
    state_slot_ptr,  # [T] int
    page_idx_ptr,  # [T, Wc] int (front-packed, -1 padded)
    pre_indptr_ptr,  # [T+1] int32 (prefix stream ragged indptr)
    ext_indptr_ptr,  # [T+1] int32 (extend stream ragged indptr)
    pre_out_ptr,
    ext_out_ptr,
    swa_pages,
    ring_stride,  # SWA ring per-slot stride
    win: tl.constexpr,
    Wc: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per token: write extend rows + prefix (SWA ring slots ++ swa_pages+compressed slots) as two ragged segments"""
    t = tl.program_id(0)
    pos = tl.load(positions_ptr + t).to(tl.int32)
    cstart = tl.load(chunk_start_ptr + t).to(tl.int32)
    cuq = tl.load(cu_q_ptr + t).to(tl.int32)
    s = tl.load(state_slot_ptr + t).to(tl.int32)

    tpic = pos - cstart
    swa_low = tl.maximum(pos - win + 1, 0)
    extend_count = tl.minimum(tpic + 1, win)
    prefix_swa_count = tl.minimum(tl.maximum(cstart - swa_low, 0), win)

    ebase = tl.load(ext_indptr_ptr + t)
    pbase = tl.load(pre_indptr_ptr + t)

    # ---- extend: rows into the current-chunk kv tensor ----
    ext_start = cuq + tpic - extend_count + 1
    for off in tl.range(0, win, BLOCK):
        k = off + tl.arange(0, BLOCK)
        m = k < extend_count
        tl.store(ext_out_ptr + ebase + k, ext_start + k, mask=m)

    # ---- prefix SWA: prior-chunk ring slots (stride = ring_stride) ----
    for off in tl.range(0, win, BLOCK):
        k = off + tl.arange(0, BLOCK)
        m = k < prefix_swa_count
        gp = swa_low + k
        tl.store(pre_out_ptr + pbase + k, s * ring_stride + (gp % ring_stride), mask=m)

    # ---- prefix compressed: swa_pages + front-packed page index ----
    if HAS_COMPRESS:
        nc = tl.load(pre_indptr_ptr + t + 1) - pbase - prefix_swa_count
        cbase = pbase + prefix_swa_count
        for off in tl.range(0, Wc, BLOCK):
            j = off + tl.arange(0, BLOCK)
            m = j < nc
            j_clamped = tl.minimum(j, Wc - 1)
            pi = tl.load(page_idx_ptr + t * Wc + j_clamped, mask=m, other=0).to(
                tl.int32
            )
            tl.store(pre_out_ptr + cbase + j, pi + swa_pages, mask=m)


def build_prefill_indices(
    *,
    compress_ratio: int,
    state_slot: torch.Tensor,  # [T] int (per token)
    positions: torch.Tensor,  # [T] int (per token absolute position)
    chunk_start: torch.Tensor,  # [T] int (absolute start of this chunk for token's seq)
    cu_q: torch.Tensor,  # [T] int (row in extend `kv` of the seq's first chunk token)
    win: int,  # SWA attention window length
    ring_stride: int,  # SWA ring per-slot stride / modulo (win_with_spec)
    swa_pages: int,
    c128_page_indices: Optional[torch.Tensor],
    c4_sparse_page_indices: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ragged prefill indices: prefix (SWA ring + swa_pages + compressed) into unified_kv + extend into current-chunk kv; returns (prefix_indices, prefix_indptr, extend_indices, extend_indptr)."""
    device = state_slot.device
    T = state_slot.shape[0]
    assert positions.is_contiguous() and chunk_start.is_contiguous()
    assert cu_q.is_contiguous() and state_slot.is_contiguous()

    if compress_ratio == 0:
        page_idx = None
    elif compress_ratio == 128:
        assert c128_page_indices is not None
        page_idx = c128_page_indices[:T]
    elif compress_ratio == 4:
        assert c4_sparse_page_indices is not None
        page_idx = c4_sparse_page_indices[:T]
    else:
        raise ValueError(f"bad compress_ratio {compress_ratio}")

    has_compress = page_idx is not None
    if has_compress:
        assert page_idx.is_contiguous()
    Wc = page_idx.shape[1] if has_compress else 0

    block = min(1024, triton.next_power_of_2(max(win, Wc, 1)))
    prefix_len = torch.empty(T, dtype=torch.int32, device=device)
    extend_len = torch.empty(T, dtype=torch.int32, device=device)
    _prefill_lengths_kernel[(T,)](
        positions,
        chunk_start,
        page_idx if has_compress else positions,  # dummy ptr when no compress
        prefix_len,
        extend_len,
        win=win,
        Wc=Wc if has_compress else 1,
        HAS_COMPRESS=has_compress,
        BLOCK=block,
        num_warps=4,
    )
    kv_indptr_prefix = _lengths_to_indptr(prefix_len)
    kv_indptr_extend = _lengths_to_indptr(extend_len)

    # OPUS reads KV in fixed KV_TILE_SIZE (64) tiles and *prefetches one tile
    # ahead* (`next_kv_page = load_kv_page(tile_idx + 1)`). On the final tile that
    # prefetch reads the index slot a whole extra tile past the valid range, and
    # the last partial tile likewise loads a full 64-slot group (masking only the
    # scores, not the index loads). With a ragged (non-64-aligned) indptr these
    # reads run up to ~2 tiles beyond a token's valid count, so the index buffers
    # need in-range padding there. Zero-fill (page 0 — a valid row, masked out of
    # the softmax) plus a two-tile pad keeps every over-read both valid and within
    # the allocation. (The aiter op_test never trips this: its CSR counts are
    # always 64-aligned, so there is no partial tile and no stray prefetch.)
    _OPUS_KV_TILE = 64
    _OPUS_PAD = 2 * _OPUS_KV_TILE
    kv_indices_prefix = torch.zeros(
        T * (win + Wc) + _OPUS_PAD, dtype=torch.int32, device=device
    )
    kv_indices_extend = torch.zeros(
        T * win + _OPUS_PAD, dtype=torch.int32, device=device
    )

    _build_prefill_indices_kernel[(T,)](
        positions,
        chunk_start,
        cu_q,
        state_slot,
        page_idx if has_compress else state_slot,  # dummy ptr when no compress
        kv_indptr_prefix,
        kv_indptr_extend,
        kv_indices_prefix,
        kv_indices_extend,
        swa_pages,
        ring_stride,
        win=win,
        Wc=Wc if has_compress else 1,
        HAS_COMPRESS=has_compress,
        BLOCK=block,
        num_warps=4,
    )
    return kv_indices_prefix, kv_indptr_prefix, kv_indices_extend, kv_indptr_extend


def prefill(
    *,
    q: torch.Tensor,  # [T, H, D]
    unified_kv: torch.Tensor,  # [pages, D] bf16, or [pages,512] fp8 (MXFP8 NoPE)
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_extend: torch.Tensor,  # [T, D] current-chunk K (bf16, norm+rope'd)
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    unified_kv_rope: Optional[torch.Tensor] = None,  # [pages,64] bf16 (fp8 KV)
    kv_extend_nope: Optional[torch.Tensor] = None,  # [T,512] fp8 pre-packed extend
    kv_extend_rope: Optional[torch.Tensor] = None,  # [T,64] bf16 pre-packed extend
) -> torch.Tensor:
    return sparse_attn_v4_paged_prefill(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_extend,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        unified_kv_rope=unified_kv_rope,
        kv_extend_nope=kv_extend_nope,
        kv_extend_rope=kv_extend_rope,
    )
