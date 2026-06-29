# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# The following kernel is imported from ATOM.
# Source: atom/model_ops/v4_kernels/paged_prefill.py

"""Sparse prefill attention with two KV sources: paged `unified_kv` (history)
and per-fwd flat `kv` (current chunk's input).

Designed for V4 prefill: indexes the two KV sources directly without
materialising a per-fwd `kv_flat_sa` packed tensor.

Caller contract:
  unified_kv:        [total_pages, D] BF16 — prefix source. Same buffer as
    decode kernel: SWA ring slots in `[0, swa_pages)`, compress pages in
    `[swa_pages, total_pages)`. For prefill, prefix indices select
    (a) prior-chunk SWA history, (b) CSA topk, (c) HCA all-committed.
  kv_indices_prefix: [total_prefix_indices] int32 — flat per-token slot
    lists. Per-token entries live in
    `kv_indices_prefix[kv_indptr_prefix[t] : kv_indptr_prefix[t+1]]`.
    `-1` entries are skipped (sentinel).
  kv_indptr_prefix:  [N+1] int32 — true prefix sum (variable per-token len).

  kv:                [total_tokens, D] BF16 — extend source = current
    fwd's just-computed K (NOT yet written to swa_kv ring). Layout matches
    `swa_write` input.
  kv_indices_extend: [total_extend_indices] int32 — flat per-token row idx
    lists into `kv`. Per-token entries live in
    `kv_indices_extend[kv_indptr_extend[t] : kv_indptr_extend[t+1]]`.
    `-1` entries are skipped (rare for extend; usually all valid).
  kv_indptr_extend:  [N+1] int32 — true prefix sum.

  attn_sink:         [H] per-head learnable softmax-denom bias (V4 specific).
  softmax_scale:     float.

Per-token K loop iterates two regions sequentially, sharing the online
softmax accumulator (m_i, l_i, acc) across regions. Order of regions does
not affect correctness (online softmax is order-invariant).

Returns:
  out: [N, H, D] same dtype as q.

Numerics: identical online-softmax + sink finalization to
`sparse_attn_v4_paged_decode` — bit-exact when the extend region is empty
(then equivalent to a decode call with the same prefix indices).
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.dsv4.unified_kv_kernels import mxfp8 as _mxfp8
from sglang.srt.layers.attention.dsv4.unified_kv_kernels.paged_decode import (
    _load_mxfp8_kv_tile,
)
from sglang.srt.utils.common import is_gfx95_supported

_MXFP8_NOPE = _mxfp8.DIM_NOPE  # 448
_MXFP8_ROPE = _mxfp8.DIM_ROPE  # 64
_MXFP8_BLK = _mxfp8.FP8_BLOCK  # 32
_MXFP8_FP8_DTYPE = _mxfp8.FP8_DTYPE  # e4m3fn (MXFP8 is gfx950-only)

# OPUS gfx950 paged-prefill kernels are preferred when importable; otherwise
# fall back to the Triton implementation below. The bf16 prefix uses
# ``pa_sparse_prefill_opus``; the MXFP8 fp8 prefix uses the split-input
# ``pa_sparse_prefill_fp8_opus`` (ROCm/aiter #3751).
try:
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus

    _HAS_OPUS = is_gfx95_supported()
except ImportError:
    pa_sparse_prefill_opus = None
    _HAS_OPUS = False

try:
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus

    _HAS_FP8_OPUS = is_gfx95_supported()
except ImportError:
    pa_sparse_prefill_fp8_opus = None
    _HAS_FP8_OPUS = False


@triton.jit
def _sparse_attn_v4_paged_prefill_kernel(
    q_ptr,  # [N, H, D]
    unified_kv_ptr,  # [total_pages, D] bf16/fp16, or [pages,512] fp8 (MXFP8) — prefix src
    unified_u8_ptr,  # unified_kv viewed as uint8 (E8M0 bytes) when QUANT_KV (dummy otherwise)
    rope_ptr,  # [total_pages, 64] bf16 prefix RoPE when QUANT_KV (dummy otherwise)
    kv_indices_prefix_ptr,  # [total_prefix_indices] int32
    kv_indptr_prefix_ptr,  # [N+1] int32
    kv_ptr,  # [total_tokens, D]    — extend source (live bf16, never fp8)
    kv_indices_extend_ptr,  # [total_extend_indices] int32
    kv_indptr_extend_ptr,  # [N+1] int32
    attn_sink_ptr,  # [H]
    out_ptr,  # [N, H, D]
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    pkv_stride_n: tl.constexpr,  # unified_kv stride 0 (= D usually)
    pkv_stride_d: tl.constexpr,  # unified_kv stride 1 (= 1 usually)
    rope_stride_n,  # row stride of prefix rope buffer (QUANT_KV only)
    ekv_stride_n: tl.constexpr,  # kv stride 0
    ekv_stride_d: tl.constexpr,  # kv stride 1
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_KV: tl.constexpr,  # True → MXFP8 dequant of the unified_kv prefix
    NOPE: tl.constexpr,  # NoPE dim (448) when QUANT_KV
    FP8_BLK: tl.constexpr,  # MXFP8 block width (32) when QUANT_KV
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q = tl.load(
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)

    # ===== Region 1: prefix from unified_kv =====
    p_start = tl.load(kv_indptr_prefix_ptr + t)
    p_end = tl.load(kv_indptr_prefix_ptr + t + 1)
    p_len = p_end - p_start

    for k_start in tl.range(0, p_len, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < p_len
        slot = tl.load(
            kv_indices_prefix_ptr + p_start + k_pos,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (slot >= 0)
        slot_clamped = tl.maximum(slot, 0)

        if QUANT_KV:
            # MXFP8: reconstruct the [BLOCK_K, 512] bf16 prefix tile (E8M0-dequant
            # 448 NoPE dims + bf16 RoPE tail) — identical layout/read as decode.
            # The extend region (Region 2) reads live bf16 kv and is unchanged.
            kv = _load_mxfp8_kv_tile(
                unified_kv_ptr,
                unified_u8_ptr,
                rope_ptr,
                slot_clamped,
                valid,
                d_offs,
                pkv_stride_n,
                rope_stride_n,
                NOPE,
                FP8_BLK,
                q.dtype,
                BLOCK_K,
                BLOCK_D,
            )
        else:
            kv = tl.load(
                unified_kv_ptr
                + slot_clamped[:, None] * pkv_stride_n
                + d_offs[None, :] * pkv_stride_d,
                mask=valid[:, None] & d_mask[None, :],
                other=0.0,
            )

        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    # ===== Region 2: extend from kv (per-fwd flat) =====
    e_start = tl.load(kv_indptr_extend_ptr + t)
    e_end = tl.load(kv_indptr_extend_ptr + t + 1)
    e_len = e_end - e_start

    for k_start in tl.range(0, e_len, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < e_len
        slot = tl.load(
            kv_indices_extend_ptr + e_start + k_pos,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (slot >= 0)
        slot_clamped = tl.maximum(slot, 0)

        kv = tl.load(
            kv_ptr
            + slot_clamped[:, None] * ekv_stride_n
            + d_offs[None, :] * ekv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    # ===== Sink finalization =====
    # Online softmax + sink integration: sink is a virtual extra K with V=0,
    # contributing only to the denominator. After main loops, (m_i, l_i, acc)
    # are in m_i frame; sink may shift max to m_final = max(m_i, sink), so
    # rescale BOTH l_i (for denom) AND acc (for numerator) by alpha to switch
    # to m_final frame. The sink itself adds exp(sink - m_final) to l_final
    # but contributes 0 to acc since V_sink = 0.
    sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    m_final = tl.maximum(m_i, sink)
    alpha = tl.exp(m_i - m_final)
    l_final = l_i * alpha + tl.exp(sink - m_final)

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, (acc * alpha[:, None]) / denom[:, None], 0.0)
    tl.store(
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out,
        mask=h_mask[:, None] & d_mask[None, :],
    )


def _sparse_attn_v4_paged_prefill_triton(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    unified_kv_rope: torch.Tensor | None = None,
) -> torch.Tensor:
    if not q.is_cuda:
        raise RuntimeError(
            "Triton sparse_attn_v4_paged_prefill requires CUDA/HIP tensors"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"sparse_attn_v4_paged_prefill expects fp16/bf16 q, got {q.dtype}"
        )

    # fp8 (MXFP8) prefix path is gated purely on the caller passing a RoPE
    # buffer (i.e. the unified pool is fp8). The bf16 path is strictly
    # unchanged. The extend region always reads the live bf16/fp16 `kv`, never
    # fp8, so only the unified-pool prefix is dequantized in-kernel.
    quant_kv = unified_kv_rope is not None
    if quant_kv:
        if unified_kv.dtype != _MXFP8_FP8_DTYPE:
            raise RuntimeError(
                f"MXFP8 unified_kv_rope supplied but unified_kv is "
                f"{unified_kv.dtype}, expected {_MXFP8_FP8_DTYPE}"
            )
        if unified_kv.shape[-1] != 512:
            raise RuntimeError(
                f"MXFP8 NoPE buffer must be [pages, 512], got "
                f"{tuple(unified_kv.shape)}"
            )
        if unified_kv_rope.shape[-1] != _MXFP8_ROPE:
            raise RuntimeError(
                f"RoPE buffer must be [pages, {_MXFP8_ROPE}], got "
                f"{tuple(unified_kv_rope.shape)}"
            )
        if unified_kv_rope.dtype != q.dtype:
            raise RuntimeError(
                f"RoPE buffer dtype {unified_kv_rope.dtype} must match q "
                f"dtype {q.dtype}"
            )
        if unified_kv.stride(-1) != 1:
            unified_kv = unified_kv.contiguous()
        if unified_kv_rope.stride(-1) != 1:
            unified_kv_rope = unified_kv_rope.contiguous()
    else:
        if unified_kv.dtype != q.dtype:
            raise RuntimeError(
                f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
            )
    if kv.dtype != q.dtype:
        raise RuntimeError(f"kv dtype mismatch: kv={kv.dtype}, q={q.dtype}")
    if kv.size(-1) != q.size(-1):
        raise RuntimeError(
            f"head_dim mismatch: q={q.size(-1)}, kv={kv.size(-1)}"
        )

    T, H, D = q.shape
    out = torch.empty_like(q)
    kv_indices_prefix = kv_indices_prefix.to(torch.int32).contiguous()
    kv_indptr_prefix = kv_indptr_prefix.to(torch.int32).contiguous()
    kv_indices_extend = kv_indices_extend.to(torch.int32).contiguous()
    kv_indptr_extend = kv_indptr_extend.to(torch.int32).contiguous()

    block_h = 16  # AMD MFMA min tile
    block_d = triton.next_power_of_2(D)
    block_k = 32 if quant_kv else (16 if D >= 256 else 32)

    # The kernel reads (unified_u8_ptr, rope_ptr, rope_stride_n) only when
    # QUANT_KV — supply dummy 1-element tensors on the bf16 path so the launch
    # signature stays uniform (avoids a separate JIT specialization per call).
    if quant_kv:
        unified_u8_arg = unified_kv.view(torch.uint8)
        rope_arg = unified_kv_rope
        rope_stride_n_arg = unified_kv_rope.stride(0)
    else:
        unified_u8_arg = unified_kv  # unused at compile time (QUANT_KV False)
        rope_arg = q.new_empty(1)
        rope_stride_n_arg = 1

    _sparse_attn_v4_paged_prefill_kernel[(T, triton.cdiv(H, block_h))](
        q,
        unified_kv,
        unified_u8_arg,
        rope_arg,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        unified_kv.stride(0),
        unified_kv.stride(1),
        rope_stride_n_arg,
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        QUANT_KV=quant_kv,
        NOPE=_MXFP8_NOPE,
        FP8_BLK=_MXFP8_BLK,
        num_warps=4,
    )
    return out


def sparse_attn_v4_paged_prefill(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    unified_kv_rope: torch.Tensor | None = None,
    kv_extend_nope: torch.Tensor | None = None,
    kv_extend_rope: torch.Tensor | None = None,
    q_nope: torch.Tensor | None = None,
    q_rope: torch.Tensor | None = None,
) -> torch.Tensor:
    """V4 prefill sparse attention over two KV sources (paged unified_kv +
    flat per-fwd kv).

    Args:
      q:                 [T, H, D] BF16/FP16 — query (D == 512: 448 NoPE + 64 RoPE).
      unified_kv:        prefix source (paged). bf16 ``[pages, D]`` in the
        default path; when ``unified_kv_rope`` is provided this is the upstream
        MXFP8 NoPE pack ``[pages, 512]`` fp8 (448 fp8 + 14 E8M0 + 50 pad).
      kv_indices_prefix: [total_prefix] int32 — flat per-token slot lists into
        unified_kv. -1 sentinels skipped.
      kv_indptr_prefix:  [T+1] int32 — true prefix sum.
      kv:                [total_tokens, D] BF16/FP16 — extend source (this
        fwd's input K, NOT yet in swa_kv ring). Always bf16/fp16; the fp8 OPUS
        op packs it on the fly into its split NoPE-fp8 / RoPE-bf16 inputs.
      kv_indices_extend: [total_extend] int32 — flat per-token row idx lists
        into kv. -1 sentinels skipped.
      kv_indptr_extend:  [T+1] int32 — true prefix sum.
      attn_sink:         [H] — per-head softmax-denom bias.
      softmax_scale:     float.
      unified_kv_rope:   [total_pages, 64] bf16 — RoPE companion buffer for the
        MXFP8 unified pool. None → bf16 prefix (default path).

    Returns:
      out: [T, H, D] same dtype as q.
    """
    quant_kv = unified_kv_rope is not None

    # MXFP8 fp8 fast path: upstream split-input ``pa_sparse_prefill_fp8_opus``
    # (#3751). It consumes the pool's NoPE-fp8 / RoPE-bf16 buffers directly and
    # packs the per-step q + extend kv on the fly (no prefix repack).
    if quant_kv and _HAS_FP8_OPUS:
        from sglang.srt.layers.attention.dsv4.unified_kv_kernels import runtime as _rt

        # Fused MXFP8 pack of the per-step q + extend kv (one Triton launch each,
        # replacing the eager split+pack+cast chains). The prefix pool is already
        # native MXFP8, so only these per-call tensors are packed. The extend kv
        # pack is reused by the SWA store when the caller pre-packs it. When the
        # caller pre-packs q (fused q norm+rope+pack producer), reuse those bytes
        # and skip the standalone q re-read/pack entirely.
        if q_nope is None:
            q = q.contiguous()
            q_nope, q_rope = _rt.pack_mxfp8_dense(q)
        H = q_nope.shape[1]
        sink = attn_sink
        if sink.shape[0] != H:
            sink = sink[:H]
        sink = sink.to(torch.float32).contiguous()
        if kv_extend_nope is not None:
            kv_nope, kv_rope = kv_extend_nope, kv_extend_rope
        else:
            kv_nope, kv_rope = _rt.pack_mxfp8_dense(kv)
        # A single-row [1, 64] RoPE slice keeps its parent's row stride (512)
        # even after .contiguous() — torch treats a size-1 leading dim as already
        # contiguous and skips the copy. The OPUS op asserts the extend RoPE page
        # stride matches the prefix pool's (64), so force a width-equal row stride
        # whenever it diverges (only possible for shape[0] <= 1, where stride(0)
        # is never actually used for addressing). Mirrors the bf16 path's size-1
        # kv fixup above.
        if kv_rope.stride(0) != kv_rope.shape[-1]:
            kv_rope = kv_rope.as_strided(kv_rope.shape, (kv_rope.shape[-1], 1))
        if kv_nope.stride(0) != kv_nope.shape[-1]:
            kv_nope = kv_nope.as_strided(kv_nope.shape, (kv_nope.shape[-1], 1))
        return pa_sparse_prefill_fp8_opus(
            q_nope,
            q_rope,
            unified_kv,
            unified_kv_rope,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv_nope,
            kv_rope,
            kv_indices_extend,
            kv_indptr_extend,
            sink,
            softmax_scale,
        )

    # bf16 OPUS fast path (default, non-fp8 pool). Handles any head count: the
    # aiter kernel dispatches H<=32 to the 16mx1_16nx4 variant and H>32 (DP
    # attention carries all heads per rank) to the 16mx8_32nx1 variant.
    if (not quant_kv) and _HAS_OPUS:
        # OPUS contract differs from the Triton kernel in two ways the Triton
        # path tolerates implicitly:
        #  - it requires a FULLY-contiguous q (it only asserts stride(2)==1 but
        #    indexes assuming [T,H,D] contiguous); a non-contiguous head stride
        #    silently reads wrong/out-of-bounds addresses. q from the model is
        #    often a view, so force contiguity.
        #  - it requires ``attn_sink.size(0) == H``; attn_sink is the full
        #    per-head Parameter, so slice to the H query heads.
        q = q.contiguous()
        H = q.shape[1]
        if attn_sink.shape[0] != H:
            attn_sink = attn_sink[:H].contiguous()
        if (
            kv.stride(0) != unified_kv.stride(0)
            and kv.shape[0] == 1
            and kv.stride(1) == 1
        ):
            kv = kv.as_strided(kv.shape, (kv.shape[1], 1))
        return pa_sparse_prefill_opus(
            q,
            unified_kv,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            softmax_scale,
        )

    # Triton fallback (OPUS absent). The MXFP8 prefix is dequantized in-kernel
    # (NoPE E8M0 + bf16 RoPE -> bf16 tile) and the extend region reads live
    # bf16 kv; q stays full bf16.
    return _sparse_attn_v4_paged_prefill_triton(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        unified_kv_rope=unified_kv_rope,
    )
