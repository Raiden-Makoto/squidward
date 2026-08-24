"""Triton sparse-MLA forward for the DSA fp8 path.

Two strategies, auto-selected by sequence length:
  1. Single-pass: grid=(seq,), best when seq is large enough to fill CUs.
  2. Split-K: grid=(seq, head_blocks, kv_splits) + reduce, best for short
     sequences (MTP verify/draft with seq=1-6) where single-pass starves the GPU.

Both use the split-dim pattern: D_V processed in NUM_GROUPS chunks of 128
for native CDNA4 fp8 MFMA tile alignment.
"""

import functools
from typing import TypeAlias

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

_IS_FNUZ = is_fp8_fnuz()
_FP8_MAX = 240.0 if _IS_FNUZ else 448.0
_LOG2E = 1.4426950408889634
_G = tl.constexpr(128)
_MXFP4_GROUP_SIZE = 32
_MXFP4_VALUES_PER_BYTE = 2

PackedMXFP4: TypeAlias = tuple[torch.Tensor, torch.Tensor]
SparseMLAOutput: TypeAlias = torch.Tensor | PackedMXFP4


@triton.jit
def _quantize_mxfp4_group(x):
    """Bit-compatible specialization of AITER's MXFP4 quantizer for x[:, 32]."""
    amax = tl.max(tl.abs(x), axis=1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_unbiased = tl.log2(amax).floor() - 2
    scale_unbiased = tl.clamp(scale_unbiased, min=-127, max=127)
    scale_e8m0 = scale_unbiased.to(tl.uint8) + 127

    qx = x * tl.exp2(-scale_unbiased)
    qx = qx.to(tl.uint32, bitcast=True)
    sign = qx & 0x80000000
    qx = qx ^ sign

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= 6.0
    denormal_mask = (not saturate_mask) & (qx_fp32 < 1.0)
    normal_mask = not (saturate_mask | denormal_mask)

    denorm_exp: tl.constexpr = 149
    denorm_mask_int: tl.constexpr = denorm_exp << 23
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)
    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    normal_x = qx
    mant_odd = (normal_x >> 22) & 1
    normal_x += ((1 - 127) << 23) + (1 << 21) - 1
    normal_x += mant_odd
    normal_x = (normal_x >> 22).to(tl.uint8)

    code = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    code = tl.where(normal_mask, normal_x, code)
    code = tl.where(denormal_mask, denormal_x, code)
    code |= (sign >> 28).to(tl.uint8)

    code = code.reshape([x.shape[0], 16, 2])
    evens, odds = tl.split(code)
    return evens | (odds << 4), scale_e8m0[:, 0]


@triton.jit
def _store_mxfp4_group(
    values_ptr,
    scales_ptr,
    values,
    token,
    heads,
    head_mask,
    group_in_chunk: tl.constexpr,
    output_group,
    T,
):
    """Quantize one normalized FP32 group of 32 values and store head-major."""
    group_values = values[
        :,
        group_in_chunk * _MXFP4_GROUP_SIZE : (group_in_chunk + 1) * _MXFP4_GROUP_SIZE,
    ]
    packed, scale_e8m0 = _quantize_mxfp4_group(group_values)
    pair = tl.arange(0, _MXFP4_GROUP_SIZE // _MXFP4_VALUES_PER_BYTE)

    value_group = output_group * (_MXFP4_GROUP_SIZE // _MXFP4_VALUES_PER_BYTE)
    value_base = heads[:, None] * T * 256 + token * 256 + value_group
    scale_base = heads * T * 16 + token * 16 + output_group
    tl.store(values_ptr + value_base + pair[None, :], packed, mask=head_mask[:, None])
    tl.store(scales_ptr + scale_base, scale_e8m0, mask=head_mask)


def _allocate_output(
    q_nope: torch.Tensor,
    seq: int,
    heads: int,
    d_v: int,
    return_mxfp4: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if return_mxfp4:
        assert d_v == 512, "packed MXFP4 sparse MLA output requires d_v=512"
        values = torch.empty(
            heads, seq, d_v // 2, device=q_nope.device, dtype=torch.uint8
        )
        scales = torch.empty(
            heads,
            seq,
            d_v // _MXFP4_GROUP_SIZE,
            device=q_nope.device,
            dtype=torch.uint8,
        )
        return values, scales
    return (
        torch.empty(seq, heads, d_v, device=q_nope.device, dtype=torch.bfloat16),
        None,
    )


def _format_output(
    out: torch.Tensor, scales: torch.Tensor | None, return_mxfp4: bool
) -> SparseMLAOutput:
    if return_mxfp4:
        assert scales is not None
        return out, scales
    return out.unsqueeze(0)


# ---------------------------------------------------------------------------
# Helper functions for split-K heuristic
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _cu_count() -> int:
    from aiter.ops.triton.utils.device_info import get_num_sms

    return get_num_sms()


def _kv_splits_heuristic(
    T: int,
    H: int,
    block_h: int,
    num_cu: int | None = None,
    target_wg_per_cu: float = 2.0,
    max_kv_splits: int = 64,
) -> int:
    if num_cu is None:
        num_cu = _cu_count()
    target_wg = max(1, int(target_wg_per_cu * num_cu))
    head_blocks = max(1, (H + block_h - 1) // block_h)
    base_ctas = max(1, T * head_blocks)
    if base_ctas >= target_wg:
        return 1
    splits_to_fill = max(1, target_wg // base_ctas)
    return _prev_pow2(min(splits_to_fill, max_kv_splits))


def _row_strides(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Return (tensor, token_stride, head_stride) for a [N, H, D] q tensor.

    The kernels address q by explicit row strides, so a packed [N, H, D] layout is
    not required -- only a unit-stride last dim. Callers that pass an already-
    concatenated q (dsa_backend, GLM-5.2 path) hand us two strided views of one
    [N, H, D_V + D_TAIL] buffer; copying those would cost two extra device kernels
    per layer per forward for nothing. The fallback keeps the kernels' `+ g` / `+ dt`
    addressing valid for exotic layouts -- no caller hits it today.
    """
    if x.stride(-1) != 1:
        x = x.contiguous()
    return x, x.stride(0), x.stride(1)


def _prune_configs(configs, named_args, **kwargs):
    """Drop configs whose KV tile exceeds topk (pure waste)."""
    topk = named_args["topk"]
    keep = [c for c in configs if c.kwargs["BLOCK_N"] <= topk]
    return keep or [configs[0]]


# ---------------------------------------------------------------------------
# Single-pass split-dim kernel (autotuned, for long sequences)
# grid=(seq,), processes D_V in NUM_GROUPS chunks of 128
# ---------------------------------------------------------------------------

_SPLIT_DIM_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=ns)
    for bn in (32, 64)
    for w in (2, 4)
    for ns in (1, 2)
]


@triton.autotune(
    configs=_SPLIT_DIM_CONFIGS,
    key=["topk", "H"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _sparse_mla_fwd_split_dim_kernel(
    q_nope_ptr,  # [seq, H, D_V]   fp8
    q_rope_ptr,  # [seq, H, D_TAIL] fp8
    kv_ptr,  # [num_pages, 1, DIM] fp8
    idx_ptr,  # [seq, topk]      int32
    o_ptr,
    o_scale_ptr,
    qk_scale,  # sm_scale * LOG2E (prescaled for exp2)
    fp8_max,
    topk,
    H: tl.constexpr,
    DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    T,
    RETURN_MXFP4: tl.constexpr,
    STRIDE_QN_T: tl.constexpr,
    STRIDE_QN_H: tl.constexpr,
    STRIDE_QR_T: tl.constexpr,
    STRIDE_QR_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    s_i = tl.program_id(0)

    h = tl.arange(0, H)
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, _G)

    q_row = q_nope_ptr + s_i * STRIDE_QN_T + h[:, None] * STRIDE_QN_H
    q0 = tl.load(q_row + g[None, :]).to(q_nope_ptr.dtype.element_ty)
    if NUM_GROUPS >= 2:
        q1 = tl.load(q_row + (_G + g)[None, :]).to(q_nope_ptr.dtype.element_ty)
    if NUM_GROUPS >= 3:
        q2 = tl.load(q_row + (2 * _G + g)[None, :]).to(q_nope_ptr.dtype.element_ty)
    if NUM_GROUPS >= 4:
        q3 = tl.load(q_row + (3 * _G + g)[None, :]).to(q_nope_ptr.dtype.element_ty)
    q_tail = tl.load(
        q_rope_ptr + s_i * STRIDE_QR_T + h[:, None] * STRIDE_QR_H + dt[None, :]
    ).to(q_nope_ptr.dtype.element_ty)

    neg_large = -3.4028234663852886e38
    m_i = tl.full([H], neg_large, tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc0 = tl.zeros([H, _G], tl.float32)
    if NUM_GROUPS >= 2:
        acc1 = tl.zeros([H, _G], tl.float32)
    if NUM_GROUPS >= 3:
        acc2 = tl.zeros([H, _G], tl.float32)
    if NUM_GROUPS >= 4:
        acc3 = tl.zeros([H, _G], tl.float32)

    inv_fp8_max = 1.0 / fp8_max
    n = tl.arange(0, BLOCK_N)
    for k0 in range(0, topk, BLOCK_N):
        kmask = (k0 + n) < topk
        idx = tl.load(idx_ptr + s_i * topk + k0 + n, mask=kmask, other=-1)
        valid = (idx >= 0) & kmask
        page = tl.where(valid, idx, 0).to(tl.int64)
        kbase = kv_ptr + page[:, None] * DIM

        kv0 = tl.load(kbase + g[None, :], mask=valid[:, None], other=0.0).to(
            q_nope_ptr.dtype.element_ty
        )
        if NUM_GROUPS >= 2:
            kv1 = tl.load(kbase + (_G + g)[None, :], mask=valid[:, None], other=0.0).to(
                q_nope_ptr.dtype.element_ty
            )
        if NUM_GROUPS >= 3:
            kv2 = tl.load(
                kbase + (2 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(q_nope_ptr.dtype.element_ty)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(
                kbase + (3 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(q_nope_ptr.dtype.element_ty)
        kv_tail = tl.load(
            kbase + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
        ).to(q_nope_ptr.dtype.element_ty)

        qk = tl.dot(q0, tl.trans(kv0))
        if NUM_GROUPS >= 2:
            qk += tl.dot(q1, tl.trans(kv1))
        if NUM_GROUPS >= 3:
            qk += tl.dot(q2, tl.trans(kv2))
        if NUM_GROUPS >= 4:
            qk += tl.dot(q3, tl.trans(kv3))
        qk += tl.dot(q_tail, tl.trans(kv_tail))
        qk = qk * qk_scale
        qk = tl.where(valid[None, :], qk, neg_large)

        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_fp8 = (p * fp8_max).to(q_nope_ptr.dtype.element_ty)
        acc0 = acc0 * alpha[:, None] + tl.dot(p_fp8, kv0).to(tl.float32) * inv_fp8_max
        if NUM_GROUPS >= 2:
            acc1 = (
                acc1 * alpha[:, None] + tl.dot(p_fp8, kv1).to(tl.float32) * inv_fp8_max
            )
        if NUM_GROUPS >= 3:
            acc2 = (
                acc2 * alpha[:, None] + tl.dot(p_fp8, kv2).to(tl.float32) * inv_fp8_max
            )
        if NUM_GROUPS >= 4:
            acc3 = (
                acc3 * alpha[:, None] + tl.dot(p_fp8, kv3).to(tl.float32) * inv_fp8_max
            )
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    inv_l = 1.0 / l_safe
    acc0 = acc0 * inv_l[:, None]
    if NUM_GROUPS >= 2:
        acc1 = acc1 * inv_l[:, None]
    if NUM_GROUPS >= 3:
        acc2 = acc2 * inv_l[:, None]
    if NUM_GROUPS >= 4:
        acc3 = acc3 * inv_l[:, None]

    if RETURN_MXFP4:
        for group in tl.static_range(4):
            _store_mxfp4_group(o_ptr, o_scale_ptr, acc0, s_i, h, h < H, group, group, T)
        if NUM_GROUPS >= 2:
            for group in tl.static_range(4):
                _store_mxfp4_group(
                    o_ptr, o_scale_ptr, acc1, s_i, h, h < H, group, 4 + group, T
                )
        if NUM_GROUPS >= 3:
            for group in tl.static_range(4):
                _store_mxfp4_group(
                    o_ptr, o_scale_ptr, acc2, s_i, h, h < H, group, 8 + group, T
                )
        if NUM_GROUPS >= 4:
            for group in tl.static_range(4):
                _store_mxfp4_group(
                    o_ptr, o_scale_ptr, acc3, s_i, h, h < H, group, 12 + group, T
                )
    else:
        o_base = o_ptr + s_i * H * D_V
        tl.store(
            o_base + h[:, None] * D_V + g[None, :],
            acc0.to(o_ptr.dtype.element_ty),
        )
        if NUM_GROUPS >= 2:
            tl.store(
                o_base + h[:, None] * D_V + (_G + g)[None, :],
                acc1.to(o_ptr.dtype.element_ty),
            )
        if NUM_GROUPS >= 3:
            tl.store(
                o_base + h[:, None] * D_V + (2 * _G + g)[None, :],
                acc2.to(o_ptr.dtype.element_ty),
            )
        if NUM_GROUPS >= 4:
            tl.store(
                o_base + h[:, None] * D_V + (3 * _G + g)[None, :],
                acc3.to(o_ptr.dtype.element_ty),
            )


def _triton_sparse_mla_fwd_single(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    return_mxfp4: bool = False,
) -> SparseMLAOutput:
    """Single-pass prefill: grid=(seq,), loops over all topk per CTA."""
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    assert d_v % 128 == 0, f"Triton sparse MLA requires d_v divisible by 128, got {d_v}"
    num_groups = d_v // 128
    assert (
        num_groups <= 4
    ), f"Triton sparse MLA supports d_v up to 512 (4 groups), got d_v={d_v}"
    d_tail = q_rope.shape[-1]
    dim = kv.shape[-1]
    topk = indices.shape[-1]
    q_nope, stride_qn_t, stride_qn_h = _row_strides(q_nope)
    q_rope, stride_qr_t, stride_qr_h = _row_strides(q_rope)
    idx_flat = indices.squeeze(1).contiguous() if indices.dim() == 3 else indices
    out, out_scales = _allocate_output(q_nope, seq, H, d_v, return_mxfp4)
    qk_scale = float(sm_scale) * _LOG2E
    if H < 16:
        # Pad H to 16 so fp8 tl.dot maps to native MFMA tiles on CDNA4.
        # Without padding, M=H<16 fp8 dots fall back to a scalar path.
        H_pad = 16
        q_nope_pad = torch.zeros(
            seq, H_pad, d_v, device=q_nope.device, dtype=q_nope.dtype
        )
        q_rope_pad = torch.zeros(
            seq, H_pad, d_tail, device=q_rope.device, dtype=q_rope.dtype
        )
        q_nope_pad[:, :H, :] = q_nope
        q_rope_pad[:, :H, :] = q_rope
        # Freshly allocated and packed; re-read the strides for the padded shape.
        q_nope_pad, stride_qn_t, stride_qn_h = _row_strides(q_nope_pad)
        q_rope_pad, stride_qr_t, stride_qr_h = _row_strides(q_rope_pad)
        # Small-head padding is a BF16-only fallback. Packed production uses
        # native H>=16 MFMA geometry and avoids slice/copy output kernels.
        assert not return_mxfp4
        out_pad, _ = _allocate_output(q_nope, seq, H_pad, d_v, False)
        _sparse_mla_fwd_split_dim_kernel[(seq,)](
            q_nope_pad,
            q_rope_pad,
            kv,
            idx_flat,
            out_pad,
            out_pad,
            qk_scale,
            _FP8_MAX,
            topk,
            H=H_pad,
            DIM=dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=num_groups,
            T=seq,
            RETURN_MXFP4=False,
            STRIDE_QN_T=stride_qn_t,
            STRIDE_QN_H=stride_qn_h,
            STRIDE_QR_T=stride_qr_t,
            STRIDE_QR_H=stride_qr_h,
        )
        out = out_pad[:, :H, :].contiguous()
    else:
        _sparse_mla_fwd_split_dim_kernel[(seq,)](
            q_nope,
            q_rope,
            kv,
            idx_flat,
            out,
            out_scales if out_scales is not None else out,
            qk_scale,
            _FP8_MAX,
            topk,
            H=H,
            DIM=dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=num_groups,
            T=seq,
            RETURN_MXFP4=return_mxfp4,
            STRIDE_QN_T=stride_qn_t,
            STRIDE_QN_H=stride_qn_h,
            STRIDE_QR_T=stride_qr_t,
            STRIDE_QR_H=stride_qr_h,
        )
    return _format_output(out, out_scales, return_mxfp4)


def _prev_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


# ---------------------------------------------------------------------------
# Split-K kernels (for short sequences: MTP verify/draft, decode)
# grid=(seq, head_blocks, kv_splits) + reduce
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_mla_fused_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    out_ptr,
    out_scale_ptr,
    qk_scale,
    fp8_max,
    topk: tl.constexpr,
    H: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    T,
    RETURN_MXFP4: tl.constexpr,
    STRIDE_QN_T: tl.constexpr,
    STRIDE_QN_H: tl.constexpr,
    STRIDE_QR_T: tl.constexpr,
    STRIDE_QR_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Single-pass with head-block splitting. grid=(seq, head_blocks)."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, _G)

    fp8_type = q_nope_ptr.dtype.element_ty
    inv_fp8_max = 1.0 / fp8_max

    qn_row = q_nope_ptr + t * STRIDE_QN_T + h_offs[:, None] * STRIDE_QN_H
    q0 = tl.load(qn_row + g[None, :], mask=h_mask[:, None], other=0.0).to(fp8_type)
    if NUM_GROUPS >= 2:
        q1 = tl.load(
            qn_row + (_G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    if NUM_GROUPS >= 3:
        q2 = tl.load(
            qn_row + (2 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    if NUM_GROUPS >= 4:
        q3 = tl.load(
            qn_row + (3 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    q_tail = tl.load(
        q_rope_ptr + t * STRIDE_QR_T + h_offs[:, None] * STRIDE_QR_H + dt[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(fp8_type)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 2:
        acc1 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 3:
        acc2 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 4:
        acc3 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    num_tiles = tl.cdiv(topk, BLOCK_K)

    for j in tl.range(0, num_tiles, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < topk

        slot = tl.load(idx_ptr + t * topk + k_pos, mask=valid, other=0)
        valid = valid & (slot >= 0)
        page = tl.where(valid, slot, 0).to(tl.int64)

        kv_base = kv_ptr + page[:, None] * KV_DIM
        kv0 = tl.load(kv_base + g[None, :], mask=valid[:, None], other=0.0).to(fp8_type)
        if NUM_GROUPS >= 2:
            kv1 = tl.load(
                kv_base + (_G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(fp8_type)
        if NUM_GROUPS >= 3:
            kv2 = tl.load(
                kv_base + (2 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(fp8_type)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(
                kv_base + (3 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(fp8_type)
        kv_tail = tl.load(
            kv_base + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
        ).to(fp8_type)

        scores = tl.dot(q0, tl.trans(kv0))
        if NUM_GROUPS >= 2:
            scores += tl.dot(q1, tl.trans(kv1))
        if NUM_GROUPS >= 3:
            scores += tl.dot(q2, tl.trans(kv2))
        if NUM_GROUPS >= 4:
            scores += tl.dot(q3, tl.trans(kv3))
        scores += tl.dot(q_tail, tl.trans(kv_tail))
        scores = scores * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_fp8 = (p * fp8_max).to(fp8_type)
        acc0 = acc0 * alpha[:, None] + tl.dot(p_fp8, kv0).to(tl.float32) * inv_fp8_max
        if NUM_GROUPS >= 2:
            acc1 = (
                acc1 * alpha[:, None] + tl.dot(p_fp8, kv1).to(tl.float32) * inv_fp8_max
            )
        if NUM_GROUPS >= 3:
            acc2 = (
                acc2 * alpha[:, None] + tl.dot(p_fp8, kv2).to(tl.float32) * inv_fp8_max
            )
        if NUM_GROUPS >= 4:
            acc3 = (
                acc3 * alpha[:, None] + tl.dot(p_fp8, kv3).to(tl.float32) * inv_fp8_max
            )
        m_i = m_new

    denom = tl.maximum(l_i, 1.0e-30)
    inv_denom = 1.0 / denom
    acc0 = tl.where(l_i[:, None] > 0.0, acc0 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 2:
        acc1 = tl.where(l_i[:, None] > 0.0, acc1 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 3:
        acc2 = tl.where(l_i[:, None] > 0.0, acc2 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 4:
        acc3 = tl.where(l_i[:, None] > 0.0, acc3 * inv_denom[:, None], 0.0)

    if RETURN_MXFP4:
        for group in tl.static_range(4):
            _store_mxfp4_group(
                out_ptr, out_scale_ptr, acc0, t, h_offs, h_mask, group, group, T
            )
        if NUM_GROUPS >= 2:
            for group in tl.static_range(4):
                _store_mxfp4_group(
                    out_ptr,
                    out_scale_ptr,
                    acc1,
                    t,
                    h_offs,
                    h_mask,
                    group,
                    4 + group,
                    T,
                )
        if NUM_GROUPS >= 3:
            for group in tl.static_range(4):
                _store_mxfp4_group(
                    out_ptr,
                    out_scale_ptr,
                    acc2,
                    t,
                    h_offs,
                    h_mask,
                    group,
                    8 + group,
                    T,
                )
        if NUM_GROUPS >= 4:
            for group in tl.static_range(4):
                _store_mxfp4_group(
                    out_ptr,
                    out_scale_ptr,
                    acc3,
                    t,
                    h_offs,
                    h_mask,
                    group,
                    12 + group,
                    T,
                )
    else:
        o_base = out_ptr + t * H * D_V
        tl.store(
            o_base + h_offs[:, None] * D_V + g[None, :],
            acc0.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
        if NUM_GROUPS >= 2:
            tl.store(
                o_base + h_offs[:, None] * D_V + (_G + g)[None, :],
                acc1.to(tl.bfloat16),
                mask=h_mask[:, None],
            )
        if NUM_GROUPS >= 3:
            tl.store(
                o_base + h_offs[:, None] * D_V + (2 * _G + g)[None, :],
                acc2.to(tl.bfloat16),
                mask=h_mask[:, None],
            )
        if NUM_GROUPS >= 4:
            tl.store(
                o_base + h_offs[:, None] * D_V + (3 * _G + g)[None, :],
                acc3.to(tl.bfloat16),
                mask=h_mask[:, None],
            )


@triton.jit
def _sparse_mla_split_k_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    lse_partial_ptr,
    acc_partial_ptr,
    qk_scale,
    fp8_max,
    topk: tl.constexpr,
    H: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    STRIDE_QN_T: tl.constexpr,
    STRIDE_QN_H: tl.constexpr,
    STRIDE_QR_T: tl.constexpr,
    STRIDE_QR_H: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Split-K partial kernel. grid=(seq, head_blocks, kv_splits)."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, _G)

    fp8_type = q_nope_ptr.dtype.element_ty
    inv_fp8_max = 1.0 / fp8_max

    qn_row = q_nope_ptr + t * STRIDE_QN_T + h_offs[:, None] * STRIDE_QN_H
    q0 = tl.load(qn_row + g[None, :], mask=h_mask[:, None], other=0.0).to(fp8_type)
    if NUM_GROUPS >= 2:
        q1 = tl.load(
            qn_row + (_G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    if NUM_GROUPS >= 3:
        q2 = tl.load(
            qn_row + (2 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    if NUM_GROUPS >= 4:
        q3 = tl.load(
            qn_row + (3 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    q_tail = tl.load(
        q_rope_ptr + t * STRIDE_QR_T + h_offs[:, None] * STRIDE_QR_H + dt[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(fp8_type)

    tiles_per_segment = tl.cdiv(topk, KV_SPLITS * BLOCK_K)
    if pid_k * tiles_per_segment * BLOCK_K >= topk:
        return
    num_tiles = tl.cdiv(topk, BLOCK_K)
    tile_start = pid_k * tiles_per_segment
    tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 2:
        acc1 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 3:
        acc2 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 4:
        acc3 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    for j in tl.range(tile_start, tile_end, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < topk

        slot = tl.load(idx_ptr + t * topk + k_pos, mask=valid, other=0)
        valid = valid & (slot >= 0)
        page = tl.where(valid, slot, 0).to(tl.int64)

        kv_base = kv_ptr + page[:, None] * KV_DIM
        kv0 = tl.load(kv_base + g[None, :], mask=valid[:, None], other=0.0).to(fp8_type)
        if NUM_GROUPS >= 2:
            kv1 = tl.load(
                kv_base + (_G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(fp8_type)
        if NUM_GROUPS >= 3:
            kv2 = tl.load(
                kv_base + (2 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(fp8_type)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(
                kv_base + (3 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(fp8_type)
        kv_tail = tl.load(
            kv_base + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
        ).to(fp8_type)

        scores = tl.dot(q0, tl.trans(kv0))
        if NUM_GROUPS >= 2:
            scores += tl.dot(q1, tl.trans(kv1))
        if NUM_GROUPS >= 3:
            scores += tl.dot(q2, tl.trans(kv2))
        if NUM_GROUPS >= 4:
            scores += tl.dot(q3, tl.trans(kv3))
        scores += tl.dot(q_tail, tl.trans(kv_tail))
        scores = scores * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        p_fp8 = (p * fp8_max).to(fp8_type)
        acc0 = acc0 * alpha[:, None] + tl.dot(p_fp8, kv0).to(tl.float32) * inv_fp8_max
        if NUM_GROUPS >= 2:
            acc1 = (
                acc1 * alpha[:, None] + tl.dot(p_fp8, kv1).to(tl.float32) * inv_fp8_max
            )
        if NUM_GROUPS >= 3:
            acc2 = (
                acc2 * alpha[:, None] + tl.dot(p_fp8, kv2).to(tl.float32) * inv_fp8_max
            )
        if NUM_GROUPS >= 4:
            acc3 = (
                acc3 * alpha[:, None] + tl.dot(p_fp8, kv3).to(tl.float32) * inv_fp8_max
            )
        m_i = m_new
        l_i = l_new

    neg_large = -3.4028234663852886e38
    denom = tl.maximum(l_i, 1.0e-30)
    inv_denom = 1.0 / denom
    has_data = l_i > 0.0
    acc0 = tl.where(has_data[:, None], acc0 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 2:
        acc1 = tl.where(has_data[:, None], acc1 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 3:
        acc2 = tl.where(has_data[:, None], acc2 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 4:
        acc3 = tl.where(has_data[:, None], acc3 * inv_denom[:, None], 0.0)

    lse = tl.where(has_data, tl.log2(l_i) + m_i, neg_large)

    H_padded = tl.cdiv(H, BLOCK_H) * BLOCK_H
    lse_base = t * KV_SPLITS * H_padded + pid_k * H_padded
    tl.store(lse_partial_ptr + lse_base + h_offs, lse, mask=h_mask)

    ap_base = t * KV_SPLITS * H_padded * D_V + pid_k * H_padded * D_V
    tl.store(
        acc_partial_ptr + ap_base + h_offs[:, None] * D_V + g[None, :],
        acc0.to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    if NUM_GROUPS >= 2:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (_G + g)[None, :],
            acc1.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
    if NUM_GROUPS >= 3:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (2 * _G + g)[None, :],
            acc2.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
    if NUM_GROUPS >= 4:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (3 * _G + g)[None, :],
            acc3.to(tl.bfloat16),
            mask=h_mask[:, None],
        )


@triton.jit
def _sparse_mla_reduce_kernel(
    lse_partial_ptr,
    acc_partial_ptr,
    out_ptr,
    out_scale_ptr,
    H: tl.constexpr,
    D_V: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    ACTIVE_SPLITS: tl.constexpr,
    D_CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    T,
    RETURN_MXFP4: tl.constexpr,
):
    """Reduce split-K partials via log-space combine. grid=(seq, H, d_v_chunks)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    dc = tl.program_id(2)

    d_offs = dc * D_CHUNK + tl.arange(0, D_CHUNK)
    k_offs = tl.arange(0, ACTIVE_SPLITS)
    d_mask = d_offs < D_V

    H_padded = tl.cdiv(H, 16) * 16

    lse_base = t * KV_SPLITS * H_padded
    lse_p = tl.load(lse_partial_ptr + lse_base + k_offs * H_padded + h)

    ap_base = t * KV_SPLITS * H_padded * D_V
    a_p = tl.load(
        acc_partial_ptr
        + ap_base
        + k_offs[:, None] * H_padded * D_V
        + h * D_V
        + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    lse_max = tl.max(lse_p, axis=0)
    weights = tl.exp2(lse_p - lse_max)
    w_sum = tl.sum(weights, axis=0)
    scale = tl.exp2(lse_p - lse_max - tl.log2(tl.maximum(w_sum, 1.0e-30)))
    out = tl.sum(a_p * scale[:, None], axis=0)

    if RETURN_MXFP4:
        # Packed reduction launches exactly one 32-value MXFP4 group per CTA.
        _store_mxfp4_group(
            out_ptr,
            out_scale_ptr,
            out[None, :],
            t,
            h + tl.arange(0, 1),
            h + tl.arange(0, 1) < H,
            0,
            dc,
            T,
        )
    else:
        tl.store(
            out_ptr + t * H * D_V + h * D_V + d_offs,
            out.to(tl.bfloat16),
            mask=d_mask,
        )


def _triton_sparse_mla_fwd_splitk(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    kv_splits: int,
    return_mxfp4: bool = False,
) -> SparseMLAOutput:
    """Split-K path for short sequences."""
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    d_tail = q_rope.shape[-1]
    kv_dim = kv.shape[-1]
    topk = indices.shape[-1]
    idx_flat = indices.squeeze(1).contiguous() if indices.dim() == 3 else indices
    q_nope, stride_qn_t, stride_qn_h = _row_strides(q_nope)
    q_rope, stride_qr_t, stride_qr_h = _row_strides(q_rope)

    BLOCK_H = 16
    BLOCK_K = 64
    n_head_blocks = (H + BLOCK_H - 1) // BLOCK_H
    h_padded = n_head_blocks * BLOCK_H

    num_groups = d_v // 128
    assert (
        num_groups <= 4
    ), f"Triton sparse MLA supports d_v up to 512 (4 groups), got d_v={d_v}"
    qk_scale = float(sm_scale) * _LOG2E

    max_kv_splits = topk // BLOCK_K
    kv_splits = min(kv_splits, max_kv_splits)

    out, out_scales = _allocate_output(q_nope, seq, H, d_v, return_mxfp4)

    if kv_splits == 1:
        _sparse_mla_fused_kernel[(seq, n_head_blocks)](
            q_nope,
            q_rope,
            kv,
            idx_flat,
            out,
            out_scales if out_scales is not None else out,
            qk_scale,
            _FP8_MAX,
            topk=topk,
            H=H,
            KV_DIM=kv_dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=num_groups,
            T=seq,
            RETURN_MXFP4=return_mxfp4,
            STRIDE_QN_T=stride_qn_t,
            STRIDE_QN_H=stride_qn_h,
            STRIDE_QR_T=stride_qr_t,
            STRIDE_QR_H=stride_qr_h,
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        return _format_output(out, out_scales, return_mxfp4)

    tiles_per_split = (topk + kv_splits * BLOCK_K - 1) // (kv_splits * BLOCK_K)
    active_splits = (topk + tiles_per_split * BLOCK_K - 1) // (
        tiles_per_split * BLOCK_K
    )
    active_splits = min(active_splits, kv_splits)

    lse_partial = torch.empty(
        seq, kv_splits, h_padded, dtype=torch.float32, device=q_nope.device
    )
    acc_partial = torch.empty(
        seq, kv_splits, h_padded, d_v, dtype=torch.bfloat16, device=q_nope.device
    )

    _sparse_mla_split_k_kernel[(seq, n_head_blocks, kv_splits)](
        q_nope,
        q_rope,
        kv,
        idx_flat,
        lse_partial,
        acc_partial,
        qk_scale,
        _FP8_MAX,
        topk=topk,
        H=H,
        KV_DIM=kv_dim,
        D_V=d_v,
        D_TAIL=d_tail,
        NUM_GROUPS=num_groups,
        STRIDE_QN_T=stride_qn_t,
        STRIDE_QN_H=stride_qn_h,
        STRIDE_QR_T=stride_qr_t,
        STRIDE_QR_H=stride_qr_h,
        KV_SPLITS=kv_splits,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    D_CHUNK = _MXFP4_GROUP_SIZE if return_mxfp4 else 64
    _sparse_mla_reduce_kernel[(seq, H, (d_v + D_CHUNK - 1) // D_CHUNK)](
        lse_partial,
        acc_partial,
        out,
        out_scales if out_scales is not None else out,
        H=H,
        D_V=d_v,
        KV_SPLITS=kv_splits,
        ACTIVE_SPLITS=active_splits,
        D_CHUNK=D_CHUNK,
        BLOCK_K=BLOCK_K,
        T=seq,
        RETURN_MXFP4=return_mxfp4,
        num_warps=4,
    )
    return _format_output(out, out_scales, return_mxfp4)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def triton_sparse_mla_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    return_mxfp4: bool = False,
) -> SparseMLAOutput:
    """Unified sparse MLA forward. Auto-selects single-pass vs split-K.

    q_nope: [seq, H, d_v] fp8, q_rope: [seq, H, dim-d_v] fp8,
    kv: [num_pages, 1, dim] fp8, indices: [seq, 1, topk].

    By default returns [1, seq, H, d_v] BF16 to match tilelang_sparse_fwd.
    With ``return_mxfp4=True``, returns packed E2M1 values [H, seq, d_v/2]
    uint8 and row-major E8M0 scales [H, seq, d_v/32] uint8. The packed path
    is intentionally narrow: gfx950 production geometry, d_v=512 and H>=16.
    """
    seq = q_nope.shape[0]
    H = q_nope.shape[1]
    return_mxfp4 = d_v == 512 and H >= 16
    if return_mxfp4 and (d_v != 512 or H < 16):
        return_mxfp4 = False
    num_cu = _cu_count()
    BLOCK_H = 16
    BLOCK_K = 64
    topk = indices.shape[-1]
    max_kv_splits = topk // BLOCK_K
    head_blocks = max(1, (H + BLOCK_H - 1) // BLOCK_H)
    base_ctas = seq * head_blocks
    kv_work_per_cta = topk // BLOCK_K
    if base_ctas > num_cu:
        return _triton_sparse_mla_fwd_single(
            q_nope, q_rope, kv, indices, sm_scale, d_v, return_mxfp4
        )
    kv_splits = min(
        _kv_splits_heuristic(
            seq, H, BLOCK_H, target_wg_per_cu=1.0, max_kv_splits=max_kv_splits
        ),
        max_kv_splits,
    )
    return _triton_sparse_mla_fwd_splitk(
        q_nope, q_rope, kv, indices, sm_scale, d_v, kv_splits, return_mxfp4
    )
