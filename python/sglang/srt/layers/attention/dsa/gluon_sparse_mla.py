"""Gluon sparse-MLA prefill kernel (gfx950), iteration 1.

Per-query flash attention over the indexer-selected topk KV, using the CDNA4
gluon dialect so the scattered topk KV gather can be issued as explicit
`buffer_load` and overlapped with the MFMA (the plain-Triton kernel is
memory-bound and does not hide the gather latency). Opt-in via
SGLANG_DSA_SPARSE_MLA_GLUON=1; the triton kernel remains the default fallback.

NOTE: layout-heavy WIP validated against `utilities/bench_sparse_mla.py --check`.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _gluon_sparse_mla_fwd_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    o_ptr,
    sm_scale,
    topk,
    H: gl.constexpr,
    D_V: gl.constexpr,
    D_TAIL: gl.constexpr,
    DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    s_i = gl.program_id(0)

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=False,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    # q: [H, DIM] with main (D_V) followed by tail (D_TAIL), loaded from the two
    # source tensors into one [H, DIM] tile so qk is a single MFMA over K=DIM.
    h_a = gl.arange(0, H, layout=gl.SliceLayout(1, dot_a_layout))
    dv_a = gl.arange(0, D_V, layout=gl.SliceLayout(0, dot_a_layout))
    dt_a = gl.arange(0, D_TAIL, layout=gl.SliceLayout(0, dot_a_layout))
    q_main = gl.load(
        q_nope_ptr + s_i * H * D_V + h_a[:, None] * D_V + dv_a[None, :]
    )
    q_tail = gl.load(
        q_rope_ptr + s_i * H * D_TAIL + h_a[:, None] * D_TAIL + dt_a[None, :]
    )

    m_i = gl.full([H], -float("inf"), gl.float32, layout=gl.SliceLayout(1, mfma_layout))
    l_i = gl.zeros([H], gl.float32, layout=gl.SliceLayout(1, mfma_layout))
    acc = gl.zeros([H, D_V], gl.float32, layout=mfma_layout)

    dv_b = gl.arange(0, D_V, layout=gl.SliceLayout(1, dot_b_layout))
    dt_b = gl.arange(0, D_TAIL, layout=gl.SliceLayout(1, dot_b_layout))
    n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))
    # BLOCK_N mask in the mfma column layout for qk masking / softmax
    n_m = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    for k0 in range(0, topk, BLOCK_N):
        idx = gl.load(idx_ptr + s_i * topk + k0 + n, mask=(k0 + n) < topk, other=-1)
        valid = idx >= 0
        page = gl.where(valid, idx, 0)
        idx_m = gl.load(
            idx_ptr + s_i * topk + k0 + n_m, mask=(k0 + n_m) < topk, other=-1
        )
        valid_m = idx_m >= 0
        # gather KV transposed to [D, BLOCK_N] for MFMA operand-b (K-major)
        kbase = page[None, :] * DIM
        kv_main = gl.load(
            kv_ptr + kbase + dv_b[:, None], mask=valid[None, :], other=0.0
        )  # [D_V, BLOCK_N]
        kv_tail = gl.load(
            kv_ptr + kbase + (D_V + dt_b)[:, None], mask=valid[None, :], other=0.0
        )  # [D_TAIL, BLOCK_N]

        acc_qk = gl.zeros([H, BLOCK_N], gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna4.mfma_scaled(
            a=q_main, a_scale=None, a_format="e4m3",
            b=kv_main, b_scale=None, b_format="e4m3", acc=acc_qk,
        )
        qk = gl.amd.cdna4.mfma_scaled(
            a=q_tail, a_scale=None, a_format="e4m3",
            b=kv_tail, b_scale=None, b_format="e4m3", acc=qk,
        )
        qk = qk * sm_scale
        qk = gl.where(valid_m[None, :], qk, -float("inf"))

        m_new = gl.maximum(m_i, gl.max(qk, axis=1))
        m_safe = gl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = gl.exp(m_i - m_safe)
        p = gl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + gl.sum(p, axis=1)

        # pv: p [H, BLOCK_N] @ v [BLOCK_N, D_V]; v = kv_main^T
        p_a = gl.convert_layout(p.to(q_nope_ptr.dtype.element_ty), dot_a_layout)
        v = gl.convert_layout(kv_main, dot_b_layout)  # WIP: needs [BLOCK_N, D_V]
        pv = gl.amd.cdna4.mfma_scaled(
            a=p_a, a_scale=None, a_format="e4m3",
            b=v, b_scale=None, b_format="e4m3", acc=gl.zeros([H, D_V], gl.float32, layout=mfma_layout),
        )
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    l_safe = gl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    dv_o = gl.arange(0, D_V, layout=gl.SliceLayout(0, mfma_layout))
    h_o = gl.arange(0, H, layout=gl.SliceLayout(1, mfma_layout))
    gl.store(
        o_ptr + s_i * H * D_V + h_o[:, None] * D_V + dv_o[None, :],
        acc.to(o_ptr.dtype.element_ty),
    )


def gluon_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v=512):
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    d_tail = q_rope.shape[-1]
    dim = kv.shape[-1]
    topk = indices.shape[-1]
    q_nope = q_nope.contiguous()
    q_rope = q_rope.contiguous()
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    _gluon_sparse_mla_fwd_kernel[(seq,)](
        q_nope, q_rope, kv, indices, out, sm_scale, topk,
        H=H, D_V=d_v, D_TAIL=d_tail, DIM=dim,
        BLOCK_N=64, NUM_WARPS=4,
        num_warps=4,
    )
    return out.unsqueeze(0)
