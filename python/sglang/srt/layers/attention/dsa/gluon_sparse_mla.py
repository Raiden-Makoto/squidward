"""Gluon sparse-MLA prefill kernel (gfx950 / CDNA4), correctness-first.

Per-query flash attention over the indexer-selected topk KV in the gluon
dialect. Uses vendored `tl_dot` helpers (from triton's triton_to_gluon_translator
amd_helpers) that build a backend-valid CDNA4 MFMA layout ([32,32,32] fp8,
transposed) and auto-convert dot operands via `amd_mfma`. This guarantees
*correctness*, not peak performance -- the async-gather pipeline / dedup is a
later perf pass. Opt-in via SGLANG_DSA_SPARSE_MLA_GLUON=1; triton is the default.
"""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import mfma as amd_mfma
from triton.language.target_info import current_target


@gluon.constexpr_function
def _num_threads_per_warp(target=None):
    if target is None:
        target = current_target()
    if target is not None and target.backend == "hip":
        gfx_major = int(target.arch[3:-2])
        return gl.constexpr(32 if gfx_major >= 10 else 64)
    return gl.constexpr(32)


@gluon.constexpr_function
def _default_blocked(shape, num_warps, target=None):
    rank = len(shape)
    spt = [1] * rank
    tpw = [1] * rank
    tpw[rank - 1] = _num_threads_per_warp(target)
    wpc = [1] * rank
    wpc[0] = num_warps
    order = list(range(rank - 1, -1, -1))
    return gl.BlockedLayout(
        size_per_thread=spt, threads_per_warp=tpw, warps_per_cta=wpc, order=order
    )


@gluon.constexpr_function
def _cdna_version(target=None):
    return 4 if target is not None and target.arch == "gfx950" else 3


@gluon.constexpr_function
def _mfma_instr_k(ebw, target=None):
    v = _cdna_version(target)
    if ebw == 8:  # fp8: CDNA4 32x32x64 / CDNA3 32x32x16
        return 64 if v == 4 else 16
    kbits = 128 if v == 3 else 256
    return kbits // ebw


@gluon.constexpr_function
def _mfma_layout(num_warps, ebw, target=None):
    ik = _mfma_instr_k(ebw, target)
    return gl.amd.AMDMFMALayout(
        version=_cdna_version(target),
        instr_shape=[32, 32, ik],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )


@gluon.constexpr_function
def _mfma_kwidth(a_ty, b_ty, target=None):
    mb = min(a_ty.element_ty.primitive_bitwidth, b_ty.element_ty.primitive_bitwidth)
    return _mfma_instr_k(mb, target) // 2


@gluon.jit
def tl_dot(a, b, acc, out_dtype: gl.constexpr = gl.float32):
    M: gl.constexpr = a.type.shape[0]
    N: gl.constexpr = b.type.shape[1]
    nw: gl.constexpr = gl.num_warps()
    mb: gl.constexpr = min(
        a.type.element_ty.primitive_bitwidth, b.type.element_ty.primitive_bitwidth
    )
    tgt: gl.constexpr = current_target()
    ml: gl.constexpr = _mfma_layout(nw, mb, tgt)
    kw: gl.constexpr = _mfma_kwidth(a.type, b.type, tgt)
    al: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=ml, k_width=kw)
    bl: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=ml, k_width=kw)
    a = gl.convert_layout(a, al)
    b = gl.convert_layout(b, bl)
    if acc is not None:
        accum = gl.convert_layout(acc, ml)
    else:
        accum = gl.zeros([M, N], out_dtype, layout=ml)
    res = amd_mfma(a, b, accum)
    if acc is not None:
        rl: gl.constexpr = acc.type.layout
    else:
        rl: gl.constexpr = _default_blocked([M, N], nw, tgt)
    return gl.convert_layout(res, rl)


@gluon.jit
def _gluon_sparse_mla_fwd_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    o_ptr,
    sm_scale,
    fp8_max,
    topk,
    H: gl.constexpr,
    D_V: gl.constexpr,
    D_TAIL: gl.constexpr,
    DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    s_i = gl.program_id(0)
    fp8_ty: gl.constexpr = q_nope_ptr.dtype.element_ty

    qm_bl: gl.constexpr = _default_blocked([H, D_V], NUM_WARPS)
    qt_bl: gl.constexpr = _default_blocked([H, D_TAIL], NUM_WARPS)
    kmT_bl: gl.constexpr = _default_blocked([D_V, BLOCK_N], NUM_WARPS)
    ktT_bl: gl.constexpr = _default_blocked([D_TAIL, BLOCK_N], NUM_WARPS)
    v_bl: gl.constexpr = _default_blocked([BLOCK_N, D_V], NUM_WARPS)
    qk_bl: gl.constexpr = _default_blocked([H, BLOCK_N], NUM_WARPS)
    acc_bl: gl.constexpr = _default_blocked([H, D_V], NUM_WARPS)

    hq = gl.arange(0, H, layout=gl.SliceLayout(1, qm_bl))
    dv_q = gl.arange(0, D_V, layout=gl.SliceLayout(0, qm_bl))
    q_main = gl.load(q_nope_ptr + s_i * H * D_V + hq[:, None] * D_V + dv_q[None, :])
    hqt = gl.arange(0, H, layout=gl.SliceLayout(1, qt_bl))
    dt_q = gl.arange(0, D_TAIL, layout=gl.SliceLayout(0, qt_bl))
    q_tail = gl.load(q_rope_ptr + s_i * H * D_TAIL + hqt[:, None] * D_TAIL + dt_q[None, :])

    m_i = gl.full([H], -float("inf"), gl.float32, layout=gl.SliceLayout(1, qk_bl))
    l_i = gl.zeros([H], gl.float32, layout=gl.SliceLayout(1, qk_bl))
    acc = gl.zeros([H, D_V], gl.float32, layout=acc_bl)

    # index aranges (one per consumer layout)
    n_km = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kmT_bl))
    dv_km = gl.arange(0, D_V, layout=gl.SliceLayout(1, kmT_bl))
    n_kt = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, ktT_bl))
    dt_kt = gl.arange(0, D_TAIL, layout=gl.SliceLayout(1, ktT_bl))
    n_v = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, v_bl))
    dv_v = gl.arange(0, D_V, layout=gl.SliceLayout(0, v_bl))
    n_qk = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, qk_bl))

    for k0 in range(0, topk, BLOCK_N):
        # gather page ids in each consumer layout
        i_km = gl.load(idx_ptr + s_i * topk + k0 + n_km, mask=(k0 + n_km) < topk, other=-1)
        p_km = gl.where(i_km >= 0, i_km, 0)
        kv_main_t = gl.load(
            kv_ptr + p_km[None, :] * DIM + dv_km[:, None],
            mask=(i_km >= 0)[None, :], other=0.0,
        )  # [D_V, BLOCK_N]
        i_kt = gl.load(idx_ptr + s_i * topk + k0 + n_kt, mask=(k0 + n_kt) < topk, other=-1)
        p_kt = gl.where(i_kt >= 0, i_kt, 0)
        kv_tail_t = gl.load(
            kv_ptr + p_kt[None, :] * DIM + (D_V + dt_kt)[:, None],
            mask=(i_kt >= 0)[None, :], other=0.0,
        )  # [D_TAIL, BLOCK_N]

        qk = tl_dot(q_main, kv_main_t, None)
        qk = tl_dot(q_tail, kv_tail_t, qk)
        qk = qk * sm_scale
        i_qk = gl.load(idx_ptr + s_i * topk + k0 + n_qk, mask=(k0 + n_qk) < topk, other=-1)
        qk = gl.where((i_qk >= 0)[None, :], qk, -float("inf"))

        m_new = gl.maximum(m_i, gl.max(qk, axis=1))
        m_safe = gl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = gl.exp(m_i - m_safe)
        p = gl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + gl.sum(p, axis=1)

        i_v = gl.load(idx_ptr + s_i * topk + k0 + n_v, mask=(k0 + n_v) < topk, other=-1)
        p_v = gl.where(i_v >= 0, i_v, 0)
        v = gl.load(
            kv_ptr + p_v[:, None] * DIM + dv_v[None, :],
            mask=(i_v >= 0)[:, None], other=0.0,
        )  # [BLOCK_N, D_V]
        p_fp8 = (p * fp8_max).to(fp8_ty)
        pv = tl_dot(p_fp8, v, None) * (1.0 / fp8_max)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    l_safe = gl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    h_o = gl.arange(0, H, layout=gl.SliceLayout(1, acc_bl))
    dv_o = gl.arange(0, D_V, layout=gl.SliceLayout(0, acc_bl))
    gl.store(
        o_ptr + s_i * H * D_V + h_o[:, None] * D_V + dv_o[None, :],
        acc.to(o_ptr.dtype.element_ty),
    )


_FNUZ = False
try:
    from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

    _FNUZ = is_fp8_fnuz()
except Exception:
    pass
_FP8_MAX = 240.0 if _FNUZ else 448.0


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
        q_nope, q_rope, kv, indices, out, sm_scale, _FP8_MAX, topk,
        H=H, D_V=d_v, D_TAIL=d_tail, DIM=dim,
        BLOCK_N=64, NUM_WARPS=4,
        num_warps=4,
    )
    return out.unsqueeze(0)
