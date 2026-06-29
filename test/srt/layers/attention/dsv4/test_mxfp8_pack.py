# SPDX-License-Identifier: MIT
"""Bit-parity test for the MXFP8 NoPE packer vs aiter's reference _quantize_nope.

gfx950-only (e4m3fn + the upstream pa_sparse_prefill_fp8_opus format). Skipped
elsewhere. Run on the gbt box:
    python -m pytest test/srt/layers/attention/dsv4/test_mxfp8_pack.py -q
"""

import pytest
import torch

from sglang.srt.layers.attention.dsv4.unified_kv_kernels import mxfp8


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    return "gfx95" in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(
    not _is_gfx950(), reason="MXFP8 e4m3fn packing is gfx950-only"
)


def _ref_quantize_nope(real: torch.Tensor):
    """Verbatim transcription of aiter op_tests/test_pa_sparse_prefill_opus.py::
    _quantize_nope (commit bd003d9d)."""
    r = real.shape[0]
    blk = real.reshape(r, 14, 32).to(torch.float32)
    amax = blk.abs().amax(dim=-1)
    e = torch.ceil(torch.log2(amax.clamp(min=1e-30) / 448.0)).to(torch.int32)
    e = torch.where(amax == 0, torch.zeros_like(e), e)
    e_byte = (e + 127).clamp(0, 255).to(torch.uint8)
    s = torch.exp2(e.to(torch.float32)).unsqueeze(-1)
    q = (blk / s).to(torch.float8_e4m3fn)
    deq = (q.to(torch.float32) * s).reshape(r, 448)
    packed = torch.zeros(r, 512, dtype=torch.uint8, device=real.device)
    packed[:, :448] = q.reshape(r, 448).view(torch.uint8)
    packed[:, 448 : 448 + 14] = e_byte
    return packed.view(torch.float8_e4m3fn), deq


@pytest.mark.parametrize("rows,scale", [(1, 1.0), (64, 1.0), (257, 8.0), (1024, 0.01), (33, 1e3)])
def test_pack_bit_parity(rows, scale):
    torch.manual_seed(0)
    x = torch.randn(rows, 448, device="cuda") * scale
    if rows == 257:
        x[0, :32] = 0.0  # zero-block edge case
    packed, _ = _ref_quantize_nope(x)
    ours = mxfp8.pack_nope_mxfp8(x)
    assert torch.equal(ours.view(torch.uint8), packed.view(torch.uint8))


@pytest.mark.parametrize("rows", [1, 64, 257])
def test_dequant_roundtrip(rows):
    torch.manual_seed(1)
    x = torch.randn(rows, 448, device="cuda")
    _, ref_deq = _ref_quantize_nope(x)
    packed = mxfp8.pack_nope_mxfp8(x)
    deq = mxfp8.dequant_nope_mxfp8(packed)
    assert torch.equal(deq.to(torch.float32), ref_deq)


@pytest.mark.parametrize(
    "rows,scale", [(1, 1.0), (64, 1.0), (257, 8.0), (1024, 0.01), (33, 1e3)]
)
def test_dense_pack_bit_parity(rows, scale):
    """The fused Triton dense packer (``runtime.pack_mxfp8_dense``) must produce
    bytes identical to the eager/aiter reference for the NoPE stream and a plain
    bf16 cast for the RoPE stream."""
    from sglang.srt.layers.attention.dsv4.unified_kv_kernels import runtime

    torch.manual_seed(2)
    x = torch.randn(rows, mxfp8.DIM_HEAD, device="cuda") * scale
    if rows == 257:
        x[0, :32] = 0.0  # zero-block edge case
    ref_nope, _ = _ref_quantize_nope(x[:, : mxfp8.DIM_NOPE].contiguous())
    nope, rope = runtime.pack_mxfp8_dense(x)
    assert torch.equal(nope.view(torch.uint8), ref_nope.view(torch.uint8))
    assert torch.equal(rope, x[:, mxfp8.DIM_NOPE :].to(torch.bfloat16))


@pytest.mark.parametrize(
    "tokens,heads,scale", [(1, 16, 1.0), (64, 16, 1.0), (257, 16, 8.0), (33, 128, 0.01)]
)
def test_fused_q_norm_rope_pack_parity(tokens, heads, scale):
    """The fused prefill-q producer (``runtime.fused_q_norm_rope_mxfp8_pack``)
    must be byte-identical to the JIT ``fused_q_norm_rope`` (bf16 q_out) followed
    by the standalone ``pack_mxfp8_dense(q_out)`` it replaces."""
    from sglang.jit_kernel.dsv4.elementwise import fused_q_norm_rope
    from sglang.srt.layers.attention.dsv4.unified_kv_kernels import runtime

    torch.manual_seed(4)
    eps = 1e-6
    max_pos = 4096
    rope_pairs = mxfp8.DIM_ROPE // 2  # 32
    q_in = (
        torch.randn(tokens, heads, mxfp8.DIM_HEAD, device="cuda", dtype=torch.bfloat16)
        * scale
    ).contiguous()
    positions = torch.randint(0, max_pos, (tokens,), device="cuda", dtype=torch.int64)
    freqs_cis = torch.randn(
        max_pos, rope_pairs, 2, device="cuda", dtype=torch.float32
    )
    freqs_cis = torch.view_as_complex(freqs_cis)

    # Reference: JIT norm+rope -> bf16 q_out, then standalone MXFP8 dense pack.
    q_out = torch.empty_like(q_in)
    fused_q_norm_rope(q_in, q_out, eps, freqs_cis, positions)
    ref_nope, ref_rope = runtime.pack_mxfp8_dense(q_out)

    # Ours: single fused launch over the raw (pre norm+rope) q.
    nope, rope = runtime.fused_q_norm_rope_mxfp8_pack(q_in, eps, freqs_cis, positions)

    # NoPE (the fp8 stream this fusion targets) is byte-exact.
    assert torch.equal(nope.view(torch.uint8), ref_nope.view(torch.uint8))
    # RoPE is bf16 and can differ by <=1 ULP vs the JIT kernel: the complex
    # rotation ``xr*fr - xi*fi`` rounds differently under the HIP intrinsic's
    # fma contraction vs Triton's separate mul/sub. This is codegen rounding
    # noise (~2^-8 relative), well below attention's own bf16 noise; assert a
    # tight 1-ULP bound rather than exact bytes.
    torch.testing.assert_close(
        rope.float(), ref_rope.float(), rtol=2**-7, atol=2**-7
    )


@pytest.mark.parametrize("rows", [1, 64, 257])
def test_copy_scatter_matches_quant_scatter(rows):
    """The copy-by-loc store (reusing pre-packed bytes) must yield pool bytes
    identical to the quant-on-store scatter for the same ``loc`` set (incl. the
    ``loc < 0`` skip sentinel)."""
    from sglang.srt.layers.attention.dsv4.unified_kv_kernels import runtime

    torch.manual_seed(3)
    x = torch.randn(rows, mxfp8.DIM_HEAD, device="cuda")
    pages = rows + 5
    loc = torch.arange(rows, device="cuda", dtype=torch.int64)
    loc[::3] = -1  # exercise the skip sentinel

    def _empty_pool():
        nope = torch.zeros(
            pages, mxfp8.NOPE_PACKED_WIDTH, dtype=torch.uint8, device="cuda"
        ).view(mxfp8.FP8_DTYPE)
        rope = torch.zeros(pages, mxfp8.DIM_ROPE, dtype=torch.bfloat16, device="cuda")
        return nope, rope

    q_nope, q_rope = _empty_pool()
    runtime._launch_mxfp8_pack(
        kv=x, loc=loc, unified_kv_nope=q_nope, unified_kv_rope=q_rope
    )

    packed_nope, packed_rope = runtime.pack_mxfp8_dense(x)
    c_nope, c_rope = _empty_pool()
    runtime._launch_mxfp8_copy_scatter(
        nope_src=packed_nope,
        rope_src=packed_rope,
        loc=loc,
        unified_kv_nope=c_nope,
        unified_kv_rope=c_rope,
    )
    assert torch.equal(c_nope.view(torch.uint8), q_nope.view(torch.uint8))
    assert torch.equal(c_rope, q_rope)
