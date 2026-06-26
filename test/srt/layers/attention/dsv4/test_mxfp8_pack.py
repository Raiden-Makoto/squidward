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
