import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models.glm5_next import Glm5NextDecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _layer(mock_packer=True):
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        mhc=True,
        hc_mult=4,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )
    layer.hc_ffn_fn = torch.nn.Parameter(
        torch.empty(24, 4 * 4096, device="meta", dtype=torch.float32)
    )
    layer.hc_ffn_scale = torch.nn.Parameter(
        torch.empty(3, device="meta", dtype=torch.float32)
    )
    layer.hc_ffn_base = torch.nn.Parameter(
        torch.empty(24, device="meta", dtype=torch.float32)
    )
    layer.register_buffer("_hc_ffn_fn_packed", None, persistent=False)
    layer._hc_ffn_fn_packed_source = None
    if mock_packer:
        layer._get_hc_ffn_fn_packed = Mock(return_value="packed_fn")
    return layer


def _inputs(m):
    return (
        torch.empty(m, 4096, device="meta", dtype=torch.bfloat16),
        torch.empty(m, 4 * 4096, device="meta", dtype=torch.bfloat16),
        torch.empty(m, 4 * 4, device="meta", dtype=torch.float32),
        torch.empty(m, 4, device="meta", dtype=torch.float32),
        torch.empty(4096, device="meta", dtype=torch.bfloat16),
    )


@patch("sglang.srt.models.glm5_next._use_aiter_gfx95", True)
@patch("sglang.srt.models.glm5_next.apply_mhc_post_pre_boundary")
def test_glm53_large_m_uses_packed_forced_fusion(mock_apply):
    layer = _layer()
    for m in (8192, 16384):
        hidden, residual, h_res, h_post, norm_weight = _inputs(m)
        mock_apply.return_value = (
            residual.view(m, 4, 4096),
            hidden,
            h_post.view(m, 4),
            h_res.view(m, 4, 4),
            True,
        )
        result = layer.hc_ffn_post_pre(
            hidden, residual, h_res, h_post, norm_weight, 1e-6
        )
        assert result is not None
        kwargs = mock_apply.call_args.kwargs
        assert kwargs["hc_fn"] == "packed_fn"
        assert kwargs["force_fused"] is True
        assert kwargs["w_preshuffle_bf16"] is True


@patch("sglang.srt.models.glm5_next._use_aiter_gfx95", True)
@patch("sglang.srt.models.glm5_next.apply_mhc_post_pre_boundary")
def test_glm53_unmeasured_large_m_falls_back(mock_apply):
    layer = _layer()
    inputs = _inputs(4096)
    assert layer.hc_ffn_post_pre(*inputs, 1e-6) is None
    mock_apply.assert_not_called()
    layer._get_hc_ffn_fn_packed.assert_not_called()


def test_glm53_packed_weight_cache_tracks_parameter_version():
    layer = _layer(mock_packer=False)
    pack = Mock(
        side_effect=lambda weight: torch.empty(
            weight.shape, device="meta", dtype=torch.int32
        )
    )
    modules = {
        "aiter": ModuleType("aiter"),
        "aiter.ops": ModuleType("aiter.ops"),
        "aiter.ops.mhc": ModuleType("aiter.ops.mhc"),
    }
    modules["aiter.ops.mhc"].mhc_shuffle_fn = pack
    with patch.dict(sys.modules, modules):
        first = layer._get_hc_ffn_fn_packed()
        second = layer._get_hc_ffn_fn_packed()
        assert first is second
        pack.assert_called_once()

        with torch.no_grad():
            layer.hc_ffn_fn.add_(1)
        third = layer._get_hc_ffn_fn_packed()
        assert third is not first
        assert pack.call_count == 2
