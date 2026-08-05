"""CPU tests for the opt-in GLM-5.2 PTPC FP8 projection path."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_utils, unquant
from sglang.test.test_utils import CustomTestCase


class TestUnquantFp8ProjPtpc(CustomTestCase):
    def test_gate_truth_table(self):
        layer = SimpleNamespace()
        for enabled in (False, True):
            for use_aiter in (False, True):
                for marked in (False, True):
                    for gfx950 in (False, True):
                        with self.subTest(
                            enabled=enabled,
                            use_aiter=use_aiter,
                            marked=marked,
                            gfx950=gfx950,
                        ), envs.SGLANG_DSA_FP8_PROJ_GEMM.override(
                            enabled
                        ), patch.object(
                            unquant, "_use_aiter", use_aiter
                        ), patch.object(
                            unquant,
                            "is_gfx95_supported",
                            return_value=gfx950,
                        ) as gfx95_supported:
                            if marked:
                                layer._fp8_proj_gemm = True
                            elif hasattr(layer, "_fp8_proj_gemm"):
                                del layer._fp8_proj_gemm
                            self.assertEqual(
                                unquant._fp8_proj_gemm_enabled(layer),
                                enabled and use_aiter and marked and gfx950,
                            )
                            self.assertEqual(
                                gfx95_supported.call_count,
                                int(enabled and use_aiter and marked),
                            )

    def test_default_off_preserves_bf16_linear(self):
        self.assertFalse(envs.SGLANG_DSA_FP8_PROJ_GEMM.default)
        method = unquant.UnquantizedLinearMethod()
        weight = torch.randn(4, 8, dtype=torch.bfloat16)
        layer = SimpleNamespace(weight=torch.nn.Parameter(weight, requires_grad=False))
        x = torch.randn(3, 8, dtype=torch.bfloat16)

        with envs.SGLANG_DSA_FP8_PROJ_GEMM.override(False), patch.object(
            unquant, "_use_aiter", True
        ), patch.object(unquant, "is_gfx95_supported", return_value=True), patch.object(
            method, "_repack_bf16_to_fp8_ptpc"
        ) as repack, patch.object(
            unquant, "_is_cpu_amx_available", False
        ):
            method.process_weights_after_loading(layer)

        repack.assert_not_called()
        with patch.object(
            unquant, "use_intel_amx_backend", return_value=False
        ), patch.object(unquant, "_use_aiter", False):
            actual = method.apply(layer, x)
        torch.testing.assert_close(actual, F.linear(x, weight), rtol=0, atol=0)
        self.assertFalse(hasattr(layer, "_fp8_proj_ready"))

    def test_non_rocm_gate_does_not_import_aiter(self):
        method = unquant.UnquantizedLinearMethod()
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.zeros(4, 8, dtype=torch.bfloat16), requires_grad=False
            ),
            _fp8_proj_gemm=True,
        )
        with envs.SGLANG_DSA_FP8_PROJ_GEMM.override(True), patch.object(
            unquant, "_use_aiter", True
        ), patch.object(
            unquant, "is_gfx95_supported", return_value=False
        ), patch.object(
            method, "_repack_bf16_to_fp8_ptpc"
        ) as repack, patch.object(
            unquant, "_is_cpu_amx_available", False
        ):
            method.process_weights_after_loading(layer)
        repack.assert_not_called()

    def test_ptpc_repack_preserves_bf16_and_shuffles(self):
        method = unquant.UnquantizedLinearMethod()
        weight = torch.arange(32, dtype=torch.bfloat16).view(4, 8)
        parameter = torch.nn.Parameter(weight.clone(), requires_grad=False)
        layer = SimpleNamespace(weight=parameter)
        fp8_weight = torch.arange(32, dtype=torch.float32).view(4, 8)
        weight_scale = torch.arange(4, dtype=torch.float32).view(4, 1)
        shuffled = fp8_weight + 1

        aiter = types.ModuleType("aiter")
        aiter.dtypes = SimpleNamespace(fp8=object())
        aiter.pertoken_quant = MagicMock(return_value=(fp8_weight, weight_scale))
        aiter_ops = types.ModuleType("aiter.ops")
        aiter_shuffle = types.ModuleType("aiter.ops.shuffle")
        aiter_shuffle.shuffle_weight = MagicMock(return_value=shuffled)

        with patch.dict(
            sys.modules,
            {
                "aiter": aiter,
                "aiter.ops": aiter_ops,
                "aiter.ops.shuffle": aiter_shuffle,
            },
        ):
            method._repack_bf16_to_fp8_ptpc(layer)

        self.assertIs(layer.weight, parameter)
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(layer._fp8_proj_weight, shuffled)
        torch.testing.assert_close(layer._fp8_proj_weight_scale, weight_scale)
        self.assertTrue(layer._fp8_proj_ready)
        aiter.pertoken_quant.assert_called_once()
        self.assertEqual(
            aiter_shuffle.shuffle_weight.call_args.args[1],
            (16, 16),
        )

    def test_apply_dispatches_to_ptpc_consumer(self):
        method = unquant.UnquantizedLinearMethod()
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.zeros(4, 8, dtype=torch.bfloat16), requires_grad=False
            ),
            _fp8_proj_weight=torch.zeros(4, 8),
            _fp8_proj_weight_scale=torch.ones(4, 1),
            _fp8_proj_ready=True,
        )
        fp8_input = torch.zeros(3, 8)
        input_scale = torch.ones(3, 1)
        bias = torch.zeros(4)
        expected = torch.ones(3, 4)

        with patch.object(
            fp8_utils, "apply_fp8_ptpc_linear", return_value=expected
        ) as apply_ptpc:
            actual = method.apply(layer, (fp8_input, input_scale), bias)

        self.assertIs(actual, expected)
        apply_ptpc.assert_called_once()
        args = apply_ptpc.call_args
        self.assertIs(args.args[0][0], fp8_input)
        self.assertIs(args.args[0][1], input_scale)
        self.assertIs(args.args[1], layer._fp8_proj_weight)
        self.assertIs(args.args[2], layer._fp8_proj_weight_scale)
        self.assertIs(args.kwargs["bias"], bias)


if __name__ == "__main__":
    unittest.main(verbosity=3)
