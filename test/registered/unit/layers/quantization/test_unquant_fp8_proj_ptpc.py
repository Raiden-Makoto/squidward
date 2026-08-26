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

from sglang.srt.distributed import parallel_state
from sglang.srt.environ import envs
from sglang.srt.layers import communicator
from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.quantization import fp8_utils, unquant
from sglang.srt.models import deepseek_v2
from sglang.test.test_utils import CustomTestCase


class TestUnquantFp8ProjPtpc(CustomTestCase):
    FUSED_QKV_A_N = 2624
    FUSED_QKV_A_K = 6144

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

    def test_fused_rmsnorm_quant_keeps_bf16_for_dsa(self):
        x = torch.empty(3, 8, dtype=torch.bfloat16)
        weight = torch.empty(8, dtype=torch.bfloat16)
        residual = torch.empty_like(x)

        with patch.object(
            communicator, "_aiter_fp8_dtype", torch.float8_e4m3fn
        ), patch.object(
            communicator,
            "_aiter_fused_rmsnorm_per_token_quant",
            create=True,
        ) as fused_norm_quant:
            without_residual = (
                communicator._fused_rmsnorm_fp8_per_token_quant_keep_bf16(
                    x, weight, 1e-6
                )
            )
            with_residual, residual_out = (
                communicator._fused_rmsnorm_fp8_per_token_quant_keep_bf16(
                    x, weight, 1e-6, residual=residual
                )
            )

        for output in (without_residual, with_residual):
            bf16_output, fp8_output, scale = output
            self.assertEqual(bf16_output.dtype, torch.bfloat16)
            self.assertEqual(fp8_output.dtype, torch.float8_e4m3fn)
            self.assertEqual(scale.dtype, torch.float32)
            self.assertEqual(tuple(bf16_output.shape), (3, 8))
            self.assertEqual(tuple(fp8_output.shape), (3, 8))
            self.assertEqual(tuple(scale.shape), (3, 1))
        self.assertEqual(tuple(residual_out.shape), (3, 8))
        self.assertEqual(fused_norm_quant.call_count, 2)

    def test_fused_allreduce_dispatches_per_token_quant(self):
        expected = (
            torch.empty(3, 8, dtype=torch.float8_e4m3fn),
            torch.empty(3, 8, dtype=torch.bfloat16),
            torch.empty(3, 1, dtype=torch.float32),
            torch.empty(3, 8, dtype=torch.bfloat16),
        )
        fused_per_token = MagicMock(return_value=expected)
        coordinator = SimpleNamespace(
            ca_comm=SimpleNamespace(
                disabled=False,
                custom_fused_ar_rms_quant=fused_per_token,
            ),
            world_size=4,
        )
        x = torch.empty(3, 8, dtype=torch.bfloat16)
        residual = torch.empty_like(x)
        weight = torch.empty(8, dtype=torch.bfloat16)

        with patch.object(parallel_state, "is_hip", return_value=True), patch.object(
            parallel_state, "is_gfx95_supported", return_value=True
        ):
            actual = (
                parallel_state.GroupCoordinator.fused_allreduce_rmsnorm_quant_per_group(
                    coordinator,
                    x,
                    residual,
                    weight,
                    1e-6,
                    group_size=0,
                    emit_bf16=True,
                )
            )

        self.assertIs(actual, expected)
        fused_per_token.assert_called_once()
        args = fused_per_token.call_args
        self.assertIs(args.args[0], x)
        self.assertIs(args.args[1], residual)
        self.assertIs(args.args[2], weight)
        self.assertEqual(args.args[3], 1e-6)
        self.assertTrue(args.kwargs["emit_bf16"])

    def test_dsa_indexer_uses_bf16_side_output(self):
        fp8_input = torch.empty(3, 8, dtype=torch.float8_e4m3fn)
        input_scale = torch.empty(3, 1, dtype=torch.float32)
        bf16_input = torch.empty(3, 8, dtype=torch.bfloat16)

        with patch.object(dsa_indexer, "_use_aiter", True), patch.object(
            dsa_indexer, "_is_gfx95_supported", True
        ):
            actual = dsa_indexer._unwrap_aiter_bf16_side_output(
                (fp8_input, input_scale, bf16_input)
            )

        self.assertIs(actual, bf16_input)

    def test_qkv_a_consumes_prequantized_norm_output(self):
        fp8_input = torch.empty(3, 8, dtype=torch.float8_e4m3fn)
        input_scale = torch.empty(3, 1, dtype=torch.float32)
        bf16_input = torch.empty(3, 8, dtype=torch.bfloat16)
        expected = torch.empty(3, 4, dtype=torch.bfloat16)
        fused_proj = MagicMock(return_value=(expected, None))
        fused_proj._fp8_proj_ready = True
        attention = SimpleNamespace(
            q_lora_rank=1,
            fused_qkv_a_proj_with_mqa=fused_proj,
            _use_min_latency_fused_a_gemm=False,
        )

        actual = deepseek_v2.DeepseekV2AttentionMLA.prepare_qkv_latent(
            attention,
            (fp8_input, input_scale, bf16_input),
            forward_batch=None,
        )

        self.assertIs(actual, expected)
        fused_proj.assert_called_once_with((fp8_input, input_scale))

    def test_ptpc_projection_selects_fused_per_token_norm_quant(self):
        fused_proj = SimpleNamespace(
            weight=torch.empty(4, 8, dtype=torch.bfloat16),
            _fp8_proj_gemm=True,
        )
        decoder_layer = SimpleNamespace(
            self_attn=SimpleNamespace(fused_qkv_a_proj_with_mqa=fused_proj)
        )

        with envs.SGLANG_DSA_FP8_PROJ_GEMM.override(True), patch.object(
            deepseek_v2, "_is_gfx95_supported", True
        ), patch.object(deepseek_v2, "_use_aiter_gfx95", True):
            quant_format = (
                deepseek_v2.DeepseekV2DecoderLayer._detect_gfx95_quant_format(
                    decoder_layer
                )
            )

        self.assertEqual(quant_format, "fp8_ptpc")

    def test_fused_qkv_a_uses_bf16_through_m512(self):
        method = unquant.UnquantizedLinearMethod()
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.empty(4, 8, dtype=torch.bfloat16, device="meta"),
                requires_grad=False,
            ),
            _fp8_proj_weight=torch.empty(4, 8, device="meta"),
            _fp8_proj_weight_scale=torch.empty(4, 1, device="meta"),
            _fp8_proj_ready=True,
            _fp8_proj_gemm_bf16_max_m=512,
        )
        bf16_output = torch.empty(512, 4, dtype=torch.bfloat16, device="meta")
        ptpc_output = torch.empty(513, 4, dtype=torch.bfloat16, device="meta")

        with patch.object(unquant, "_use_aiter", True), patch.object(
            unquant.tgemm, "mm", return_value=bf16_output
        ) as bf16_gemm, patch.object(
            fp8_utils, "apply_fp8_ptpc_linear", return_value=ptpc_output
        ) as apply_ptpc:
            x_bf16 = torch.empty(512, 8, dtype=torch.bfloat16, device="meta")
            self.assertIs(method.apply(layer, x_bf16), bf16_output)
            bf16_gemm.assert_called_once_with(
                x_bf16, layer.weight, None, otype=x_bf16.dtype
            )
            apply_ptpc.assert_not_called()

            x_ptpc = torch.empty(513, 8, dtype=torch.bfloat16, device="meta")
            self.assertIs(method.apply(layer, x_ptpc), ptpc_output)
            apply_ptpc.assert_called_once_with(
                x_ptpc,
                layer._fp8_proj_weight,
                layer._fp8_proj_weight_scale,
                bias=None,
            )

    def test_fused_qkv_a_exact_shape_repack_dispatch_and_bf16_rollback(self):
        method = unquant.UnquantizedLinearMethod()
        weight = torch.empty(
            self.FUSED_QKV_A_N,
            self.FUSED_QKV_A_K,
            dtype=torch.bfloat16,
            device="meta",
        )
        parameter = torch.nn.Parameter(weight, requires_grad=False)
        layer = SimpleNamespace(weight=parameter, _fp8_proj_gemm=True)
        fp8_weight = torch.empty(
            self.FUSED_QKV_A_N,
            self.FUSED_QKV_A_K,
            dtype=torch.uint8,
            device="meta",
        )
        weight_scale = torch.empty(
            self.FUSED_QKV_A_N, 1, dtype=torch.float32, device="meta"
        )
        shuffled = torch.empty_like(fp8_weight)

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
        self.assertEqual(
            tuple(layer.weight.shape),
            (self.FUSED_QKV_A_N, self.FUSED_QKV_A_K),
        )
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertEqual(
            tuple(layer._fp8_proj_weight.shape),
            (self.FUSED_QKV_A_N, self.FUSED_QKV_A_K),
        )
        self.assertEqual(
            tuple(layer._fp8_proj_weight_scale.shape),
            (self.FUSED_QKV_A_N, 1),
        )
        self.assertTrue(layer._fp8_proj_ready)

        x = torch.empty(2, self.FUSED_QKV_A_K, dtype=torch.bfloat16, device="meta")
        expected_ptpc = torch.empty(
            2, self.FUSED_QKV_A_N, dtype=torch.bfloat16, device="meta"
        )
        with patch.object(
            fp8_utils, "apply_fp8_ptpc_linear", return_value=expected_ptpc
        ) as apply_ptpc:
            actual_ptpc = method.apply(layer, x)

        self.assertIs(actual_ptpc, expected_ptpc)
        self.assertEqual(
            tuple(actual_ptpc.shape),
            (2, self.FUSED_QKV_A_N),
        )
        apply_ptpc.assert_called_once_with(
            x,
            layer._fp8_proj_weight,
            layer._fp8_proj_weight_scale,
            bias=None,
        )

        layer._fp8_proj_ready = False
        expected_bf16 = torch.empty_like(expected_ptpc)
        with patch.object(
            unquant, "use_intel_amx_backend", return_value=False
        ), patch.object(unquant, "_use_aiter", False), patch.object(
            unquant.F, "linear", return_value=expected_bf16
        ) as bf16_linear:
            actual_bf16 = method.apply(layer, x)

        self.assertIs(actual_bf16, expected_bf16)
        bf16_linear.assert_called_once_with(x, parameter, None)


if __name__ == "__main__":
    unittest.main(verbosity=3)
