"""Unit tests for GLM packed-MXFP4 MLA absorbed-BMM selection."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common import deepseek_weight_loader as weight_loader
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _make_loader():
    self_attn = SimpleNamespace(
        kv_b_proj=SimpleNamespace(weight=torch.randn(12, 8, dtype=torch.bfloat16)),
        qk_nope_head_dim=4,
        v_head_dim=2,
        w_kc=None,
        w_vc=None,
        w_scale=None,
        w_scale_k=None,
        w_scale_v=None,
    )
    loader = object.__new__(weight_loader.DeepseekV2WeightLoaderMixin)
    loader.model = SimpleNamespace(
        start_layer=0,
        end_layer=1,
        layers=[SimpleNamespace(self_attn=self_attn)],
    )
    loader.config = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"], num_hidden_layers=1
    )
    loader.quant_config = None
    return loader, self_attn


class TestGlmMxfp4AbsorbedWeightSelection(CustomTestCase):
    def test_toggle_defaults_off(self):
        self.assertFalse(envs.SGLANG_USE_MXFP4_MLA_BMM.default)
        self.assertEqual(
            envs.SGLANG_EXPERIMENTAL_MXFP4_MLA_BMM_IMPL.default, "a16"
        )
        self.assertEqual(
            envs.SGLANG_EXPERIMENTAL_MXFP4_MLA_K_BMM_CONFIG.default, ()
        )
        self.assertEqual(
            envs.SGLANG_EXPERIMENTAL_MXFP4_MLA_V_BMM_CONFIG.default, ()
        )

    def test_flag_off_preserves_fp8_rollback(self):
        loader, self_attn = _make_loader()
        fp8_weight = torch.empty_like(
            self_attn.kv_b_proj.weight, dtype=torch.float8_e4m3fn
        )
        fp8_scale = torch.tensor(0.25)
        with (
            envs.SGLANG_USE_MXFP4_MLA_BMM.override(False),
            mock.patch.object(weight_loader, "_use_aiter_gfx95", True),
            mock.patch(
                "sglang.srt.layers.quantization.fp8_utils.input_to_float8",
                return_value=(fp8_weight, fp8_scale),
            ) as input_to_float8,
            mock.patch.object(
                weight_loader, "quark_post_load_weights", create=True
            ) as quark_post_load_weights,
        ):
            loader.post_load_weights(
                weight_names=["model.layers.0.self_attn.kv_b_proj"]
            )
        input_to_float8.assert_called_once_with(
            self_attn.kv_b_proj.weight, dtype=torch.float8_e4m3fn
        )
        quark_post_load_weights.assert_not_called()
        self.assertEqual(self_attn.w_kc.dtype, torch.float8_e4m3fn)
        self.assertEqual(self_attn.w_vc.dtype, torch.float8_e4m3fn)
        self.assertIs(self_attn.w_scale, fp8_scale)
        self.assertEqual(self_attn.w_kc.stride(), (32, 1, 4))
        self.assertEqual(self_attn.w_vc.stride(), (16, 1, 8))

    def test_flag_on_assigns_packed_weights_and_scales(self):
        loader, self_attn = _make_loader()
        w_kc = torch.arange(32, dtype=torch.uint8).view(2, 2, 8)
        w_scale_k = torch.arange(16, dtype=torch.uint8).view(2, 1, 8)
        w_vc = torch.arange(16, dtype=torch.uint8).view(2, 2, 4)
        w_scale_v = torch.arange(4, dtype=torch.uint8).view(2, 2, 1)
        with (
            envs.SGLANG_USE_MXFP4_MLA_BMM.override(True),
            mock.patch.object(weight_loader, "_use_aiter_gfx95", True),
            mock.patch.object(
                weight_loader,
                "quark_post_load_weights",
                create=True,
                return_value=(w_kc, w_scale_k, w_vc, w_scale_v),
            ) as quark_post_load_weights,
            mock.patch(
                "sglang.srt.layers.quantization.fp8_utils.input_to_float8"
            ) as input_to_float8,
        ):
            loader.post_load_weights(
                weight_names=["model.layers.0.self_attn.kv_b_proj"]
            )
        quark_post_load_weights.assert_called_once_with(
            self_attn, self_attn.kv_b_proj.weight, "mxfp4"
        )
        input_to_float8.assert_not_called()
        self.assertTrue(torch.equal(self_attn.w_kc, w_kc))
        self.assertEqual(self_attn.w_kc.stride(), (16, 1, 2))
        self.assertTrue(torch.equal(self_attn.w_vc, w_vc.transpose(1, 2)))
        self.assertEqual(self_attn.w_vc.stride(), (8, 1, 4))
        self.assertIs(self_attn.w_scale_k, w_scale_k)
        self.assertIs(self_attn.w_scale_v, w_scale_v)
        torch.testing.assert_close(
            self_attn.w_kc.transpose(-2, -1), w_kc.transpose(-2, -1)
        )
        torch.testing.assert_close(
            self_attn.w_scale_k.transpose(-2, -1), w_scale_k.transpose(-2, -1)
        )
        torch.testing.assert_close(self_attn.w_vc.transpose(-2, -1), w_vc)
        torch.testing.assert_close(
            self_attn.w_scale_v.transpose(-2, -1), w_scale_v.transpose(-2, -1)
        )


class TestMxfp4KDispatch(CustomTestCase):
    def test_uses_prequant_packed_activation_kernel(self):
        x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        weight = torch.zeros(2, 4, 4, dtype=torch.uint8)
        scale = torch.zeros(2, 4, 1, dtype=torch.uint8)
        output = torch.empty(2, 3, 4, dtype=torch.bfloat16)

        with (
            envs.SGLANG_USE_MXFP4_MLA_BMM.override(False),
            mock.patch.object(
                forward_mla, "batched_gemm_afp4wfp4_pre_quant", create=True
            ) as prequant_bmm,
        ):
            result = forward_mla._run_mxfp4_k_bmm(x, weight, scale, output)

        prequant_bmm.assert_called_once_with(x, weight, scale, torch.bfloat16, output)
        self.assertIsNone(result)

    def test_feature_uses_safe_k_block_without_split_k(self):
        x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        weight = torch.zeros(2, 4, 4, dtype=torch.uint8)
        scale = torch.zeros(2, 4, 1, dtype=torch.uint8)
        output = torch.empty(2, 3, 4, dtype=torch.bfloat16)
        tuned_config = {"BLOCK_SIZE_K": 256, "NUM_KSPLIT": 4}

        with (
            envs.SGLANG_USE_MXFP4_MLA_BMM.override(True),
            mock.patch.object(
                forward_mla,
                "_get_mxfp4_bmm_config",
                create=True,
                return_value=(tuned_config, None),
            ),
            mock.patch.object(
                forward_mla, "batched_gemm_a16wfp4", create=True
            ) as atom_bmm,
            mock.patch.object(
                forward_mla, "batched_gemm_afp4wfp4_pre_quant", create=True
            ) as prequant_bmm,
        ):
            result = forward_mla._run_mxfp4_k_bmm(x, weight, scale, output)

        args, kwargs = atom_bmm.call_args
        self.assertIs(args[0], x)
        self.assertIs(args[1], weight)
        self.assertIs(args[2], scale)
        self.assertIs(kwargs["y"], output)
        self.assertEqual(kwargs["config"]["NUM_KSPLIT"], 1)
        self.assertEqual(kwargs["config"]["BLOCK_SIZE_K"], 64)
        self.assertEqual(tuned_config["NUM_KSPLIT"], 4)
        self.assertEqual(tuned_config["BLOCK_SIZE_K"], 256)
        self.assertFalse(kwargs["transpose_bm"])
        prequant_bmm.assert_not_called()
        self.assertIsNone(result)

    def test_feature_output_matches_bf16_reference(self):
        x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        weight = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        scale = torch.zeros(2, 4, 1, dtype=torch.uint8)
        output = torch.empty(2, 3, 4, dtype=torch.bfloat16)
        expected = torch.bmm(x, weight.transpose(-2, -1))

        def reference_bmm(x, weight, _scale, *, y, transpose_bm, **_kwargs):
            result = torch.bmm(x, weight.transpose(-2, -1))
            if transpose_bm:
                result = result.transpose(0, 1)
            y.copy_(result)
            return y

        with (
            envs.SGLANG_USE_MXFP4_MLA_BMM.override(True),
            mock.patch.object(
                forward_mla,
                "_get_mxfp4_bmm_config",
                create=True,
                return_value=({"NUM_KSPLIT": 4}, None),
            ),
            mock.patch.object(
                forward_mla,
                "batched_gemm_a16wfp4",
                create=True,
                side_effect=reference_bmm,
            ),
        ):
            result = forward_mla._run_mxfp4_k_bmm(x, weight, scale, output)

        self.assertIsNone(result)
        torch.testing.assert_close(output, expected)


class TestMxfp4VDispatch(CustomTestCase):
    def _inputs(self):
        return (
            torch.randn(2, 3, 8, dtype=torch.bfloat16),
            torch.zeros(2, 4, 4, dtype=torch.uint8),
            torch.zeros(2, 4, 1, dtype=torch.uint8),
            torch.empty(3, 2, 4, dtype=torch.bfloat16),
        )

    def test_flag_off_uses_existing_prequant_kernel(self):
        x, weight, scale, output = self._inputs()
        with (
            envs.SGLANG_USE_MXFP4_MLA_BMM.override(False),
            mock.patch.object(
                forward_mla, "batched_gemm_afp4wfp4_pre_quant", create=True
            ) as prequant_bmm,
            mock.patch.object(
                forward_mla, "batched_gemm_a16wfp4", create=True
            ) as atom_bmm,
        ):
            result = forward_mla._run_mxfp4_v_bmm(x, weight, scale, output)
        args, kwargs = prequant_bmm.call_args
        self.assertEqual(kwargs, {})
        self.assertIs(args[0], x)
        self.assertIs(args[1], weight)
        self.assertIs(args[2], scale)
        self.assertIs(args[3], torch.bfloat16)
        self.assertEqual(args[4].shape, (2, 3, 4))
        self.assertEqual(args[4].stride(), output.transpose(0, 1).stride())
        self.assertEqual(args[4].data_ptr(), output.data_ptr())
        atom_bmm.assert_not_called()
        self.assertIs(result, output)

    def test_flag_on_uses_atom_batch_major_dispatch(self):
        x, weight, scale, output = self._inputs()
        tuned_config = {"BLOCK_SIZE_K": 256, "NUM_KSPLIT": 4}
        with (
            envs.SGLANG_USE_MXFP4_MLA_BMM.override(True),
            mock.patch.object(
                forward_mla,
                "_get_mxfp4_bmm_config",
                create=True,
                return_value=(tuned_config, None),
            ),
            mock.patch.object(
                forward_mla,
                "batched_gemm_a16wfp4",
                create=True,
                return_value=output,
            ) as atom_bmm,
            mock.patch.object(
                forward_mla, "batched_gemm_afp4wfp4_pre_quant", create=True
            ) as prequant_bmm,
        ):
            result = forward_mla._run_mxfp4_v_bmm(x, weight, scale, output)
        args, kwargs = atom_bmm.call_args
        self.assertIs(args[0], x)
        self.assertIs(args[1], weight)
        self.assertIs(args[2], scale)
        self.assertIs(kwargs["y"], output)
        self.assertTrue(kwargs["transpose_bm"])
        self.assertTrue(kwargs["prequant"])
        self.assertIsNone(kwargs["y_scale"])
        self.assertIs(kwargs["dtype"], torch.bfloat16)
        self.assertEqual(kwargs["config"]["NUM_KSPLIT"], 1)
        self.assertEqual(kwargs["config"]["BLOCK_SIZE_K"], 256)
        self.assertEqual(tuned_config["NUM_KSPLIT"], 4)
        self.assertEqual(tuned_config["BLOCK_SIZE_K"], 256)
        prequant_bmm.assert_not_called()
        self.assertIs(result, output)
        self.assertTrue(result.is_contiguous())

    def test_feature_batch_major_output_matches_bf16_reference(self):
        x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        weight = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        scale = torch.zeros(2, 4, 1, dtype=torch.uint8)
        output = torch.empty(3, 2, 4, dtype=torch.bfloat16)
        expected = torch.bmm(x, weight.transpose(-2, -1)).transpose(0, 1)

        def reference_bmm(x, weight, _scale, *, y, transpose_bm, **_kwargs):
            result = torch.bmm(x, weight.transpose(-2, -1))
            if transpose_bm:
                result = result.transpose(0, 1)
            y.copy_(result)
            return y

        with (
            envs.SGLANG_USE_MXFP4_MLA_BMM.override(True),
            mock.patch.object(
                forward_mla,
                "_get_mxfp4_bmm_config",
                create=True,
                return_value=({"NUM_KSPLIT": 4}, None),
            ),
            mock.patch.object(
                forward_mla,
                "batched_gemm_a16wfp4",
                create=True,
                side_effect=reference_bmm,
            ),
        ):
            result = forward_mla._run_mxfp4_v_bmm(x, weight, scale, output)

        self.assertIs(result, output)
        torch.testing.assert_close(result, expected)


class TestMxfp4ExperimentalControls(CustomTestCase):
    def _config_inputs(self, *, k=192, n=512):
        return (
            torch.empty(16, 3, k, dtype=torch.bfloat16),
            torch.empty(16, n, k // 2, dtype=torch.uint8),
        )

    def test_default_a16_k_control_is_exact(self):
        x, weight = self._config_inputs()
        tuned_config = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 256,
            "NUM_KSPLIT": 4,
        }
        with (
            envs.SGLANG_EXPERIMENTAL_MXFP4_MLA_K_BMM_CONFIG.override(""),
            mock.patch.object(
                forward_mla,
                "_get_mxfp4_bmm_config",
                create=True,
                return_value=(tuned_config, None),
            ),
        ):
            config = forward_mla._get_glm_mxfp4_k_bmm_config(x, weight)

        self.assertEqual(config["BLOCK_SIZE_M"], 128)
        self.assertEqual(config["BLOCK_SIZE_N"], 256)
        self.assertEqual(config["BLOCK_SIZE_K"], 64)
        self.assertEqual(config["NUM_KSPLIT"], 1)
        self.assertEqual(tuned_config["BLOCK_SIZE_K"], 256)
        self.assertEqual(tuned_config["NUM_KSPLIT"], 4)

    def test_default_a16_v_control_preserves_tuned_tiles(self):
        x, weight = self._config_inputs(k=512, n=256)
        tuned_config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 512,
            "NUM_KSPLIT": 8,
        }
        with (
            envs.SGLANG_EXPERIMENTAL_MXFP4_MLA_V_BMM_CONFIG.override(""),
            mock.patch.object(
                forward_mla,
                "_get_mxfp4_bmm_config",
                create=True,
                return_value=(tuned_config, None),
            ),
        ):
            config = forward_mla._get_glm_mxfp4_v_bmm_config(x, weight)

        self.assertEqual(config["BLOCK_SIZE_M"], 64)
        self.assertEqual(config["BLOCK_SIZE_N"], 128)
        self.assertEqual(config["BLOCK_SIZE_K"], 512)
        self.assertEqual(config["NUM_KSPLIT"], 1)
        self.assertEqual(tuned_config["NUM_KSPLIT"], 8)

    def test_separate_k_v_overrides_are_legal_and_single_split(self):
        k_x, k_weight = self._config_inputs()
        v_x, v_weight = self._config_inputs(k=512, n=256)
        with (
            envs.SGLANG_EXPERIMENTAL_MXFP4_MLA_K_BMM_CONFIG.override("256,128,32"),
            envs.SGLANG_EXPERIMENTAL_MXFP4_MLA_V_BMM_CONFIG.override("64,256,512"),
            mock.patch.object(
                forward_mla,
                "_get_mxfp4_bmm_config",
                create=True,
                return_value=(
                    {
                        "BLOCK_SIZE_M": 1,
                        "BLOCK_SIZE_N": 1,
                        "BLOCK_SIZE_K": 1,
                        "NUM_KSPLIT": 16,
                    },
                    None,
                ),
            ),
        ):
            k_config = forward_mla._get_glm_mxfp4_k_bmm_config(k_x, k_weight)
            v_config = forward_mla._get_glm_mxfp4_v_bmm_config(v_x, v_weight)

        self.assertEqual(
            (
                k_config["BLOCK_SIZE_M"],
                k_config["BLOCK_SIZE_N"],
                k_config["BLOCK_SIZE_K"],
                k_config["NUM_KSPLIT"],
            ),
            (256, 128, 32, 1),
        )
        self.assertEqual(
            (
                v_config["BLOCK_SIZE_M"],
                v_config["BLOCK_SIZE_N"],
                v_config["BLOCK_SIZE_K"],
                v_config["NUM_KSPLIT"],
            ),
            (64, 256, 512, 1),
        )

    def test_rejects_unsafe_k_block_and_unknown_impl(self):
        x, weight = self._config_inputs()
        with (
            envs.SGLANG_EXPERIMENTAL_MXFP4_MLA_K_BMM_CONFIG.override("64,64,128"),
            mock.patch.object(
                forward_mla,
                "_get_mxfp4_bmm_config",
                create=True,
                return_value=({"NUM_KSPLIT": 1}, None),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "illegal K MXFP4 BMM config"):
                forward_mla._get_glm_mxfp4_k_bmm_config(x, weight)

        with envs.SGLANG_EXPERIMENTAL_MXFP4_MLA_BMM_IMPL.override("unknown"):
            with self.assertRaisesRegex(ValueError, "must be 'a16' or 'a4w4'"):
                forward_mla._get_glm_mxfp4_bmm_impl()


class TestMxfp4PrepackedA4W4Dispatch(CustomTestCase):
    @staticmethod
    def _fake_quantized_layout(x):
        batch, tokens, k = x.shape
        packed = torch.empty(batch, tokens * k // 2, dtype=torch.uint8)
        scales = torch.empty(tokens * k // 32, batch, dtype=torch.uint8).T
        return packed, scales

    def _run_and_check(self, *, operand):
        torch.manual_seed(7)
        batch, tokens = 2, 5
        k, n = (192, 32) if operand == "K" else (512, 64)
        x = torch.randn(batch, tokens, k, dtype=torch.bfloat16)
        reference_weight = torch.randn(batch, n, k, dtype=torch.bfloat16)
        packed_weight = torch.empty(batch, n, k // 2, dtype=torch.uint8)
        weight_scale = torch.empty(batch, n, k // 32, dtype=torch.uint8)
        expected_batched = torch.bmm(x, reference_weight.transpose(-2, -1))
        output = torch.empty(
            (batch, tokens, n) if operand == "K" else (tokens, batch, n),
            dtype=torch.bfloat16,
        )
        expected = (
            expected_batched
            if operand == "K"
            else expected_batched.transpose(0, 1).contiguous()
        )
        observed = {}

        def reference_a4w4(
            x_packed,
            weight,
            x_scale,
            w_scale,
            *,
            dtype,
            y,
            config,
        ):
            self.assertEqual(x_packed.shape, (batch, tokens, k // 2))
            self.assertEqual(x_scale.shape, (batch, tokens, k // 32))
            self.assertEqual(x_scale.stride(), (1, batch * k // 32, batch))
            self.assertIs(weight, packed_weight)
            self.assertIs(w_scale, weight_scale)
            self.assertIs(dtype, torch.bfloat16)
            self.assertEqual(config["NUM_KSPLIT"], 1)
            self.assertEqual(config["SPLITK_BLOCK_SIZE"], k)
            if operand == "K":
                self.assertLessEqual(config["BLOCK_SIZE_K"], 64)
            self.assertEqual(y.data_ptr(), output.data_ptr())
            y.copy_(expected_batched)
            observed["y"] = y

        with (
            envs.SGLANG_USE_MXFP4_MLA_BMM.override(True),
            envs.SGLANG_EXPERIMENTAL_MXFP4_MLA_BMM_IMPL.override("a4w4"),
            envs.SGLANG_EXPERIMENTAL_MXFP4_MLA_K_BMM_CONFIG.override(""),
            envs.SGLANG_EXPERIMENTAL_MXFP4_MLA_V_BMM_CONFIG.override(""),
            mock.patch.object(
                forward_mla,
                "_get_mxfp4_bmm_config",
                create=True,
                return_value=(
                    {
                        "BLOCK_SIZE_M": 128,
                        "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 128,
                        "NUM_KSPLIT": 4,
                    },
                    None,
                ),
            ),
            mock.patch.object(
                forward_mla,
                "fused_flatten_mxfp4_quant",
                create=True,
                side_effect=self._fake_quantized_layout,
            ) as quantize,
            mock.patch.object(
                forward_mla,
                "batched_gemm_afp4wfp4",
                create=True,
                side_effect=reference_a4w4,
            ) as a4w4_bmm,
            mock.patch.object(
                forward_mla, "batched_gemm_a16wfp4", create=True
            ) as a16_bmm,
        ):
            if operand == "K":
                result = forward_mla._run_mxfp4_k_bmm(
                    x, packed_weight, weight_scale, output
                )
                self.assertIsNone(result)
            else:
                result = forward_mla._run_mxfp4_v_bmm(
                    x, packed_weight, weight_scale, output
                )
                self.assertIs(result, output)

        quantize.assert_called_once_with(x)
        a4w4_bmm.assert_called_once()
        a16_bmm.assert_not_called()
        self.assertIsNotNone(observed["y"])
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, expected)
        cosine = torch.nn.functional.cosine_similarity(
            output.float().flatten(), expected.float().flatten(), dim=0
        )
        self.assertGreaterEqual(float(cosine), 0.9999985)
        self.assertTrue(output.is_contiguous())

    def test_k_prepacked_layout_capture_buffer_and_accuracy(self):
        self._run_and_check(operand="K")

    def test_v_prepacked_layout_capture_buffer_and_accuracy(self):
        self._run_and_check(operand="V")

    def test_quantizer_rejects_invalid_capture_shape_before_allocation(self):
        x = torch.empty(2, 3, 48, dtype=torch.bfloat16)
        with mock.patch.object(
            forward_mla, "fused_flatten_mxfp4_quant", create=True
        ) as quantize:
            with self.assertRaisesRegex(ValueError, "K divisible by 32"):
                forward_mla._quantize_mxfp4_bmm_activation(x)
        quantize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
