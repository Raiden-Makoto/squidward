"""Unit tests for hybrid FP8 dense-projection scale-contract dispatch."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.unquant import (
    UnquantizedLinearMethod,
    fp8_proj_num_sequences,
    fp8_proj_uses_ptpc_at_batch,
    tag_fp8_proj_input,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hybrid_layer():
    return SimpleNamespace(
        _fp8_proj_gemm="q_b_proj",
        _fp8_proj_mode="hybrid",
        _fp8_proj_ready=True,
        _fp8_proj_blockscale_weight=torch.full((128, 128), 1, dtype=torch.uint8),
        _fp8_proj_blockscale_weight_scale=torch.full((1, 1), 2.0),
        _fp8_proj_ptpc_weight=torch.full((128, 128), 3, dtype=torch.uint8),
        _fp8_proj_ptpc_weight_scale=torch.full((128, 1), 4.0),
    )


class TestHybridFp8Proj(CustomTestCase):
    def test_sequence_count_15_16_contract_boundary(self):
        layer = _hybrid_layer()
        with envs.SGLANG_DSA_FP8_PROJ_HYBRID_SEQ_MIN.override(16):
            self.assertFalse(fp8_proj_uses_ptpc_at_batch(layer, 15))
            self.assertTrue(fp8_proj_uses_ptpc_at_batch(layer, 16))

            layer._fp8_proj_mode = "blockscale"
            self.assertFalse(fp8_proj_uses_ptpc_at_batch(layer, 16))
            layer._fp8_proj_mode = "ptpc"
            self.assertTrue(fp8_proj_uses_ptpc_at_batch(layer, 15))
            layer._fp8_proj_mode = "mixed"
            self.assertFalse(fp8_proj_uses_ptpc_at_batch(layer, 16))

    def test_large_prefill_uses_sequence_count_not_token_m(self):
        layer = _hybrid_layer()
        forward_batch = SimpleNamespace(batch_size=8)
        token_m = 8192
        self.assertGreaterEqual(token_m, 16)
        self.assertEqual(fp8_proj_num_sequences(forward_batch), 8)
        self.assertFalse(
            fp8_proj_uses_ptpc_at_batch(
                layer, fp8_proj_num_sequences(forward_batch)
            )
        )

    def test_cuda_graph_padding_is_excluded_from_sequence_count(self):
        forward_batch = SimpleNamespace(batch_size=16, num_padding=1)
        self.assertEqual(fp8_proj_num_sequences(forward_batch), 15)

    def test_consumer_uses_authoritative_producer_tag_and_weights(self):
        layer = _hybrid_layer()
        method = UnquantizedLinearMethod()
        block_out = torch.empty((15, 128), dtype=torch.bfloat16)
        ptpc_out = torch.empty((16, 128), dtype=torch.bfloat16)

        with (
            mock.patch.object(
                fp8_utils,
                "aiter_w8a8_block_fp8_linear",
                return_value=block_out,
            ) as block_gemm,
            mock.patch.object(
                fp8_utils,
                "apply_fp8_ptpc_linear",
                return_value=ptpc_out,
            ) as ptpc_gemm,
        ):
            block_input = tag_fp8_proj_input(
                (
                    torch.empty((8192, 128), dtype=torch.uint8),
                    torch.empty((64, 1)),
                ),
                use_ptpc=False,
            )
            self.assertIs(method.apply(layer, block_input), block_out)
            block_gemm.assert_called_once()
            self.assertIs(
                block_gemm.call_args.args[1], layer._fp8_proj_blockscale_weight
            )
            self.assertIs(
                block_gemm.call_args.kwargs["input_scale"], block_input[1]
            )
            ptpc_gemm.assert_not_called()

            ptpc_input = tag_fp8_proj_input(
                (
                    torch.empty((2, 8, 128), dtype=torch.uint8),
                    torch.empty((16, 1)),
                ),
                use_ptpc=True,
            )
            self.assertIs(method.apply(layer, ptpc_input), ptpc_out)
            ptpc_gemm.assert_called_once()
            self.assertIs(
                ptpc_gemm.call_args.args[1], layer._fp8_proj_ptpc_weight
            )
            self.assertIs(
                ptpc_gemm.call_args.args[2], layer._fp8_proj_ptpc_weight_scale
            )

    def test_consumer_rejects_untagged_hybrid_tuple(self):
        layer = _hybrid_layer()
        value = (
            torch.empty((8, 128), dtype=torch.uint8),
            torch.empty((8, 1)),
        )
        with self.assertRaisesRegex(RuntimeError, "producer-tagged"):
            UnquantizedLinearMethod().apply(layer, value)

    def test_hybrid_load_retains_both_weight_contracts(self):
        layer = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
        layer._fp8_proj_gemm = "o_proj"
        block_weight = torch.full((128, 128), 1, dtype=torch.uint8)
        block_scale = torch.full((1, 1), 2.0)
        ptpc_weight = torch.full((128, 128), 3, dtype=torch.uint8)
        ptpc_scale = torch.full((128, 1), 4.0)

        fake_aiter = types.ModuleType("aiter")
        fake_aiter.dtypes = SimpleNamespace(fp8=torch.float8_e4m3fn)
        fake_aiter.pertoken_quant = mock.Mock(return_value=(ptpc_weight, ptpc_scale))
        fake_ops = types.ModuleType("aiter.ops")
        fake_shuffle = types.ModuleType("aiter.ops.shuffle")
        fake_shuffle.shuffle_weight = mock.Mock(side_effect=lambda weight, _: weight)

        with (
            envs.SGLANG_DSA_FP8_PROJ_MODE.override("hybrid"),
            mock.patch.dict(
                sys.modules,
                {
                    "aiter": fake_aiter,
                    "aiter.ops": fake_ops,
                    "aiter.ops.shuffle": fake_shuffle,
                },
            ),
            mock.patch.object(
                fp8_utils,
                "quant_weight_ue8m0",
                return_value=(block_weight, block_scale),
            ),
            mock.patch.object(fp8_utils, "_use_aiter_bpreshuffle_gfx95", True),
        ):
            UnquantizedLinearMethod._repack_bf16_to_fp8(layer)

        self.assertEqual(layer._fp8_proj_mode, "hybrid")
        self.assertTrue(layer._fp8_proj_ready)
        self.assertIs(layer._fp8_proj_blockscale_weight, block_weight)
        self.assertIs(layer._fp8_proj_blockscale_weight_scale, block_scale)
        self.assertIs(layer._fp8_proj_ptpc_weight, ptpc_weight)
        self.assertIs(layer._fp8_proj_ptpc_weight_scale, ptpc_scale)
        self.assertEqual(fake_shuffle.shuffle_weight.call_count, 2)


if __name__ == "__main__":
    unittest.main()
