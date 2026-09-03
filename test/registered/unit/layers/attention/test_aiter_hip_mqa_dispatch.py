import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.dsa import dsa_indexer

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAiterHipMqaDispatch(CustomTestCase):
    def setUp(self):
        self.q = torch.empty((3, 32, 128), dtype=torch.uint8)
        self.kv = torch.empty((513, 128), dtype=torch.uint8)
        self.kv_scales = torch.empty(513, dtype=torch.float32)
        self.weights = torch.empty((3, 32), dtype=torch.float32)
        self.cu_starts = torch.zeros(3, dtype=torch.int32)
        self.cu_ends = torch.full((3,), 513, dtype=torch.int32)
        self.fallback = MagicMock(return_value="fallback")

    def _dispatch(self):
        return dsa_indexer._aiter_fp8_mqa_logits_for_hip_kernel_ab(
            self.q,
            self.kv,
            self.kv_scales,
            self.weights,
            self.cu_starts,
            self.cu_ends,
            self.fallback,
        )

    def test_supported_shape_uses_hip_kernel(self):
        hip_kernel = MagicMock(return_value="hip")
        is_supported = MagicMock(return_value=True)
        with (
            patch.object(dsa_indexer, "aiter_hip_fp8_mqa_logits", hip_kernel),
            patch.object(dsa_indexer, "aiter_hip_mqa_supported", is_supported),
        ):
            result = self._dispatch()

        self.assertEqual(result, "hip")
        is_supported.assert_called_once_with(32, 128)
        hip_kernel.assert_called_once_with(
            self.q,
            self.kv,
            self.kv_scales,
            self.weights,
            self.cu_starts,
            self.cu_ends,
            clean_logits=False,
        )
        self.fallback.assert_not_called()

    def test_unsupported_shape_uses_fallback(self):
        hip_kernel = MagicMock()
        with (
            patch.object(dsa_indexer, "aiter_hip_fp8_mqa_logits", hip_kernel),
            patch.object(dsa_indexer, "aiter_hip_mqa_supported", return_value=False),
        ):
            result = self._dispatch()

        self.assertEqual(result, "fallback")
        hip_kernel.assert_not_called()
        self.fallback.assert_called_once_with(
            self.q,
            self.kv,
            self.kv_scales,
            self.weights,
            self.cu_starts,
            self.cu_ends,
            clean_logits=False,
        )

    def test_unavailable_kernel_uses_fallback(self):
        with (
            patch.object(dsa_indexer, "aiter_hip_fp8_mqa_logits", None),
            patch.object(dsa_indexer, "aiter_hip_mqa_supported", None),
        ):
            result = self._dispatch()

        self.assertEqual(result, "fallback")
        self.fallback.assert_called_once()

    def test_hip_kernel_error_propagates(self):
        with (
            patch.object(
                dsa_indexer,
                "aiter_hip_fp8_mqa_logits",
                side_effect=RuntimeError("kernel failed"),
            ),
            patch.object(dsa_indexer, "aiter_hip_mqa_supported", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                self._dispatch()

        self.fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
