import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    torch.version.hip
    and torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    "gfx950 (MI35x) only",
)
class TestGlmTopkGfx950(CustomTestCase):
    def _reference(self, logits, bias, rsf, n_valid=None):
        from aiter import biased_grouped_topk

        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            fused_append_shared_experts,
        )

        w = torch.empty(logits.shape[0], 8, dtype=torch.float32, device="cuda")
        ids = torch.empty(logits.shape[0], 8, dtype=torch.int32, device="cuda")
        biased_grouped_topk(logits, bias, w, ids, 1, 1, True, rsf)
        return fused_append_shared_experts(
            ids, w, 1, 1.0, 288, num_token_non_padded=n_valid
        )

    def _assert_bit_equal(self, ids, w, ref_ids, ref_w, msg):
        self.assertTrue(torch.equal(ids, ref_ids), msg)
        self.assertTrue(torch.equal(w.view(torch.int32), ref_w.view(torch.int32)), msg)

    def test_matches_aiter_topk_and_append(self):
        from sglang.kernels.ops.moe import glm_topk_gfx950 as T

        torch.manual_seed(0)
        # GLM-5.3-Flash-like biases: a narrow band around 7.5, a few bf16 levels wide
        bias = (7.4 + 0.4 * torch.rand(288, device="cuda")).bfloat16()
        for tok in (1, 7, 64, 192, 513, 4096):
            gaussian = torch.randn(tok, 288, device="cuda").bfloat16()
            # a few bf16 levels: exact score ties everywhere, where aiter's tie order must hold
            ties = (torch.randint(-3, 4, (tok, 288), device="cuda") * 0.25).bfloat16()
            for logits in (gaussian, ties):
                self.assertTrue(T.glm_topk_supported(logits, bias))
                w, ids = T.glm_topk(logits, bias, 2.5, 1.0)
                ref_ids, ref_w = self._reference(logits, bias, 2.5)
                self._assert_bit_equal(ids, w, ref_ids, ref_w, f"tok={tok}")
            for dtype in (torch.int32, torch.int64):
                n_valid = torch.tensor([max(1, tok - 2)], dtype=dtype, device="cuda")
                w, ids = T.glm_topk(gaussian, bias, 2.5, 1.0, n_valid)
                ref_ids, ref_w = self._reference(gaussian, bias, 2.5, n_valid)
                self._assert_bit_equal(ids, w, ref_ids, ref_w, f"padded tok={tok}")

        for bad_logits, bad_bias in (
            (gaussian.half(), bias),
            (gaussian[:, :256], bias),
            (gaussian, bias.float()),
        ):
            self.assertFalse(T.glm_topk_supported(bad_logits, bad_bias))
        gfx942 = SimpleNamespace(hip=SimpleNamespace(gcn_arch="gfx942"))
        with mock.patch.object(T, "current_arch", return_value=gfx942):
            self.assertFalse(T.glm_topk_supported(gaussian, bias))

        logits = torch.randn(192, 288, device="cuda").bfloat16()
        n_valid = torch.tensor([187], dtype=torch.int32, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gw, gids = T.glm_topk(logits, bias, 2.5, 1.0, n_valid)
        logits.copy_(torch.randn(192, 288, device="cuda").bfloat16())
        g.replay()
        torch.cuda.synchronize()
        ref_ids, ref_w = self._reference(logits, bias, 2.5, n_valid)
        self._assert_bit_equal(gids, gw, ref_ids, ref_w, "graph replay")


if __name__ == "__main__":
    unittest.main()
