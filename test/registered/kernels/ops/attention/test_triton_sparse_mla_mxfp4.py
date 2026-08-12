"""ROCm correctness coverage for sparse-MLA MXFP4 producer output."""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd
from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
from sglang.srt.layers.quantization.rocm_mxfp4_utils import batched_mxfp4_quant
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    torch.version.hip is not None and torch.cuda.is_available(),
    "requires a ROCm GPU",
)
class TestTritonSparseMlaMxfp4(CustomTestCase):
    def test_glm_epilogue_matches_post_attention_quantization(self):
        torch.manual_seed(7)
        tokens, heads, d_v, d_tail, topk, pages = 257, 16, 512, 64, 64, 512
        fp8_dtype = torch.float8_e4m3fn

        q_nope = (torch.randn(tokens, heads, d_v, device="cuda") * 0.1).to(
            fp8_dtype
        )
        q_rope = (torch.randn(tokens, heads, d_tail, device="cuda") * 0.1).to(
            fp8_dtype
        )
        kv = (torch.randn(pages, 1, d_v + d_tail, device="cuda") * 0.1).to(
            fp8_dtype
        )
        indices = torch.randint(
            0, pages, (tokens, 1, topk), dtype=torch.int32, device="cuda"
        )

        reference = triton_sparse_mla_fwd(
            q_nope,
            q_rope,
            kv,
            indices,
            sm_scale=(d_v + d_tail) ** -0.5,
            d_v=d_v,
        ).squeeze(0)
        packed, scales = triton_sparse_mla_fwd(
            q_nope,
            q_rope,
            kv,
            indices,
            sm_scale=(d_v + d_tail) ** -0.5,
            d_v=d_v,
            output_mxfp4=True,
        )
        reference_packed, reference_scales = batched_mxfp4_quant(
            reference, block_size_m=16
        )

        self.assertEqual(packed.shape, (tokens, heads, d_v // 2))
        self.assertEqual(scales.shape, (tokens, heads, d_v // 32))
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(scales.dtype, torch.uint8)
        self.assertTrue(torch.equal(packed, reference_packed))
        self.assertTrue(torch.equal(scales, reference_scales))

        fused_dequant = MXFP4QuantizeUtil.dequantize(
            packed, torch.float32, scales, [32]
        )
        reference_dequant = MXFP4QuantizeUtil.dequantize(
            reference_packed, torch.float32, reference_scales, [32]
        )
        cosine = torch.nn.functional.cosine_similarity(
            fused_dequant.flatten(), reference_dequant.flatten(), dim=0
        )
        self.assertGreaterEqual(cosine.item(), 0.9999985)


if __name__ == "__main__":
    unittest.main()
