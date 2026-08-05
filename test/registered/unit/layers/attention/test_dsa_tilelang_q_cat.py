"""Unit tests for the fused TileLang DSA q_cat handoff."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


class _StopForward(Exception):
    pass


class TestDsaTilelangQCat(CustomTestCase):
    def test_fused_prefill_forwards_q_cat_directly(self):
        q_cat = torch.randn(2, 2, 6)
        k_pe_fused = torch.randn(2, 1, 2)
        attention = MagicMock(side_effect=_StopForward)
        attention.layer_id = 3
        attention.k_scale = 1.0
        mla = SimpleNamespace(
            current_attention_backend="dsa",
            _skip_rope_for_dsa_tilelang_fused=lambda: True,
            rotary_emb=SimpleNamespace(
                cos_cache=torch.empty(0),
                sin_cache=torch.empty(0),
                is_neox_style=False,
            ),
            kv_cache_dtype="bfloat16",
            kv_lora_rank=4,
            attn_mqa=attention,
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode_or_idle=lambda: False,
                is_extend=lambda: True,
                is_target_verify=lambda: False,
            ),
            out_cache_loc=torch.tensor([0, 1]),
        )
        token_pool = SimpleNamespace(
            get_key_buffer=MagicMock(return_value=torch.empty(0))
        )

        with (
            patch.object(
                forward_mla,
                "FORWARD_ABSORB_CORE_ATTENTION_BACKENDS",
                {"dsa"},
            ),
            patch.object(
                forward_mla,
                "fused_qk_rope_cat_and_cache_mla",
                return_value=(q_cat, None, k_pe_fused, None),
                create=True,
            ),
            patch.object(forward_mla, "get_token_to_kv_pool", return_value=token_pool),
            patch.object(
                forward_mla,
                "get_attn_backend",
                return_value=SimpleNamespace(dsa_prefill_impl="tilelang"),
            ),
            self.assertRaises(_StopForward),
        ):
            forward_mla.DeepseekMLAForwardMixin.forward_absorb_core(
                mla,
                torch.randn(2, 2, 2),
                torch.randn(2, 1, 2),
                torch.randn(2, 2, 4),
                torch.randn(2, 1, 4),
                forward_batch,
                None,
                torch.tensor([0, 1]),
                None,
                None,
            )

        args, kwargs = attention.call_args
        self.assertIs(args[0], q_cat)
        self.assertIsNone(kwargs["q_rope"])
        self.assertIs(kwargs["k_rope"], k_pe_fused)
        self.assertFalse(kwargs["save_kv_cache"])

    def test_tilelang_prefill_reuses_concatenated_q_without_cat(self):
        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.forward_metadata = SimpleNamespace()
        backend.dsa_prefill_impl = "tilelang"
        backend.dsa_decode_impl = "tilelang"
        backend.use_mha = False
        backend.use_fused_topk = True
        backend.hisparse_coordinator = None
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=MagicMock(return_value=torch.empty(0))
        )
        backend._get_fused_topk_page_table = MagicMock(
            return_value=torch.tensor([[0]], dtype=torch.int32)
        )
        output = torch.randn(2, 2, 4)
        backend._forward_tilelang = MagicMock(return_value=output)
        backend.get_topk_transform_method = MagicMock(return_value=None)

        q_cat = torch.randn(2, 2, 6).contiguous()
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=3,
            tp_q_head_num=2,
            v_head_dim=4,
            head_dim=6,
            scaling=0.5,
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_target_verify=lambda: False,
                is_draft_extend_v2=lambda: False,
            )
        )

        with patch.object(
            dsa_backend, "concat_mla_absorb_q_general"
        ) as concat_mla_absorb_q:
            actual = backend.forward_extend(
                q_cat,
                None,
                None,
                layer,
                forward_batch,
                save_kv_cache=False,
                q_rope=None,
            )

        self.assertIs(actual, output)
        concat_mla_absorb_q.assert_not_called()
        q_all = backend._forward_tilelang.call_args.kwargs["q_all"]
        self.assertEqual(q_all.data_ptr(), q_cat.data_ptr())
        self.assertEqual(q_all.stride(), q_cat.stride())


if __name__ == "__main__":
    unittest.main()
