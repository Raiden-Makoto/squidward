# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.unquant import Glm53DraftLmHeadFp8Method
from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.utils import is_gfx95_supported

logger = logging.getLogger(__name__)

# Draft token counts measured faster in FP8 (torch._scaled_mm) than BF16 for the TP4
# vocab shard on gfx950; larger draft batches keep the BF16 head.
GLM53_DRAFT_FP8_LM_HEAD_MAX_M = 384


class Glm5NextForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    def prepare_cp_inputs(self, forward_batch, **kwargs):
        # The worker has rotated the target's multimodal embeddings per request.
        # Fill each appended token while indices still address full sequences,
        # then let the runner shard embeddings and target states together.
        input_embeds = self.model.embed_input_ids(
            forward_batch.input_ids, forward_batch
        )
        return input_embeds, forward_batch.positions, {}

    @classmethod
    def get_hf_to_sglang_mapper(cls, config) -> WeightsMapper:
        text_config = getattr(config, "text_config", config)
        n = text_config.num_hidden_layers
        # lookups arrive as checkpoint and normalized names, so every rule has both forms
        draft_rules: dict[str, str] = {}
        for ckpt_prefix in (f"model.language_model.layers.{n}", f"model.layers.{n}"):
            # eh_proj/enorm/hnorm sit beside the block under `model`, not in `decoder`
            draft_rules[f"{ckpt_prefix}.eh_proj"] = "model.eh_proj"
            draft_rules[f"{ckpt_prefix}.enorm"] = "model.enorm"
            draft_rules[f"{ckpt_prefix}.hnorm"] = "model.hnorm"
            draft_rules[ckpt_prefix] = "model.decoder"
        # target rules still normalize the non-draft layers and vision tower in `exclude`
        return Glm5NextForConditionalGeneration.hf_to_sglang_mapper | WeightsMapper(
            orig_to_new_substr=draft_rules,
        )

    def _resolve_nextn_quant_config(self, config, quant_config):
        """Mixed checkpoints list the BF16 NextN block in ``quantization_config.ignore``;
        inheriting global FP8 quantization would corrupt its QKV weights."""
        raw_quant_config = getattr(config, "quantization_config", None) or {}
        if hasattr(raw_quant_config, "to_dict"):
            raw_quant_config = raw_quant_config.to_dict()
        ignored = (
            raw_quant_config.get("ignore", [])
            if isinstance(raw_quant_config, dict)
            else []
        )
        nextn_layer_pattern = f"model.layers.{config.num_hidden_layers}.*"
        if nextn_layer_pattern in ignored:
            logger.warning(
                "GLM5 NextN layer %s is checkpoint-declared unquantized; "
                "using BF16 draft modules",
                nextn_layer_pattern,
            )
            return None
        return super()._resolve_nextn_quant_config(config, quant_config)

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(
            getattr(config, "text_config", config),
            quant_config=quant_config,
            prefix=prefix,
        )

    def set_embed_and_head(self, embed, head):
        super().set_embed_and_head(embed, head)
        self._prepare_draft_fp8_lm_head()

    def _prepare_draft_fp8_lm_head(self) -> None:
        processor = self.logits_processor
        if not (
            envs.SGLANG_OPT_GLM53_FAST_DECODE.get()
            and envs.SGLANG_USE_AITER.get()
            and is_gfx95_supported()
            and self.lm_head.weight.dtype == torch.bfloat16
            and type(getattr(self.lm_head, "quant_method", None)).__name__
            == "UnquantizedEmbeddingMethod"
            and not processor.use_fp32_lm_head
            and not processor.use_fp32_lm_head_output
            and processor.rl_on_policy_target is None
        ):
            return
        Glm53DraftLmHeadFp8Method.prepare(self.lm_head)
        self.lm_head.quant_method = Glm53DraftLmHeadFp8Method(
            max_m=GLM53_DRAFT_FP8_LM_HEAD_MAX_M
        )
        logger.info(
            "GLM-5.3 MTP draft lm_head runs in FP8 up to %d tokens",
            GLM53_DRAFT_FP8_LM_HEAD_MAX_M,
        )

    def load_weights(self, weights):
        if not hasattr(self, "fuse_qkv_a_proj"):
            self.fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None
        layer_id = self.config.num_hidden_layers
        layer_prefixes = (
            f"model.layers.{layer_id}.",
            f"model.language_model.layers.{layer_id}.",
        )
        nextn_weights = (
            (name, weight)
            for name, weight in weights
            if name.startswith(layer_prefixes)
        )
        return Glm5NextForConditionalGeneration.load_weights(
            self, nextn_weights, is_nextn=True
        )


EntryClass = [Glm5NextForConditionalGenerationNextN]
