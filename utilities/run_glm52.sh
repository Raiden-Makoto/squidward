#!/usr/bin/env bash
# GLM-5.2-MXFP4 launcher (gfx950).
#
# Requires transformers >= 5.12.x (image v0.5.14-rocm720-mi35x-20260630+). Older
# transformers (5.8.1) had a GlmMoeDsaConfig alias bug ("head_dim":
# "qk_rope_head_dim") that set qk_rope_head_dim=192 (==head_dim) for GLM-5.2,
# building the MLA q_b/fused_qkv_a 384-wide vs the checkpoint's 256-wide -> load
# failure. On 5.12.1 the config resolves qk_rope_head_dim=64 natively; if you
# must run on an older-transformers image, add:
#   --json-model-override-args '{"qk_rope_head_dim": 64}'
export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH}
MODEL=${HF_HOME:-/root/hf_home}/hub/models--amd--GLM-5.2-MXFP4/snapshots/386bd0e4ec821f7b07975701cec3c3b953a5576a

export SAFETENSORS_FAST_GPU=1
# gfx950 Triton fp8 sparse-MLA prefill (validated on GLM-5.2-MXFP4: 312 hits,
# GSM8K 0.950). Opt-in upstream; we default it on for this gfx950 launcher.
export SGLANG_DSA_TRITON_PREFILL=1
# InferenceMax MI355X tuning (from Jacob0226 GLM.sh / InferenceX glm5_fp8_mi355x.sh).
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

PROFILE_ARGS=""
SPEC_ARGS=""
for arg in "$@"; do
  case "$arg" in
    --profile)
      PROFILE_ARGS="--disable-cuda-graph"
      ;;
    --speculative|--spec)
      # EAGLE MTP speculative decoding, 3-1-4 (steps=3, topk=1, draft=4) matching
      # the InferenceMax MI355X recipe; topk=1 is the only DSA-supported mode.
      export SGLANG_ENABLE_SPEC_V2=1
      SPEC_ARGS="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
      ;;
  esac
done

set -x
exec sglang serve \
  --model-path "${MODEL}" \
  ${PROFILE_ARGS} \
  ${SPEC_ARGS} \
  --tp 4 \
  --host localhost \
  --port 8552 \
  --trust-remote-code \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --watchdog-timeout 1200 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --kv-cache-dtype fp8_e4m3 \
  --tokenizer-worker-num 8 \
  --chunked-prefill-size 131072
