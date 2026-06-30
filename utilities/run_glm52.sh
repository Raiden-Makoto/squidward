#!/usr/bin/env bash
# GLM-5.2-MXFP4 launcher (gfx950). Based on run_glm5.sh (GLM-5.1).
#
# TEMP WORKAROUND: the amd/GLM-5.2-MXFP4 checkpoint's loaded config reports
# qk_rope_head_dim=192 (== head_dim) even though config.json says 64. On current
# sglang main that makes the MLA attention build q_b/fused_qkv_a at 384-wide vs
# the checkpoint's 256-wide, so the model fails to load (base AND spec). Forcing
# qk_rope_head_dim=64 via --json-model-override-args corrects qk_head to 256 and
# the weights load. (EAGLE/--spec additionally needs the nextn eh_proj quark
# exclude fix; base serving works with the override alone.)
export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH}
MODEL=${HF_HOME:-/root/hf_home}/hub/models--amd--GLM-5.2-MXFP4/snapshots/386bd0e4ec821f7b07975701cec3c3b953a5576a

export SAFETENSORS_FAST_GPU=1

PROFILE_ARGS=""
SPEC_ARGS=""
for arg in "$@"; do
  case "$arg" in
    --profile)
      PROFILE_ARGS="--disable-cuda-graph"
      ;;
    --speculative|--spec)
      # EAGLE MTP speculative decoding. GLM-5.2 cookbook low-latency config is
      # 5-1-6 (steps=5, topk=1, draft=6); topk=1 is the only DSA-supported mode.
      SPEC_ARGS="--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6"
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
  --json-model-override-args '{"qk_rope_head_dim": 64}'
