#!/usr/bin/env bash
# GLM-5.2-MXFP4 launcher (gfx950).
#
export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH}
MODEL=${HF_HOME:-/root/hf_home}/hub/models--amd--GLM-5.2-MXFP4/snapshots/386bd0e4ec821f7b07975701cec3c3b953a5576a

export SAFETENSORS_FAST_GPU=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_DSA_TRITON_PREFILL=1
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

PROFILE_ARGS=""
SPEC_ARGS=""
for arg in "$@"; do
  case "$arg" in
    --profile)
      PROFILE_ARGS="--disable-cuda-graph"
      ;;
    --speculative|--spec)
      export SGLANG_ENABLE_SPEC_V2=1
      SPEC_ARGS="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --attention-backend triton"
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
  --dsa-prefill-backend triton \
  --dsa-decode-backend triton \
  --watchdog-timeout 1200 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --kv-cache-dtype fp8_e4m3 \
  --tokenizer-worker-num 8 \
  --chunked-prefill-size 131072
