#!/usr/bin/env bash
export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH}
MODEL=${HF_HOME:-/root/hf_home}/hub/models--amd--GLM-5.1-MXFP4/snapshots/4dded8b53961222a5d98378fa8b975bc8816599d

export SAFETENSORS_FAST_GPU=1

PROFILE_ARGS=""
SPEC_ARGS=""
for arg in "$@"; do
  case "$arg" in
    --profile)
      PROFILE_ARGS="--disable-cuda-graph"
      ;;
    --speculative|--spec)
      # EAGLE MTP speculative decoding (validated on MI355X/gfx950, lossless;
      # see glm5-prefill-opt-log.mdc). depth 4 (steps=4, draft=5) is the tuned
      # optimum after the fused_metadata_copy ROCm fix unblocked depth>=4
      # (accept ~4.6, GSM8K 0.95/invalid 0). topk=1 (chain) is the only mode the
      # DSA backend supports. Opt-in, not default.
      SPEC_ARGS="--speculative-algorithm EAGLE --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5"
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
  --tokenizer-worker-num 8
