#!/usr/bin/env bash
# GLM-5.1-FP8 launcher (gfx950, TP8). Used to exercise the dense-MHA short-context
# chunked-prefill fallback path (#30808 + the kv_a_quanted-after-refetch fix).
# Model is served from /dev/shm (RAM tmpfs); override with GLM51_MODEL=.
export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH}
MODEL=${GLM51_MODEL:-/dev/shm/GLM-5.1-FP8}

export SAFETENSORS_FAST_GPU=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

# Small CHUNKED_PREFILL_SIZE forces prefill to split into chunks — the path where
# the dense-MHA fallback re-fetched KV and previously reused a stale kv_a_quanted.
exec sglang serve \
  --model-path "${MODEL}" \
  --tp ${TP:-8} \
  --host localhost \
  --port 8551 \
  --trust-remote-code \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --dsa-prefill-backend triton \
  --dsa-decode-backend triton \
  --enable-aiter-allreduce-fusion \
  --watchdog-timeout 1200 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --kv-cache-dtype fp8_e4m3 \
  --tokenizer-worker-num 8 \
  --chunked-prefill-size ${CHUNKED_PREFILL_SIZE:-131072}
