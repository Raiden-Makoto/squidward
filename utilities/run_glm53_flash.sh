#!/usr/bin/env bash
# GLM-5.3-Flash FP8 launcher for MI355X/gfx950.

set -euo pipefail

export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH:-}

MODEL=${MODEL_PATH:-/data2/hf_home/hub/models--zai-org--GLM-5.3-Flash/snapshots/03eb5366286afd40d2221b1d9c63a6dd1ba4832e}
PORT=${PORT:-8554}
TP=${TP:-4}

export SAFETENSORS_FAST_GPU=1
export SGLANG_USE_AITER=${SGLANG_USE_AITER:-1}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES}}

PROFILE_ARGS=()
EXTRA_ARGS=()
BACKEND_ARGS=(
  --dsa-prefill-backend tilelang
  --dsa-decode-backend tilelang
)

for arg in "$@"; do
  case "${arg}" in
    --profile)
      PROFILE_ARGS=(--disable-cuda-graph)
      ;;
    --use-aiter|--aiter)
      export SGLANG_USE_AITER=1
      ;;
    --use-triton|--triton)
      BACKEND_ARGS=(
        --dsa-prefill-backend triton
        --dsa-decode-backend triton
      )
      ;;
    --use-tilelang|--tilelang)
      BACKEND_ARGS=(
        --dsa-prefill-backend tilelang
        --dsa-decode-backend tilelang
      )
      ;;
    *)
      EXTRA_ARGS+=("${arg}")
      ;;
  esac
done

set -x
exec python3 -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp "${TP}" \
  --trust-remote-code \
  --kv-cache-dtype bfloat16 \
  --context-length 131072 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --moe-runner-backend aiter \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  "${BACKEND_ARGS[@]}" \
  "${PROFILE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
