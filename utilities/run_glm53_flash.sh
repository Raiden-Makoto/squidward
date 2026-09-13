#!/usr/bin/env bash
# GLM-5.3-Flash Quark MXFP4 experts + FP8 attention launcher for MI355X/gfx950.

set -euo pipefail

export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH:-}

MODEL=${MODEL_PATH:-/data2/hf_home/hub/models--amd--GLM-5.3-Flash-Quark-MXFP4/snapshots/b5688f25491202978c19c4d036eef579f61bbe07}
PORT=${PORT:-8554}
TP=${TP:-4}

export SAFETENSORS_FAST_GPU=1
export SGLANG_USE_AITER=${SGLANG_USE_AITER:-1}
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=${ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}
export AITER_QUICK_REDUCE_QUANTIZATION=${AITER_QUICK_REDUCE_QUANTIZATION:-INT4}
export AITER_USE_FLYDSL_MOE_SORTING=${AITER_USE_FLYDSL_MOE_SORTING:-1}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5,6,7}

PROFILE_ARGS=()
EXTRA_ARGS=()

for arg in "$@"; do
  case "${arg}" in
    --profile)
      PROFILE_ARGS=(--disable-cuda-graph)
      ;;
    --use-aiter|--aiter)
      export SGLANG_USE_AITER=1
      ;;
    --use-triton|--triton)
      echo "GLM-5.3-Flash does not support the Triton DSA backend on ROCm." >&2
      exit 2
      ;;
    --use-tilelang|--tilelang)
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
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --moe-runner-backend aiter \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  "${PROFILE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
