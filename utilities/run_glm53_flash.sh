#!/usr/bin/env bash
# GLM-5.3-Flash Quark MXFP4 experts + FP8 attention launcher for MI355X/gfx950.
#
# Usage:
#   bash utilities/run_glm53_flash.sh --mxfp4
#   bash utilities/run_glm53_flash.sh --fp8
#   bash utilities/run_glm53_flash.sh --mxfp4 --profile
#   TP=8 bash utilities/run_glm53_flash.sh --fp8

set -euo pipefail

export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH:-}

MXFP4_MODEL=/data2/hf_home/hub/models--amd--GLM-5.3-Flash-Quark-MXFP4/snapshots/b5688f25491202978c19c4d036eef579f61bbe07
FP8_MODEL=/data2/hf_home/hub/models--zai-org--GLM-5.3-Flash/snapshots/03eb5366286afd40d2221b1d9c63a6dd1ba4832e
MODEL=${MODEL_PATH:-${MXFP4_MODEL}}
PORT=${PORT:-8554}
TP=${TP:-4}

export SAFETENSORS_FAST_GPU=1
export SGLANG_USE_AITER=${SGLANG_USE_AITER:-1}

PROFILE_ARGS=()
EXTRA_ARGS=()
TP_ARGS=()

for arg in "$@"; do
  case "${arg}" in
    --mxfp4)
      MODEL=${MXFP4_MODEL}
      ;;
    --fp8)
      MODEL=${FP8_MODEL}
      ;;
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

case "${TP}" in
  4)
    export SGLANG_ROCM_FUSED_DECODE_MLA=0
    export ROCM_QUICK_REDUCE_QUANTIZATION=${ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}
    export AITER_QUICK_REDUCE_QUANTIZATION=${AITER_QUICK_REDUCE_QUANTIZATION:-INT4}
    export AITER_USE_FLYDSL_MOE_SORTING=${AITER_USE_FLYDSL_MOE_SORTING:-1}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5,6,7}
    TP_ARGS=(
      --attention-backend dsa
      --context-length 65536
      --linear-attn-backend triton
      --disable-shared-experts-fusion
      --max-running-requests 64
      --cuda-graph-backend-decode full
      --cuda-graph-max-bs-decode 64
      --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}'
      --watchdog-timeout 1200
    )
    ;;
  8)
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    TP_ARGS=(
      --context-length 131072
    )
    ;;
  *)
    echo "TP must be 4 or 8, got ${TP}." >&2
    exit 2
    ;;
esac

set -x
exec python3 -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp "${TP}" \
  --trust-remote-code \
  --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --moe-runner-backend aiter \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  "${TP_ARGS[@]}" \
  "${PROFILE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
