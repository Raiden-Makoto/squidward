#!/usr/bin/env bash
# GLM-5.3 Quark MXFP4 experts + FP8 attention launcher (MI355X/gfx950).

export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH}
MODEL=${MODEL_PATH:-/data2/hf_home/hub/GLM-5.3-Quark-MXFP4-AttnFP8}
PORT=${PORT:-8553}
TP=${TP:-4}

export SAFETENSORS_FAST_GPU=1
export SGLANG_USE_AITER=${SGLANG_USE_AITER:-1}
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_QUICK_REDUCE_QUANTIZATION=INT4  # ATOM only
# GLM-5.2 optimization paths are retained for controlled GLM-5.3 bring-up.
# Hadamard, PTPC projection, and MXFP4 MLA BMM are validated.
export SGLANG_DSA_FP8_PROJ_GEMM=${SGLANG_DSA_FP8_PROJ_GEMM:-1}
export SGLANG_USE_MXFP4_MLA_BMM=${SGLANG_USE_MXFP4_MLA_BMM:-1}
export SGLANG_DSA_FUSE_HADAMARD_QUANT=${SGLANG_DSA_FUSE_HADAMARD_QUANT:-1}
export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=${SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD:-0}

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5,6,7}
export AITER_USE_FLYDSL_MOE_SORTING=1  # match ATOM + the FlyDSL gemm1 kernels stock aiter tunes to

PROFILE_ARGS=""
EXTRA_ARGS=""
BACKEND_ARGS="--dsa-prefill-backend tilelang --dsa-decode-backend tilelang"
ALLREDUCE_FUSION=${ALLREDUCE_FUSION-"--enable-aiter-allreduce-fusion"}
for arg in "$@"; do
  case "$arg" in
    --profile)
      PROFILE_ARGS="--disable-cuda-graph"
      ;;
    --use-aiter|--aiter)
      export SGLANG_USE_AITER=1
      ALLREDUCE_FUSION="--enable-aiter-allreduce-fusion"
      ;;
    --use-triton|--triton)
      BACKEND_ARGS="--dsa-prefill-backend triton --dsa-decode-backend triton"
      ;;
    --use-tilelang|--tilelang)
      BACKEND_ARGS="--dsa-prefill-backend tilelang --dsa-decode-backend tilelang"
      ;;
    *)
      EXTRA_ARGS="${EXTRA_ARGS} ${arg}"
      ;;
  esac
done

set -x
exec sglang serve \
  --model-path "${MODEL}" \
  ${PROFILE_ARGS} \
  ${EXTRA_ARGS} \
  ${ALLREDUCE_FUSION} \
  --tp "${TP}" \
  --host localhost \
  --port "${PORT}" \
  --trust-remote-code \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  ${BACKEND_ARGS} \
  --watchdog-timeout 1200 \
  --mem-fraction-static 0.80 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --kv-cache-dtype fp8_e4m3 \
  --tokenizer-worker-num 8 \
  --chunked-prefill-size ${CHUNKED_PREFILL_SIZE:-131072}
