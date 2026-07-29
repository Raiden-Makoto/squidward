#!/usr/bin/env bash
# GLM-5.2-MXFP4 launcher (gfx950).
#
export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH}
MODEL=${HF_HOME:-/root/hf_home}/hub/models--amd--GLM-5.2-MXFP4/snapshots/386bd0e4ec821f7b07975701cec3c3b953a5576a
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export SAFETENSORS_FAST_GPU=1
# Master aiter switch (env-only, upstream default off); whole GLM-5.2 gfx950 path needs it.
export SGLANG_USE_AITER=${SGLANG_USE_AITER:-1}
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_QUICK_REDUCE_QUANTIZATION=INT4  # ATOM only
# Tuned a8w8_blockscale_bpreshuffle rows (also upstream via ROCm/aiter#4243); pin in-repo copy.
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE:-${SCRIPT_DIR}/glm5_a8w8_blockscale_bpreshuffle_tuned.csv}
# Tuned MoE (fmoe) config: upstream aiter glm5_fp4 tiles.
export AITER_CONFIG_FMOE=${AITER_CONFIG_FMOE:-${SCRIPT_DIR}/glm5_fp4_tuned_fmoe.csv}

export HIP_VISIBLE_DEVICES=4,5,6,7
export AITER_USE_FLYDSL_MOE_SORTING=1  # match ATOM + the FlyDSL kernels in AITER_CONFIG_FMOE

PROFILE_ARGS=""
SPEC_ARGS=""
EXTRA_ARGS=""
# aiter TP allreduce+RMSNorm fusion, on by default (self-disables under deterministic
# inference). Set ALLREDUCE_FUSION="" to drop it.
ALLREDUCE_FUSION=${ALLREDUCE_FUSION-"--enable-aiter-allreduce-fusion"}
for arg in "$@"; do
  case "$arg" in
    --profile)
      PROFILE_ARGS="--disable-cuda-graph"
      ;;
    --speculative|--spec)
      export SGLANG_ENABLE_SPEC_V2=1
      SPEC_ARGS="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --attention-backend triton"
      ;;
    --use-aiter|--aiter)
      # No-op: SGLANG_USE_AITER + allreduce fusion are already on by default above.
      # Kept for backwards compat so old launch commands still parse.
      export SGLANG_USE_AITER=1
      ALLREDUCE_FUSION="--enable-aiter-allreduce-fusion"
      ;;
    *)
      # forward any other flag straight to `sglang serve`
      EXTRA_ARGS="${EXTRA_ARGS} ${arg}"
      ;;
  esac
done

set -x
exec sglang serve \
  --model-path "${MODEL}" \
  ${PROFILE_ARGS} \
  ${SPEC_ARGS} \
  ${EXTRA_ARGS} \
  ${ALLREDUCE_FUSION} \
  --tp 4 \
  --host localhost \
  --port 8552 \
  --trust-remote-code \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --watchdog-timeout 1200 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --kv-cache-dtype fp8_e4m3 \
  --tokenizer-worker-num 8 \
  --chunked-prefill-size ${CHUNKED_PREFILL_SIZE:-131072}
