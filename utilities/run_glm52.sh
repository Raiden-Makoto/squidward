#!/usr/bin/env bash
# GLM-5.2-MXFP4 launcher (gfx950).
#
export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH}
MODEL=${HF_HOME:-/root/hf_home}/hub/models--amd--GLM-5.2-MXFP4/snapshots/386bd0e4ec821f7b07975701cec3c3b953a5576a
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export SAFETENSORS_FAST_GPU=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
# fp8 dense projections (q_b_proj/o_proj) -> tuned fp8 a8w8_blockscale_bpreshuffle CK GEMM.
# Dense GEMM section -8.8ms/-9.5% (o_proj 45.9->27.9). Untuned = +8.8ms regression, so the
# tuned config below is REQUIRED. The box aiter is stock (config edits are box-local/lost),
# so we carry the tuned rows for q_b_proj(4096,2048)/o_proj(6144,4096) in-repo and point aiter
# at them via env (same pattern as AITER_CONFIG_FMOE). Disable via SGLANG_DSA_FP8_PROJ_GEMM=0.
export SGLANG_DSA_FP8_PROJ_GEMM=${SGLANG_DSA_FP8_PROJ_GEMM:-1}
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE:-${SCRIPT_DIR}/glm5_a8w8_blockscale_bpreshuffle_tuned.csv}
# Tuned MoE (fmoe) config: upstream aiter glm5_fp4 tiles (faster than the old
# glm_fp8fp4). Override by exporting AITER_CONFIG_FMOE before launch.
export AITER_CONFIG_FMOE=${AITER_CONFIG_FMOE:-${SCRIPT_DIR}/glm5_fp4_tuned_fmoe.csv}

PROFILE_ARGS=""
SPEC_ARGS=""
EXTRA_ARGS=""
for arg in "$@"; do
  case "$arg" in
    --profile)
      PROFILE_ARGS="--disable-cuda-graph"
      ;;
    --speculative|--spec)
      export SGLANG_ENABLE_SPEC_V2=1
      SPEC_ARGS="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --attention-backend triton"
      ;;
    *)
      # forward any other flag straight to `sglang serve`
      # (e.g. --enable-aiter-allreduce-fusion for A/B sweeps)
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
  --tp 4 \
  --host localhost \
  --port 8552 \
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
