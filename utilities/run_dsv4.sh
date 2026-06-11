#!/usr/bin/env bash
export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH}

# DP attention: set DP_MODE=tp8dp8 to enable (--dp 8 --enable-dp-attention). Default is off.
DP_MODE="${DP_MODE:-off}"
MODEL=${HF_HOME:-/root/hf_home}/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/89d501aed998d33fa4f4702102ec1bb2331e10f6
PORT="${PORT:-8000}"

# --- env (verbatim from run_sgl_dsv4.sh) ---
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=${SGLANG_USE_ROCM700A:-0}
export SGLANG_OPT_USE_FUSED_COMPRESS=true
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
# fp8 1x64 block-scale unified-KV attention pool (the point of this branch).
# Strictly opt-in via this var (unified_kv_use_fp8()); --kv-cache-dtype fp8_e4m3
# alone only fp8s the indexer pool. Default ON here; override with =0 for bf16.
export SGLANG_UNIFIED_KV_FP8="${SGLANG_UNIFIED_KV_FP8:-1}"
export SGLANG_OPT_FP8_WO_A_GEMM=false
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_OPT_USE_AITER_INDEXER=true
export SGLANG_OPT_USE_TILELANG_INDEXER=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
export SGLANG_ROCM_USE_MULTI_STREAM=false
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_EAGER_INPUT_NO_COPY=true

# for SGLANG vs ATOM, "unified_kv_triton" is the ATOM baseline. "triton" is the SGLANG baseline.
# for production vs B200, use the "unified_kv_triton" backend.

# Script-only flags (not passed to sglang): --profile, --triton, --unified
PROFILE_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --triton)
            export SGLANG_HACK_FLASHMLA_BACKEND=triton
            ;;
        --unified)
            export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
            ;;
        --profile)
            PROFILE_ARGS="--disable-cuda-graph"
            ;;
    esac
done

DP_ARGS=""
if [ "$DP_MODE" = "tp8dp8" ]; then
    DP_ARGS="--dp 8 --enable-dp-attention"
fi

SGL_EXTRA_ARGS="--enable-prefill-delayer --prefill-delayer-max-delay-ms 5000"

set -x
exec sglang serve \
    --model-path "${MODEL}" \
    --trust-remote-code \
    --tp 8 \
    ${DP_ARGS} \
    ${PROFILE_ARGS} \
    --disable-radix-cache \
    --attention-backend dsv4 \
    --page-size 256 \
    --mem-fraction-static 0.90 \
    --swa-full-tokens-ratio 0.1 \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 16384 \
    --cuda-graph-max-bs 512 \
    --max-running-requests 512 \
    --port "${PORT}" \
    ${SGL_EXTRA_ARGS:-}
