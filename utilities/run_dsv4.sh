export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH}
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_OPT_USE_FUSED_COMPRESS=true
# for SGLANG vs ATOM, "unified_kv_triton" is the ATOM baseline. "triton" is the SGLANG baseline.
# for production vs B200, use the "unified_kv_triton" backend.
# Default backend is "unified_kv_triton"
SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton

# Parse command line argument for backend selection
for arg in "$@"; do
    case "$arg" in
        --triton)
            SGLANG_HACK_FLASHMLA_BACKEND=triton
            ;;
        --unified)
            SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
            ;;
    esac
done

export SGLANG_HACK_FLASHMLA_BACKEND
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


# --dp 8 --enable-dp-attention \
# --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-eagle-topk 1
# --disable-cuda-graph \
MODEL=${HF_HOME:-/root/hf_home}/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/89d501aed998d33fa4f4702102ec1bb2331e10f6

sglang serve \
    --model-path ${MODEL} \
    --trust-remote-code \
    --tp 8 \
    $( [[ " $@ " =~ " --hc " ]] && echo "--dp 8 --enable-dp-attention" ) \
    --disable-radix-cache \
    --attention-backend dsv4 \
    --max-running-requests 256 \
    --page-size 256 \
    --mem-fraction-static 0.90 \
    --swa-full-tokens-ratio 0.1 \
    --chunked-prefill-size 8192 \
    --port 8000 \
    --disable-shared-experts-fusion \export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH}
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_OPT_USE_FUSED_COMPRESS=true
# for SGLANG vs ATOM, "unified_kv_triton" is the ATOM baseline. "triton" is the SGLANG baseline.
# for production vs B200, use the "unified_kv_triton" backend.
# Default backend is "unified_kv_triton"
SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton

# Parse command line argument for backend selection
for arg in "$@"; do
    case "$arg" in
        --triton)
            SGLANG_HACK_FLASHMLA_BACKEND=triton
            ;;
        --unified)
            SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
            ;;
    esac
done

export SGLANG_HACK_FLASHMLA_BACKEND
# This branch exercises the ATOM-ported wide decode kernel
export SGLANG_OPT_DSV4_DECODE_ATOM_PORT=1
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


# --dp 8 --enable-dp-attention \
# --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-eagle-topk 1
MODEL=${HF_HOME:-/root/hf_home}/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/89d501aed998d33fa4f4702102ec1bb2331e10f6

sglang serve \
    --model-path ${MODEL} \
    --trust-remote-code \
    --tp 8 \
    $( [[ " $@ " =~ " --hc " ]] && echo "--dp 8 --enable-dp-attention" ) \
    $( [[ " $@ " =~ " --profile " ]] && echo "--disable-cuda-graph" ) \
    --disable-radix-cache \
    --attention-backend dsv4 \
    --max-running-requests 256 \
    --page-size 256 \
    --mem-fraction-static 0.90 \
    --swa-full-tokens-ratio 0.1 \
    --chunked-prefill-size 8192 \
    --port 8000 \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4

    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4
