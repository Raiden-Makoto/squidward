#!/usr/bin/env bash
# GLM-5.3-Flash Quark MXFP4 AgentX recipe for MI355X / SGLang with the
# checkpoint's built-in nextn (MTP) head.
#
# The MI355X sibling of recipe_glm53flash_fp4_b200_sglang_mtp.sh, kept as a
# separate file the way upstream keeps glm5.2_fp4_b200_sglang_mtp.sh and
# glm5.2_fp4_mi355x_sglang_mtp.sh apart. Structure, env contract and every
# agentic-specific choice (radix cache on, extra_buffer, 2xCONC running
# requests, metrics) are the B200 recipe's; the backend and selector flags
# come from the validated ROCm PR stack:
#   modelopt_fp4       -> Quark MXFP4, auto-detected from quantization_config
#   DSA trtllm         -> HIP default (Triton for #41615's validated geometry)
#   flashinfer_trtllm  -> aiter MoE
#   fp8_e4m3 KV        -> bfloat16 KV
# That launch disables radix cache because it is a fixed-length test; here it
# stays on, since cross-turn prefix reuse is what the agentic scenario measures.
#
# Driven by ix_agentx_glm53flash.sh --platform mi355x, which supplies the CI
# env; it is not meant to be run standalone.
#
# Architecture facts that drive the settings below (config.json):
#   45 layers, hybrid: 11 DeepSeek-sparse-attention (MLA+DSA) layers at
#   [3,7,...,43] and 34 KDA linear-attention layers. kv_lora_rank 512,
#   qk_rope_head_dim 0, index_topk 2048, num_nextn_predict_layers 1,
#   max_position_embeddings 1048576.
# Only 11 layers hold per-token KV, so even at bf16 the KV pool is not what an
# agentic working set runs out of on a 288 GB part, and HiCache is OFF by
# default. The 34 KDA layers carry recurrent state, so prefix caching needs the
# hybrid-mamba radix path: extra_buffer requires linear_attn_backend=triton,
# which is SGLang's default on ROCm. The strategy is pinned below so a backend
# change fails extra_buffer's validation at startup instead of auto-resolving
# to no_buffer (page_size 1, which DSA's forced page_size 64 rejects).

set -eo pipefail
set -x

source "${INFMAX_CONTAINER_WORKSPACE:?INFMAX_CONTAINER_WORKSPACE must point at the InferenceX checkout}/benchmarks/benchmark_lib.sh"

check_env_vars MODEL MODEL_PATH TP EP_SIZE CONC PORT RESULT_DIR DURATION
check_env_vars KV_OFFLOADING TOTAL_CPU_DRAM_GB DP_ATTENTION EVAL_ONLY
check_env_vars MEM_FRACTION_STATIC CHUNKED_PREFILL_SIZE SPEC_DECODING

[ -d "$MODEL_PATH" ] || { echo "Error: MODEL_PATH=$MODEL_PATH does not exist" >&2; exit 1; }

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

rocm-smi || true

resolve_trace_source
install_agentic_deps

# ---- HiCache host tier (opt-in) -----------------------------------------
# HICACHE_SIZE is an absolute per-rank GB figure, the same units the GLM-5.2
# MI355X lane settled on after the ratio form proved unstable across boots.
# Unset/0 leaves the whole cache device-resident.
CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    : "${HICACHE_SIZE:=0}"
    if ! [[ "$HICACHE_SIZE" =~ ^[0-9]+$ ]]; then
        echo "Error: HICACHE_SIZE must be a non-negative integer, got $HICACHE_SIZE" >&2
        exit 1
    fi
    # TOTAL_CPU_DRAM_GB is the aggregate budget for all TP ranks on the node
    # (1199 GB for TP4 on mi355x-amds at dram-utilization 0.80).
    MAX_HICACHE_SIZE=$((TOTAL_CPU_DRAM_GB / TP))
    if [ "$HICACHE_SIZE" -gt "$MAX_HICACHE_SIZE" ]; then
        echo "Error: HICACHE_SIZE=$HICACHE_SIZE GB/rank exceeds the ${TOTAL_CPU_DRAM_GB} GB node budget split over TP=$TP (${MAX_HICACHE_SIZE} GB/rank)" >&2
        exit 1
    fi
    if [ "$HICACHE_SIZE" -le 0 ]; then
        echo "Error: KV_OFFLOADING=dram was requested but HICACHE_SIZE is 0" >&2
        exit 1
    fi
    echo "HiCache CPU tier: ${HICACHE_SIZE} GB/rank, node budget ${TOTAL_CPU_DRAM_GB} GB, write_back / direct / page_first_direct"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-write-policy write_back
        --hicache-io-backend direct
        --hicache-mem-layout page_first_direct
        --hicache-size "$HICACHE_SIZE"
    )
fi

# ---- speculative decoding ------------------------------------------------
# GLM-5.3-Flash ships one nextn layer, so EAGLE drafts off the checkpoint with
# no separate draft model. num-draft-tokens is steps+1 (the verify step covers
# the bonus token).
SPEC_ARGS=()
if [ "$SPEC_DECODING" = "mtp" ]; then
    check_env_vars SPEC_NUM_STEPS SPEC_NUM_DRAFT_TOKENS
    SPEC_ARGS=(
        --speculative-algorithm EAGLE
        --speculative-num-steps "$SPEC_NUM_STEPS"
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens "$SPEC_NUM_DRAFT_TOKENS"
    )
fi

# Acceptance policy: see the B200 recipe. There is no golden AL curve for
# GLM-5.3-Flash at any K, so ACC_MODE=real is the default and these are tuning
# numbers, not submittable ones.
if [ "${EVAL_ONLY}" != "true" ] && [ "${ACC_MODE:-real}" = "golden" ]; then
    check_env_vars GOLDEN_AL
    export SGLANG_SIMULATE_ACC_LEN="$GOLDEN_AL"
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

# ---- parallelism and scheduling -----------------------------------------
PARALLEL_ARGS=(--tp "$TP" --ep-size "$EP_SIZE")
if [ "$DP_ATTENTION" = "true" ]; then
    echo "Error: DP attention is not wired for this recipe; GLM-5.3-Flash at TP4 is the low-latency arm." >&2
    exit 1
fi

# AgentX concurrency counts live session trees, not requests: a trajectory can
# fan out to subagents, so the engine must accept more than CONC in flight or
# the client's bursts queue behind an artificial cap.
#
# At conc 64 the device KV pool is the wall (100% full, no HiCache). Cutting
# max-running from 128 to 80 frees the KDA intermediate buffer for KV and,
# together with HiCache 169 and mamba-ratio 0.5, is the measured winner:
# interactivity 2.7 -> 20.6 and 12,349 -> 55,272 tok/s/GPU on a 1200s window.
# Conc 32 peaks at 49% of the pool with 2x and no HiCache, so it stays at 2x.
MAX_RUNNING_REQUESTS=$((2 * CONC))
case "$CONC" in
    64) MAX_RUNNING_REQUESTS=80 ;;
esac
[ -n "${MAX_RUNNING_REQUESTS_OVERRIDE:-}" ] && MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS_OVERRIDE"
[ "$MAX_RUNNING_REQUESTS" -lt 8 ] && MAX_RUNNING_REQUESTS=8

# --cuda-graph-max-bs-decode counts requests; the spec-decode graph runner
# scales by --speculative-num-draft-tokens itself. Prefill graphs stay off:
# SGLang disables them for KDA hybrid linear attention regardless. The cap is a
# knob (CUDA_GRAPH_MAX_BS_CAP) for the same reason as on B200: above conc 32 a
# cap of 64 leaves larger decode batches eager.
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
if [ "$CUDA_GRAPH_MAX_BS" -gt "${CUDA_GRAPH_MAX_BS_CAP:-64}" ]; then
    CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS_CAP:-64}"
fi

CONTEXT_ARGS=()
if [ -n "${CONTEXT_LENGTH:-}" ]; then
    CONTEXT_ARGS=(--context-length "$CONTEXT_LENGTH")
fi

# Hybrid memory split; see the B200 recipe. Left unset SGLang solves for the
# KDA state pool itself, and the driver greps the resulting pool sizes out of
# server.log after each point.
MAMBA_ARGS=()
[ -n "${MAMBA_FULL_MEMORY_RATIO:-}" ] && MAMBA_ARGS+=(--mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO")
[ -n "${MAX_MAMBA_CACHE_SIZE:-}" ] && MAMBA_ARGS+=(--max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE")

export PYTHONNOUSERSITE=1
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
# RM/glm53 contains #38764. Enable the validated gfx950 PTPC selector for the
# BF16 KDA projections that remain unquantized in the Quark checkpoint. An
# explicitly set empty value is retained for selector-off A/B runs.
export SGLANG_OPT_GLM53_KDA_PTPC_MODULES="${SGLANG_OPT_GLM53_KDA_PTPC_MODULES-qkv_proj,f_a_proj,g_a_proj,o_proj}"
# Keep the communication tuple aligned with the validated TP4 GLM-5.3 launch.
export ROCM_QUICK_REDUCE_QUANTIZATION="${ROCM_QUICK_REDUCE_QUANTIZATION-INT4}"
export AITER_QUICK_REDUCE_QUANTIZATION="${AITER_QUICK_REDUCE_QUANTIZATION-INT4}"
# Agentic warmup dispatches hundreds of long prompts at once; allow up to 15
# minutes of TCP progress before AIPerf calls a connection dead, and keep
# uvicorn's keep-alive longer than an inter-turn idle gap (its 5s default lets
# a pooled socket close exactly as AIPerf reuses it -> ECONNRESET at warmup).
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900
# The DSA indexer's fp32 MQA-logits scratch sits outside every pool that
# --mem-fraction-static sizes. On ROCm it is budgeted as this fraction of free
# HBM (and capped at aiter's 2 GiB); 0.04 is what the GLM-5.2 MI355X lane runs.
export SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION="${SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION:-0.04}"

# aiter's JIT baton records the builder's pid in the lock file and only breaks
# the lock when that pid is gone. A server killed mid-build leaves the lock
# behind, and every rank of the next boot blocks in file_baton.wait() with the
# GPUs idle. Nothing is building before launch, so any surviving lock is stale.
find "${AITER_JIT_DIR:-/sgl-workspace/aiter/aiter/jit}/build" -maxdepth 1 -name 'lock_*' -delete 2>/dev/null || true

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    # No --quantization: Quark MXFP4 (with its block-FP8 attention and MTP
    # exceptions) is resolved from the checkpoint's quantization_config.
    --kv-cache-dtype bfloat16
    --attention-backend dsa
    # Do not pin the DSA sub-backends. #41615 teaches the existing HIP default
    # to select Triton for the validated GLM-5.3 gfx950 geometry; explicitly
    # requesting TileLang must remain a true TileLang baseline.
    --moe-runner-backend aiter
    # #41870 lets the target model auto-enable fusion after validating the
    # Quark layouts. Do not pass --enforce-shared-experts-fusion: until #41258
    # lands, that would bypass the MTP draft's architecture guard.
    # Prefix caching across turns is the whole point of the agentic scenario, so
    # unlike the fixed-length sweeps radix cache stays on, and the KDA half of
    # the model needs the ping-pong state donation path to be cacheable at all.
    --linear-attn-backend triton
    --mamba-radix-cache-strategy extra_buffer
    # GLM-5.3 keeps GLM-4.7's tool-call format; glm45 leaves calls as raw text.
    --tool-call-parser glm47
    --reasoning-parser glm45
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}'
    "${CONTEXT_ARGS[@]}"
    "${MAMBA_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${CACHE_ARGS[@]}"
    --watchdog-timeout 1800
    --enable-metrics
)

# Escape hatch for A/B-ing a flag without editing this file; see the B200 recipe.
if [ -n "${EXTRA_SERVER_ARGS:-}" ]; then
    read -r -a _extra <<< "$EXTRA_SERVER_ARGS"
    SGLANG_CMD+=("${_extra[@]}")
fi

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

{
    echo "=== SGLANG_SIMULATE_ACC_* at launch (empty => real verification) ==="
    env | grep -E '^SGLANG_SIMULATE_ACC_' | sort || true
    echo "=== RM/glm53 optimization selectors ==="
    echo "SGLANG_USE_AITER=$SGLANG_USE_AITER"
    echo "SGLANG_OPT_GLM53_KDA_PTPC_MODULES=$SGLANG_OPT_GLM53_KDA_PTPC_MODULES"
    echo "ROCM_QUICK_REDUCE_QUANTIZATION=$ROCM_QUICK_REDUCE_QUANTIZATION"
    echo "AITER_QUICK_REDUCE_QUANTIZATION=$AITER_QUICK_REDUCE_QUANTIZATION"
    echo "==================================================================="
} | tee "$SERVER_LOG"

echo "Starting SGLang server for MI355X (GLM-5.3-Flash MXFP4, TP${TP}/EP${EP_SIZE}, conc=${CONC})..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if ! grep -Fq "prefill=triton, decode=triton" "$SERVER_LOG"; then
    echo "Error: #41615 unified Triton DSA did not enable; refusing to benchmark a mismatched tuple." >&2
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

if ! grep -Fq "Shared experts fusion optimization enabled." "$SERVER_LOG"; then
    echo "Error: #41870 shared-expert fusion did not enable; refusing to benchmark a mismatched tuple." >&2
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    export SWEBENCH_AGENT_STEP_LIMIT=150
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
