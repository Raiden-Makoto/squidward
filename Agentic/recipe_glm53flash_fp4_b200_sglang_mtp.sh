#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 AgentX recipe for B200 / SGLang with the checkpoint's
# built-in nextn (MTP) head.
#
# InferenceX has no GLM-5.3-Flash recipe: upstream's only GLM-5.3 entry is
# glm5.3-fp8-mi355x-tilert-agentic (the full model, ROCm, multi-node PD). This
# file is the B200 SGLang sibling, built from
#   benchmarks/single_node/agentic/glm5.2_fp4_b200_sglang_mtp.sh
# for the AgentX plumbing, and from the launch line of the local fixed-length
# TP4 MTP run under
#   results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130/
#     bench-Fixed-MTP-NVFP4-TP4/
# for every GLM-5.3-Flash-specific backend choice (DSA trtllm, flashinfer_trtllm
# MoE, fp8 KV, modelopt_fp4).
#
# Driven by ix_agentx_glm53flash.sh --platform b200, which supplies the CI env;
# it is not meant to be run standalone.
#
# Architecture facts that drive the settings below (config.json):
#   45 layers, hybrid: 11 DeepSeek-sparse-attention (MLA+DSA) layers at
#   [3,7,...,43] and 34 KDA linear-attention layers. kv_lora_rank 512,
#   qk_rope_head_dim 0, index_topk 2048, num_nextn_predict_layers 1,
#   max_position_embeddings 1048576.
# Two consequences the GLM-5.2 recipe does not have to deal with:
#   1. Only 11 layers hold per-token KV, so fp8 KV costs ~6.6 KB/token and TP4
#      fits ~12.4M tokens in HBM at --mem-fraction-static 0.85. That is why
#      HiCache is OFF by default here while GLM-5.2 needs a 169 GB/rank host
#      tier: the agentic working set is not what runs out first.
#   2. The 34 KDA layers carry recurrent state, so prefix caching needs the
#      hybrid-mamba radix path. SGLang resolves --mamba-radix-cache-strategy
#      auto to extra_buffer for Glm5NextForConditionalGeneration on the triton
#      linear-attn backend; it is pinned explicitly below so a backend change
#      cannot silently drop it to no_buffer (which needs page_size 1 and would
#      fail outright against DSA's forced page_size 64).

set -eo pipefail
set -x

source "${INFMAX_CONTAINER_WORKSPACE:?INFMAX_CONTAINER_WORKSPACE must point at the InferenceX checkout}/benchmarks/benchmark_lib.sh"

check_env_vars MODEL MODEL_PATH TP EP_SIZE CONC PORT RESULT_DIR DURATION
check_env_vars KV_OFFLOADING TOTAL_CPU_DRAM_GB DP_ATTENTION EVAL_ONLY
check_env_vars MEM_FRACTION_STATIC CHUNKED_PREFILL_SIZE SPEC_DECODING

[ -d "$MODEL_PATH" ] || { echo "Error: MODEL_PATH=$MODEL_PATH does not exist" >&2; exit 1; }

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

nvidia-smi

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
    # (865 GB for TP4 on a b200-nscale-shaped host at dram-utilization 0.80).
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
# the bonus token), matching the local fixed-length run at steps 5 / 6 tokens.
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

# Acceptance policy. InferenceX requires a spec-decode AgentX point to pin
# acceptance to the committed golden AL for the model, thinking mode and draft
# length, so throughput is not a function of how lucky the drafter got on this
# corpus. golden_al_distribution/glm5.3_mtp.yaml only carries glm-5.3-fp8 at
# K=3, and it is itself flagged PROVISIONAL (copied from GLM-5.2). There is no
# curve for GLM-5.3-*Flash* at any K, so ACC_MODE=real is the default here and
# these numbers are tuning numbers, not submittable ones. ACC_MODE=golden with
# GOLDEN_AL set reproduces the board's methodology once a curve exists.
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
# This knob has a second job that pulls the other way. The KDA intermediate
# state buffer is sized from max_running_requests * speculative_num_draft_tokens
# and comes straight out of the KV pool: 13.3 GB at conc 32, 26.6 GB at conc 64.
# The two only conflict once the KV pool is tight, which on this model happens
# at conc 64 and not before (conc 32 peaks at 21-24% pool usage with HiCache on).
#
# Measured at conc 64 with HiCache, 1.25x against the 2x default:
#   2.00x (128): 49,405 tok/s/GPU, TTFT p90 25.5s, intvty 25.3
#   1.25x  (80): 58,436 tok/s/GPU, TTFT p90 14.1s, intvty 23.7
# 18% more throughput and 45% better TTFT for 6% of interactivity. Note 1.25x
# is actively harmful without HiCache (ITL p90 202ms vs 148ms) -- a bigger KV
# pool does not help when the working set overflows it either way, and the lost
# scheduling headroom just adds queueing. It only pays once HiCache is catching
# the overflow.
#
# Per-concurrency like upstream's own hicache-size table in
# glm5.2_fp4_b200_sglang_mtp.sh, rather than one multiplier for the curve.
MAX_RUNNING_REQUESTS=$((2 * CONC))
case "$CONC" in
    64) MAX_RUNNING_REQUESTS=80 ;;
esac
[ -n "${MAX_RUNNING_REQUESTS_OVERRIDE:-}" ] && MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS_OVERRIDE"
[ "$MAX_RUNNING_REQUESTS" -lt 8 ] && MAX_RUNNING_REQUESTS=8

# --cuda-graph-max-bs-decode counts requests; the spec-decode graph runner
# scales by --speculative-num-draft-tokens itself. Prefill graphs stay off:
# SGLang disables them for KDA hybrid linear attention regardless.
#
# The cap matters from conc 32 up: max_running_requests is 2*CONC, so a cap of
# 64 leaves every decode batch above 64 requests on the eager path. Raising it
# costs capture time and HBM (each captured bs holds buffers for
# bs * speculative_num_draft_tokens tokens), which is why it is a knob and not
# just set to MAX_RUNNING_REQUESTS.
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
if [ "$CUDA_GRAPH_MAX_BS" -gt "${CUDA_GRAPH_MAX_BS_CAP:-64}" ]; then
    CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS_CAP:-64}"
fi

CONTEXT_ARGS=()
if [ -n "${CONTEXT_LENGTH:-}" ]; then
    CONTEXT_ARGS=(--context-length "$CONTEXT_LENGTH")
fi

# Hybrid memory split. Left unset, SGLang solves for the KDA state pool against
# --mamba-full-memory-ratio (fallback 0.9, i.e. ~47% of what is left after
# weights) and gives the rest to the 11 full-attention layers. That default was
# not tuned for a model this KV-light, so it is the first knob to A/B; see the
# max_total_num_tokens / max_mamba_cache_size lines the launcher greps out of
# server.log after each point.
MAMBA_ARGS=()
[ -n "${MAMBA_FULL_MEMORY_RATIO:-}" ] && MAMBA_ARGS+=(--mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO")
[ -n "${MAX_MAMBA_CACHE_SIZE:-}" ] && MAMBA_ARGS+=(--max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE")

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
# Agentic warmup dispatches hundreds of long prompts at once; allow up to 15
# minutes of TCP progress before AIPerf calls a connection dead, and keep
# uvicorn's keep-alive longer than an inter-turn idle gap (its 5s default lets
# a pooled socket close exactly as AIPerf reuses it -> ECONNRESET at warmup).
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --quantization modelopt_fp4
    # Blackwell DSA defaults to fp8 KV anyway; pinned so a default change cannot
    # silently double the per-token cost.
    --kv-cache-dtype fp8_e4m3
    --dsa-prefill-backend trtllm
    --dsa-decode-backend trtllm
    --moe-runner-backend flashinfer_trtllm
    # Prefix caching across turns is the whole point of the agentic scenario, so
    # unlike the fixed-length sweeps radix cache stays on, and the KDA half of
    # the model needs the ping-pong state donation path to be cacheable at all.
    --mamba-radix-cache-strategy extra_buffer
    # GLM-5.3 keeps GLM-4.7's tool-call format; glm45 leaves calls as raw text.
    --tool-call-parser glm47
    --reasoning-parser glm45
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    # One sequence per prefill batch. The DSA indexer sizes its fp8_mqa_logits
    # buffer as chunked_prefill_size x (sum of full context lengths in the
    # batch), so the buffer tracks how many long contexts happen to land in one
    # batch rather than anything bounded. At chunk 16384 that killed conc 32:
    # 31.4 GiB wanted against 31.1 GiB free, i.e. one batch carrying ~1.9M
    # tokens of context. Capping at one sequence bounds it by the longest
    # single context, ~17 GiB, and is what makes chunk 16384 -- and the 25%
    # throughput it buys over 8192 -- reachable at all.
    #
    # This is a trade, not a free win. Measured over a conc 32 run, 90% of
    # prefill batches already held a single sequence, so the cap is a no-op for
    # them; it binds on the ~10% that held 2-10, which are precisely the
    # batches that blow the buffer. Serialising those costs ITL p50 (6.88ms
    # against 5.89ms for the chunk-12288-no-cap alternative) and wins ITL p90
    # (24.83 vs 25.62) and TTFT p90 (8.10s vs 9.45s). p90 is the board axis, so
    # this side of the trade is the one worth taking.
    --prefill-max-requests 1
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

# Escape hatch for A/B-ing a flag without editing this file. Word-split on
# purpose: EXTRA_SERVER_ARGS='--prefill-max-requests 1 --foo bar'. Anything
# that earns its place here should graduate into the list above.
if [ -n "${EXTRA_SERVER_ARGS:-}" ]; then
    read -r -a _extra <<< "$EXTRA_SERVER_ARGS"
    SGLANG_CMD+=("${_extra[@]}")
fi

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

{
    echo "=== SGLANG_SIMULATE_ACC_* at launch (empty => real verification) ==="
    env | grep -E '^SGLANG_SIMULATE_ACC_' | sort || true
    echo "==================================================================="
} | tee "$SERVER_LOG"

echo "Starting SGLang server for B200 (GLM-5.3-Flash NVFP4, TP${TP}/EP${EP_SIZE}, conc=${CONC})..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    export SWEBENCH_AGENT_STEP_LIMIT=150
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
