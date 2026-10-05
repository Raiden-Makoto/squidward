#!/usr/bin/env bash
# InferenceX AgentX sweep for GLM-5.3-Flash FP4 on B200 (NVFP4) or MI355X
# (Quark MXFP4) / SGLang, TP4.
#
# Runs recipe_glm53flash_fp4_<platform>_sglang_mtp.sh once per concurrency with
# the environment the run-sweep workflow would inject, so the client, the
# corpus, the result schema and the failure gates are InferenceX's rather than
# ours. Both platforms go through this one driver so that everything except the
# recipe's server flags is the same code; that is what makes a B200 vs MI355X
# curve a chip comparison. The recipes are local: upstream has no
# GLM-5.3-Flash entry, so each is modelled on its glm5.2_fp4_<platform>_sglang_mtp.
#
# Run this INSIDE the container (B200: lmsysorg/sglang:v0.5.20-cu130, MI355X:
# rocm/sgl-dev:v0.5.21-rocm724-mi35x-20261001). --platform defaults to mi355x
# when rocm-smi is on PATH, else b200. GPU selection comes from
# CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES or --gpus.
#
#   ./ix_agentx_glm53flash.sh                      # TP4, conc 1 4 8 16 32 64
#   ./ix_agentx_glm53flash.sh --smoke --conc 4     # plumbing check, ~20 min
#   ./ix_agentx_glm53flash.sh --quick --conc 16    # A/B iteration, ~30 min
#   ./ix_agentx_glm53flash.sh --mtp-steps 3 --conc 16    # MTP depth A/B
#   ./ix_agentx_glm53flash.sh --hicache-size 200         # add a host tier
#   ./ix_agentx_glm53flash.sh --platform b200 --mem-fraction 0.75 --chunked-prefill 8192
#   ./ix_agentx_glm53flash.sh --dry-run            # print the env and exit
set -uo pipefail

usage() { sed -n '2,24p' "$0"; exit 1; }

# Tear down only the server this sweep started. A bare
# `pkill -f sglang.launch_server` also kills a second sweep running on the
# other four GPUs, which is how parallel A/B screening used to be impossible.
# The launcher carries --port on its command line; its workers get renamed to
# sglang::* and lose it, so go via the process group instead.
# Depth-first so children die before their parent can reap or re-fork.
kill_tree() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child"; done
    kill -9 "$pid" 2>/dev/null
    return 0
}

kill_server_on_port() {
    local port="$1" pid pgid mypgid
    # The recipe runs as our child, so the server it launches lands in OUR
    # process group. Group-killing it therefore kills this driver too, which
    # ends the sweep after the first concurrency with no error of its own --
    # the log just stops. Only group-kill when the group is demonstrably not
    # ours; otherwise walk the process tree.
    mypgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
    for pid in $(pgrep -f "launch_server.*--port[ =]$port" 2>/dev/null) \
               $(port_pids "$port"); do
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ -n "$pgid" ] && [ -n "$mypgid" ] && [ "$pgid" != "$mypgid" ]; then
            kill -9 -- "-$pgid" 2>/dev/null
        else
            kill_tree "$pid"
        fi
    done
    kill_port "$port"
    return 0
}

kill_port() {
    local pids
    pids=$(port_pids "$1")
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null
    return 0
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_HOME="${BENCH_HOME:-$(dirname "$HERE")}"
IX="${IX:-${INFMAX_CONTAINER_WORKSPACE:-/home/macui/InferenceX/inferencex-e2e}}"

PLATFORM=""
DOCKER="${DOCKER:-}"
CKPT="${CKPT:-}"

TAG=""
TP=4
EP=1
CONC_LIST="1 4 8 16 32 64"
DURATION="${DURATION:-3600}"
GPUS=""
PORT_BASE=28900
# Empty so the platform block below can default them per chip without
# clobbering a value the caller passed on the command line.
MEM_FRACTION_STATIC=""
CHUNKED_PREFILL_SIZE=""
# steps 5 / 6 draft tokens is what the local fixed-length TP4 MTP run used and
# is the deepest EAGLE ladder GLM-5.3-Flash's single nextn head sustains here.
MTP_STEPS=5
SPEC_DECODING=mtp
ACC_MODE=real
GOLDEN_AL=""
HICACHE_SIZE=""
CONTEXT_LENGTH=""
MAMBA_FULL_MEMORY_RATIO=""
MAX_MAMBA_CACHE_SIZE=""
CUDA_GRAPH_MAX_BS_CAP=""
EXTRA_SERVER_ARGS=""
ENABLE_POWER=0
QUICK=0
SMOKE=0
FAST=0
DRY_RUN=0
declare -a EXTRA_ENV=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)   PLATFORM="$2"; shift 2 ;;
        --tag)        TAG="$2"; shift 2 ;;
        --docker)     DOCKER="$2"; shift 2 ;;
        --ckpt)       CKPT="$2"; shift 2 ;;
        --tp)         TP="$2"; shift 2 ;;
        --ep)         EP="$2"; shift 2 ;;
        --conc)       CONC_LIST="$2"; shift 2 ;;
        --duration)   DURATION="$2"; shift 2 ;;
        --gpus)       GPUS="$2"; shift 2 ;;
        --port)       PORT_BASE="$2"; shift 2 ;;
        --mem-fraction)      MEM_FRACTION_STATIC="$2"; shift 2 ;;
        --chunked-prefill)   CHUNKED_PREFILL_SIZE="$2"; shift 2 ;;
        --mtp-steps)  MTP_STEPS="$2"; shift 2 ;;
        --no-mtp)     SPEC_DECODING=none; shift ;;
        --acc)        ACC_MODE="$2"; shift 2 ;;
        --golden-al)  GOLDEN_AL="$2"; ACC_MODE=golden; shift 2 ;;
        --hicache-size)      HICACHE_SIZE="$2"; shift 2 ;;
        --context-length)    CONTEXT_LENGTH="$2"; shift 2 ;;
        --mamba-ratio)       MAMBA_FULL_MEMORY_RATIO="$2"; shift 2 ;;
        --max-mamba-cache-size) MAX_MAMBA_CACHE_SIZE="$2"; shift 2 ;;
        --cuda-graph-max-bs)    CUDA_GRAPH_MAX_BS_CAP="$2"; shift 2 ;;
        --server-arg)           EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS} $2"; shift 2 ;;
        --power)      ENABLE_POWER=1; shift ;;
        --quick)      QUICK=1; shift ;;
        --smoke)      SMOKE=1; shift ;;
        --fast)       FAST=1; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --env)        EXTRA_ENV+=("$2"); shift 2 ;;
        -h|--help)    usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done

# What the caller explicitly passed. Everything below may fill a default, and
# the b200 table below is per-concurrency, so this is the only record of intent.
USER_MEM_FRACTION="$MEM_FRACTION_STATIC"
USER_CHUNK="$CHUNKED_PREFILL_SIZE"
USER_MAMBA_RATIO="$MAMBA_FULL_MEMORY_RATIO"
USER_HICACHE="$HICACHE_SIZE"

# ---- platform ------------------------------------------------------------
# Everything that differs between the two chips lives in this block and in the
# recipe. The rest of the driver must stay platform-neutral.
if [ -z "$PLATFORM" ]; then
    if command -v rocm-smi >/dev/null 2>&1; then PLATFORM=mi355x; else PLATFORM=b200; fi
fi
case "$PLATFORM" in
    b200)
        SMI=nvidia-smi
        VISIBLE_VAR=CUDA_VISIBLE_DEVICES
        RUNNER_TYPE=b200
        DRAM_RUNNER=cluster:b200-nscale
        PORT_TOOL=lsof
        : "${DOCKER:=lmsysorg/sglang:v0.5.20-cu130}"
        MODEL_ID="nvidia/GLM-5.3-Flash-NVFP4"
        : "${CKPT:=/data/huggingface/hub/nvidia/GLM-5.3-Flash-NVFP4}"
        # Measured defaults, not guesses. Each was A/B'd at conc 32 against the
        # value above it; see docs in AgentSkill sglang-benchmark/agentic-benchmark.md.
        #   mem-fraction 0.75   headroom for the DSA indexer's per-prefill buffer
        #   chunked-prefill 16k the real high-concurrency bottleneck is prefill
        #                       budget, not memory; 8k costs 25% throughput
        #   mamba-ratio 1.5     inverted-U peak: 0.9 starves KDA slots, 2.0
        #                       starts eating KV the model still needs
        #   hicache 169 GB/rank even at conc 32, where the device pool is only
        #                       21-24% used, eviction is still happening and the
        #                       host tier catches it (+29% interactivity)
        # Measured per concurrency, because the winner changes. Up to conc 16
        # the tuned settings are a net loss -- the KV pool runs at 2-9%, so
        # nothing they buy is reachable, while their costs are not conditional:
        #   conc   1      4      8     16     (P90 interactivity, tuned vs plain)
        #        -21%   -19%   -20%   -11%
        # From conc 32 the same settings win by a wide margin (+29% at conc 32)
        # because the pool is finally under pressure. Per-concurrency tables
        # are how upstream's glm5.2_fp4_b200_sglang_mtp.sh handles the same
        # situation with hicache-size.
        conc_defaults() {
            local c="$1"
            MEM_FRACTION_STATIC="${USER_MEM_FRACTION:-0.75}"
            if [ "$c" -ge 32 ]; then
                CHUNKED_PREFILL_SIZE="${USER_CHUNK:-16384}"
                MAMBA_FULL_MEMORY_RATIO="${USER_MAMBA_RATIO:-1.5}"
                HICACHE_SIZE="${USER_HICACHE:-169}"
            else
                CHUNKED_PREFILL_SIZE="${USER_CHUNK:-8192}"
                MAMBA_FULL_MEMORY_RATIO="$USER_MAMBA_RATIO"
                HICACHE_SIZE="${USER_HICACHE:-0}"
            fi
        }
        gpu_count() { nvidia-smi --query-gpu=index --format=csv,noheader | wc -l; }
        gpu_mem_used_max_mib() {
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits ${GPUS:+-i "$GPUS"} | sort -rn | head -1
        }
        # The sglang cu13 image ships lsof but not psmisc, so fuser is not available.
        port_pids() { lsof -t -i ":$1" -sTCP:LISTEN 2>/dev/null; }
        ;;
    mi355x)
        SMI=rocm-smi
        VISIBLE_VAR=HIP_VISIBLE_DEVICES
        RUNNER_TYPE=mi355x
        DRAM_RUNNER=cluster:mi355x-amds
        PORT_TOOL=fuser
        : "${DOCKER:=rocm/sgl-dev:v0.5.21-rocm724-mi35x-20261001}"
        MODEL_ID="amd/GLM-5.3-Flash-Quark-MXFP4"
        : "${CKPT:=/data2/hf_home/hub/models--amd--GLM-5.3-Flash-Quark-MXFP4/snapshots/b5688f25491202978c19c4d036eef579f61bbe07}"
        # rocm-smi, like nvidia-smi, lists every device on the node regardless
        # of the visibility variable: the count sees the whole node, and the
        # reclaim check filters to --gpus itself.
        # Measured per concurrency, and not B200's table. Conc 32 without
        # HiCache: P90 interactivity 55.9, 45,964 tok/s/GPU; HiCache 169 on
        # that point fell to 51.4. Conc 64 without a host tier collapsed
        # (2.7, 12,349 tok/s/GPU, KV pool at 100%); HiCache 169 + mamba-ratio
        # 0.5 recovered 20.6 and 55,272. max-running 80 at conc 64 lives in
        # the recipe. An explicit --hicache-size / --mamba-ratio still wins
        # at every point.
        conc_defaults() {
            local c="$1"
            MEM_FRACTION_STATIC="${USER_MEM_FRACTION:-0.85}"
            CHUNKED_PREFILL_SIZE="${USER_CHUNK:-16384}"
            if [ "$c" -ge 64 ]; then
                MAMBA_FULL_MEMORY_RATIO="${USER_MAMBA_RATIO:-0.5}"
                HICACHE_SIZE="${USER_HICACHE:-169}"
            else
                MAMBA_FULL_MEMORY_RATIO="${USER_MAMBA_RATIO:-0.9}"
                HICACHE_SIZE="${USER_HICACHE:-0}"
            fi
        }
        gpu_count() { rocm-smi --showid --csv 2>/dev/null | grep -c '^card'; }
        gpu_mem_used_max_mib() {
            rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, -v want="$GPUS" '
                /^card/ {
                    idx = substr($1, 5) + 0
                    if (want != "") { keep = 0; n = split(want, w, ","); for (i = 1; i <= n; i++) if (w[i] + 0 == idx) keep = 1; if (!keep) next }
                    if ($3 > m) m = $3
                } END { printf "%d\n", m / 1048576 }'
        }
        port_pids() { fuser "$1/tcp" 2>/dev/null; }
        ;;
    *) echo "ERROR: --platform must be 'b200' or 'mi355x', got '$PLATFORM'" >&2; exit 1 ;;
esac
RECIPE="$HERE/recipe_glm53flash_fp4_${PLATFORM}_sglang_mtp.sh"

# ---- preflight -----------------------------------------------------------
for tool in "$SMI" "$PORT_TOOL"; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "ERROR: --platform $PLATFORM needs '$tool' on PATH." >&2
        [ "$tool" = fuser ] && echo "       It is in psmisc: apt-get install -y psmisc" >&2
        exit 1
    }
done
[ -f "$RECIPE" ] || { echo "ERROR: recipe not found at $RECIPE" >&2; exit 1; }
[ -d "$CKPT" ]   || { echo "ERROR: checkpoint not found at $CKPT" >&2; exit 1; }
# benchmark_lib.sh's agentic half and the infx result package both moved in the
# 2026-09 tree; an older checkout fails deep inside install_agentic_deps.
for required in "$IX/benchmarks/benchmark_lib.sh" "$IX/benchmarks/runtime_settings.sh" \
                "$IX/utils/aiperf/pyproject.toml" \
                "$IX/infx/results/agentic/process_agentic_result.py"; do
    [ -e "$required" ] || {
        echo "ERROR: $required is missing. Update the InferenceX checkout at $IX" >&2
        echo "       (git pull && git submodule update --init utils/aiperf)." >&2
        exit 1
    }
done

[ -n "$GPUS" ] && export "$VISIBLE_VAR=$GPUS"
NGPU=$(gpu_count)
if [ "$NGPU" -lt "$TP" ]; then
    echo "ERROR: TP=$TP needs $TP GPUs but only $NGPU are visible ($VISIBLE_VAR=${!VISIBLE_VAR:-all})." >&2
    exit 1
fi

case "$ACC_MODE" in
    real) ;;
    golden)
        [ -n "$GOLDEN_AL" ] || {
            echo "ERROR: --acc golden needs --golden-al <AL>. golden_al_distribution/glm5.3_mtp.yaml" >&2
            echo "       carries only glm-5.3-fp8 at K=3 (2.99, itself marked PROVISIONAL and copied" >&2
            echo "       from GLM-5.2); there is no measured curve for GLM-5.3-Flash at any depth." >&2
            exit 1
        } ;;
    *) echo "ERROR: --acc must be 'real' or 'golden', got '$ACC_MODE'" >&2; exit 1 ;;
esac

# ---- InferenceX CI environment ------------------------------------------
# runtime_settings.sh owns the AIPERF_*/AGENTIC_* defaults that build_replay_cmd
# hard-requires (it uses check_env_vars, so an unset one aborts rather than
# falling back). Load it first, then override only what this lane changes.
# shellcheck source=/dev/null
source "$IX/benchmarks/runtime_settings.sh"

export INFMAX_CONTAINER_WORKSPACE="$IX"
export MODEL="$MODEL_ID"
export MODEL_PATH="$CKPT"
# GLM-5.3-Flash is not GLM-5.3, and upstream gives Flash variants their own
# prefix (DeepSeek-V4-Pro is dsv4, DeepSeek-V4.1-Flash is dsv41flash). The dot
# has to stay: resolve_trace_source globs glm5.3*, so glm5.3flash still selects
# the unfiltered 1M-context corpus, where a dotless glm53flash would fall
# through to the 256k variant without saying so. It also makes golden_length()
# look for glm5.3flash_mtp.yaml and fail loudly instead of quietly borrowing
# GLM-5.3's provisional curve.
export MODEL_PREFIX=glm5.3flash
export PRECISION=fp4
export FRAMEWORK=sglang
export RUNNER_TYPE
export IMAGE="$DOCKER"
export SCENARIO_TYPE=agentic-coding
export THINKING_MODE=thinking_on
export SPEC_DECODING
export DISAGG=false
export IS_MULTINODE=false
export EVAL_ONLY=false
export TP EP_SIZE="$EP" PP_SIZE=1 DCP_SIZE=1 PCP_SIZE=1
export DP_ATTENTION=false
export REQUIRE_POWER=0
export ENABLE_AGENTX_POWER="$ENABLE_POWER"
# The workflow supplies this one, not runtime_settings.sh, and build_replay_cmd
# check_env_vars-aborts on an unset (as opposed to '0') value.
export AIPERF_EXPERIMENTAL_FAST=0
export GPU_MONITOR_INTERVAL=1
export MEM_FRACTION_STATIC CHUNKED_PREFILL_SIZE
export SPEC_NUM_STEPS="$MTP_STEPS" SPEC_NUM_DRAFT_TOKENS=$((MTP_STEPS + 1))
export ACC_MODE
[ -n "$GOLDEN_AL" ] && export GOLDEN_AL
[ -n "$MAMBA_FULL_MEMORY_RATIO" ] && export MAMBA_FULL_MEMORY_RATIO
[ -n "$MAX_MAMBA_CACHE_SIZE" ] && export MAX_MAMBA_CACHE_SIZE
[ -n "$CUDA_GRAPH_MAX_BS_CAP" ] && export CUDA_GRAPH_MAX_BS_CAP
[ -n "$EXTRA_SERVER_ARGS" ] && export EXTRA_SERVER_ARGS

# The client replays the corpus unfiltered when MAX_MODEL_LEN=0. Capping the
# server without capping the client turns the over-length traces into 4xxs that
# still occupy a lane, so the two move together.
if [ -n "$CONTEXT_LENGTH" ]; then
    export CONTEXT_LENGTH MAX_MODEL_LEN="$CONTEXT_LENGTH"
else
    export MAX_MODEL_LEN=0          # native 1,048,576
fi

# KV tier, resolved per concurrency because conc_defaults sets HICACHE_SIZE
# from a per-conc table. The threshold differs by chip: B200 turns the host
# tier on at conc 32, MI355X only at conc 64.
# A lower conc sets TOTAL_CPU_DRAM_GB=0, so the node budget is cached the
# first time a point actually asks for HiCache. Otherwise the next point
# would treat that 0 as an already-resolved budget.
DRAM_BUDGET_GB=""
resolve_kv_tier() {
    # KV tier. GLM-5.3-Flash only has 11 full-attention layers, so TP4 holds ~12.4M
    # tokens of fp8 KV in HBM and a host tier is opt-in rather than load-bearing.
    if [ "${HICACHE_SIZE:-0}" -gt 0 ]; then
        export KV_OFFLOADING=dram
        export KV_OFFLOAD_BACKEND=hicache
        export KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}'
        export HICACHE_SIZE
        # Ask InferenceX's own matrix logic for the node budget instead of
        # transcribing a number. At TP4 and dram-utilization 0.80 that is 865 GB on
        # b200-nscale and 1199 GB on mi355x-amds, so a HiCache A/B across the two
        # chips is not a like-for-like host tier.
        if [ -z "$DRAM_BUDGET_GB" ]; then
            if [ "${TOTAL_CPU_DRAM_GB:-0}" -gt 0 ]; then
                DRAM_BUDGET_GB="$TOTAL_CPU_DRAM_GB"
            else
                DRAM_BUDGET_GB="$(cd "$IX" && python3 -c "
import yaml
from infx.matrix.generate import agentic_dram_offload_gb
print(agentic_dram_offload_gb(
    {'dram-utilization': 0.8},
    {'tp': $TP, 'ep': $EP, 'kv-offloading': 'dram'},
    '$DRAM_RUNNER',
    yaml.safe_load(open('configs/runners.yaml')),
))")"
            fi
        fi
        TOTAL_CPU_DRAM_GB="$DRAM_BUDGET_GB"
    else
        export KV_OFFLOADING=none
        # process_agentic_result rejects a result that names a backend it did not use.
        export KV_OFFLOAD_BACKEND=""
        export KV_OFFLOAD_BACKEND_METADATA=""
        TOTAL_CPU_DRAM_GB=0
    fi
    export TOTAL_CPU_DRAM_GB
}

# Keep the evaluator venv and Hugging Face cache on the shared /data2 volume
# mounted by the GLM-5.3 containers, not inside the disposable container layer.
#
# Keyed by port base so two sweeps sharing the node get separate venvs.
# install_agentic_deps starts with `rm -rf $AIPERF_VENV`, so a shared runtime
# dir means the second sweep to start deletes the interpreter the first one is
# running out of -- which surfaces as an unrelated-looking ModuleNotFoundError
# inside huggingface_hub, seconds into the run. Same port reuses its warm venv.
export AIPERF_RUNTIME_DIR="${AIPERF_RUNTIME_DIR:-/data2/hf_home/ix-agentic-runtime-${PORT_BASE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data2/hf_home/hub}"
mkdir -p "$AIPERF_RUNTIME_DIR" "$HF_HUB_CACHE"

for kv in "${EXTRA_ENV[@]:-}"; do [ -n "$kv" ] && export "${kv?}"; done

# ---- run-length modes ----------------------------------------------------
[ -n "$TAG" ] || TAG="TP${TP}_EP${EP}-mtp${MTP_STEPS}"
[ "$SPEC_DECODING" = "none" ] && TAG="TP${TP}_EP${EP}-nospec"
if [ "$FAST" = "1" ]; then
    # 1 warmup request per lane leaves the prefix caches nearly empty, which is
    # exactly what an agentic run is supposed to exercise. Useful only to prove
    # the pipe works end to end.
    export AIPERF_EXPERIMENTAL_FAST=1
    TAG="${TAG}-fast"
fi
if [ "$QUICK" = "1" ]; then
    # Half warmup, 10-minute window: conc 16 lands in ~30 min instead of ~100.
    # Under 900s the scenario marks submission_valid=false, so compare these
    # only against other --quick runs.
    export AIPERF_WARMUP_REQUESTS_PER_LANE=5
    export AIPERF_UNSAFE_OVERRIDE=true
    DURATION=600
    TAG="${TAG}-quick"
fi
if [ "$SMOKE" = "1" ]; then
    export AIPERF_WARMUP_REQUESTS_PER_LANE=1
    export AIPERF_UNSAFE_OVERRIDE=true
    DURATION=300
    TAG="${TAG}-smoke"
fi
export DURATION

# ---- result layout (GLM.sh's convention) ---------------------------------
MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")_$(basename "$MODEL_PATH")"
DOCKER_FILENAME="$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')"
ROOT="$BENCH_HOME/results/$MODEL_NAME/$DOCKER_FILENAME/bench-Agentic-$TAG"
mkdir -p "$ROOT"
SWEEP_LOG="$ROOT/sweep.log"

{
    echo "=== $(date -Is) GLM-5.3-Flash AgentX platform=$PLATFORM model=$MODEL_ID ==="
    echo "tag=$TAG tp=$TP ep=$EP spec=$SPEC_DECODING steps=$MTP_STEPS acc=$ACC_MODE${GOLDEN_AL:+($GOLDEN_AL)}"
    echo "mem_fraction=${USER_MEM_FRACTION:-per-conc} chunked_prefill=${USER_CHUNK:-per-conc} context=${CONTEXT_LENGTH:-native-1M}"
    echo "hicache_size=${USER_HICACHE:-per-conc} mamba_ratio=${USER_MAMBA_RATIO:-per-conc}"
    echo "duration=$DURATION conc=($CONC_LIST) gpus=${!VISIBLE_VAR:-all}"
    echo "ix=$IX@$(git -C "$IX" rev-parse --short HEAD 2>/dev/null) image=$DOCKER"
    echo "root=$ROOT"
} | tee -a "$SWEEP_LOG"

if [ "$DRY_RUN" = "1" ]; then
    # The per-concurrency resolvers live inside the sweep loop, so a dry run
    # has to walk the same list or it prints nothing useful.
    echo "--- resolved per concurrency ---"
    printf '%6s %14s %8s %12s %10s %10s\n' conc mem_fraction chunk mamba_ratio kv hicache
    for CONC in $CONC_LIST; do
        conc_defaults "$CONC"
        export CONC MEM_FRACTION_STATIC CHUNKED_PREFILL_SIZE
        [ -n "$MAMBA_FULL_MEMORY_RATIO" ] && export MAMBA_FULL_MEMORY_RATIO
        resolve_kv_tier
        printf '%6s %14s %8s %12s %10s %10s\n' "$CONC" "$MEM_FRACTION_STATIC" \
            "$CHUNKED_PREFILL_SIZE" "${MAMBA_FULL_MEMORY_RATIO:-default}" \
            "$KV_OFFLOADING" "$HICACHE_SIZE"
    done
    echo
    echo "--- recipe env (dry run, last concurrency) ---"
    env | grep -E '^(MODEL|TP|EP_SIZE|PP_SIZE|DCP_SIZE|PCP_SIZE|DP_ATTENTION|SPEC_|ACC_MODE|GOLDEN_AL|KV_|HICACHE|TOTAL_CPU_DRAM_GB|MAX_MODEL_LEN|CONTEXT_LENGTH|MAMBA_|MAX_MAMBA|CUDA_GRAPH|MEM_FRACTION|CHUNKED_|AIPERF_|AGENTIC_|INFMAX_|RUNNER_TYPE|PRECISION|FRAMEWORK|SCENARIO_|DURATION|ENABLE_AGENTX_POWER|REQUIRE_POWER|IS_MULTINODE|DISAGG|EVAL_ONLY|IMAGE|HF_HUB_CACHE)' | sort
    exit 0
fi

# ---- sweep ---------------------------------------------------------------
for CONC in $CONC_LIST; do
    export CONC
    conc_defaults "$CONC"
    export MEM_FRACTION_STATIC CHUNKED_PREFILL_SIZE
    [ -n "$MAMBA_FULL_MEMORY_RATIO" ] && export MAMBA_FULL_MEMORY_RATIO
    resolve_kv_tier
    export EXP_NAME="glm5.3flash_tp${TP}_conc${CONC}_kv${KV_OFFLOADING}_spec-${SPEC_DECODING}"
    export RESULT_FILENAME="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-pp1-dcp1-pcp1-ep${EP}-dpa${DP_ATTENTION}_disagg-${DISAGG}_spec-${SPEC_DECODING}_conc${CONC}_local-${RUNNER_TYPE}"
    export RESULT_DIR="$ROOT/$EXP_NAME"
    export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
    export PORT=$((PORT_BASE + CONC))
    export GPU_METRICS_CSV="$RESULT_DIR/gpu_metrics.csv"
    export SGLANG_TORCH_PROFILER_DIR="$RESULT_DIR"

    if [ -f "$RESULT_DIR/$RESULT_FILENAME.json" ]; then
        echo ">>> conc=$CONC already has a result, skipping." | tee -a "$SWEEP_LOG"
        continue
    fi
    mkdir -p "$RESULT_DIR"

    # A server that outlived the previous point keeps both the port and its
    # share of HBM, and booting on top of it silently halves the KV pool.
    kill_port "$PORT"
    for _ in $(seq 1 60); do
        busy=$(gpu_mem_used_max_mib)
        [ "${busy:-0}" -le 1024 ] && break
        echo "    waiting for GPU reclaim (max used=${busy} MiB)" | tee -a "$SWEEP_LOG"
        sleep 15
    done

    echo ">>> $(date -Is) starting conc=$CONC port=$PORT mem_fraction=$MEM_FRACTION_STATIC chunk=$CHUNKED_PREFILL_SIZE mamba_ratio=${MAMBA_FULL_MEMORY_RATIO:-default} kv=$KV_OFFLOADING hicache=$HICACHE_SIZE" | tee -a "$SWEEP_LOG"
    echo "    -> $RESULT_DIR" | tee -a "$SWEEP_LOG"
    bash "$RECIPE" > "$RESULT_DIR/recipe.log" 2>&1
    rc=$?
    echo ">>> $(date -Is) conc=$CONC exit=$rc" | tee -a "$SWEEP_LOG"

    # The two numbers that decide whether the hybrid memory split is sane: the
    # full-attention token pool and the KDA state slot count. Both move with
    # --mamba-ratio, and neither appears in the result JSON.
    grep -hoE 'KV Cache is allocated.*|max_mamba_cache_size=[0-9]+|Memory pool end.*' \
        "$RESULT_DIR/server.log" 2>/dev/null | sort -u | sed 's/^/    /' | tee -a "$SWEEP_LOG"

    # The recipe leaves the server up when it exits non-zero mid-flight.
    kill_server_on_port "$PORT"
    sleep 30
done

echo "=== $(date -Is) sweep done: $ROOT ===" | tee -a "$SWEEP_LOG"
