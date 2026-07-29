#!/usr/bin/env bash
# Consolidated E2E throughput/latency sweep for GLM-5.2-MXFP4 (TP4, no DP)
# against a server already running (launch with run_glm52.sh first).
#
# GLM5 is TP4-only (no DP attention), so unlike the dsv4 low/high scripts this
# runs the FULL concurrency sweep in one shot.
#
# NOTE on profiling: --profile only writes traces if the SERVER was launched
# with SGLANG_TORCH_PROFILER_DIR set to a matching dir.
#
# A/B is sequential on ONE server: launch baseline (feature flag off), sweep,
# then relaunch with the feature flag on and sweep again. Same GPUs, same port.
# There is no simultaneous two-server setup.
#
# Usage:
#   bash e2e_glm5.sh [INPUT_LEN] [OUTPUT_LEN] [ENABLE_PROFILE]
#   OUT_DIR=/root/glm5-bench-feat bash e2e_glm5.sh            # tag the run's output
#   CONCURRENCY="1 2 4 8 16 32 64" bash e2e_glm5.sh            # override sweep
#   REPS=4 bash e2e_glm5.sh                                    # repeat each conc 4x
#
# REPS repeats the whole concurrency sweep N times; every run writes its own
# log (unique run index in the filename), so e2e_table.py averages across reps
# per concurrency (central-limit smoothing for the noisy TTFT median).

# ===== Default parameters =====
INPUT_LEN=${1:-8192}
OUTPUT_LEN=${2:-1024}
ENABLE_PROFILE=${3:-0}   # 1 = enable profile, 0 = disable

# ===== Server / sweep config (override via env) =====
PORT=${PORT:-8552}
CONCURRENCY=${CONCURRENCY:-"4 4 8 16 32 64"}
REPS=${REPS:-1}

# ===== Output directory (override with OUT_DIR=...) =====
OUT_DIR=${OUT_DIR:-/sgl-workspace/squidward/results/glm5-bench}
mkdir -p "${OUT_DIR}"

# ===== Timestamp =====
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Profile traces (when --profile is used) land here
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-${OUT_DIR}/${TIMESTAMP}_traces}"
if [ "${ENABLE_PROFILE}" -eq 1 ]; then
    mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"
fi

echo "MODEL=GLM-5.2-MXFP4 (TP4)"
echo "PORT=${PORT}"
echo "INPUT_LEN=${INPUT_LEN}"
echo "OUTPUT_LEN=${OUTPUT_LEN}"
echo "CONCURRENCY=${CONCURRENCY}"
echo "REPS=${REPS}"
echo "PROFILE=${ENABLE_PROFILE}"
echo "TIMESTAMP=${TIMESTAMP}"
echo "OUT_DIR=${OUT_DIR}"
[ "${ENABLE_PROFILE}" -eq 1 ] && echo "PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR}"

n=0
for rep in $(seq 1 ${REPS})
do
for concurrency in ${CONCURRENCY}
do
    n=$((n + 1))
    prompt=$((concurrency * 4))

    LOG_FILE="${OUT_DIR}/glm5_${INPUT_LEN}_${OUTPUT_LEN}_tp4_c-${concurrency}_run-${n}_${TIMESTAMP}.log"

    CMD="PYTHONPATH=/sgl-workspace/squidward/python:\${PYTHONPATH} python3 -m sglang.bench_serving \
        --backend sglang \
        --port ${PORT} \
        --dataset-name random \
        --random-input-len ${INPUT_LEN} \
        --random-output-len ${OUTPUT_LEN} \
        --random-range-ratio 1 \
        --max-concurrency ${concurrency} \
        --num-prompts ${prompt}"

    # ===== Optional profile =====
    if [ "${ENABLE_PROFILE}" -eq 1 ]; then
        CMD="${CMD} --profile --profile-num-steps 4 --profile-by-stage"
    fi

    echo "Running [rep ${rep}/${REPS}]: ${CMD}"
    echo "Log: ${LOG_FILE}"

    eval ${CMD} 2>&1 | tee "${LOG_FILE}"
done
done
