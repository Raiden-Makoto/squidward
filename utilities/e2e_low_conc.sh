#!/usr/bin/env bash
# E2E throughput / latency sweep for DeepSeek-V4 (dsv4 backend) against a server
# already running on port 8000 (launch it with run_dsv4.sh first).
#
# NOTE on profiling: --profile only writes traces if the SERVER was launched with
# SGLANG_TORCH_PROFILER_DIR set. To capture traces, relaunch run_dsv4.sh with
# SGLANG_TORCH_PROFILER_DIR=<dir> matching the OUT_DIR/timestamp below.

# ===== Default parameters =====
INPUT_LEN=${1:-8192}
OUTPUT_LEN=${2:-1024}
ENABLE_PROFILE=${3:-0}   # 1 = enable profile, 0 = disable

# ===== Output directory (override with OUT_DIR=...) =====
OUT_DIR=${OUT_DIR:-/sgl-workspace/squidward/results/v4-bench}
mkdir -p "${OUT_DIR}"

# ===== Timestamp =====
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Profile traces (when --profile is used) land here
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-${OUT_DIR}/${TIMESTAMP}_traces}"
if [ "${ENABLE_PROFILE}" -eq 1 ]; then
    mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"
fi

echo "INPUT_LEN=${INPUT_LEN}"
echo "OUTPUT_LEN=${OUTPUT_LEN}"
echo "PROFILE=${ENABLE_PROFILE}"
echo "TIMESTAMP=${TIMESTAMP}"
echo "OUT_DIR=${OUT_DIR}"
[ "${ENABLE_PROFILE}" -eq 1 ] && echo "PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR}"

echo "TP,DP must be 8,1. Check the server startup flags."
for concurrency in 2 4 8 16 
do
    prompt=$((concurrency * 4))

    LOG_FILE="${OUT_DIR}/dsv4_${INPUT_LEN}_${OUTPUT_LEN}_tp8_c-${concurrency}_${TIMESTAMP}.log"

    CMD="PYTHONPATH=/sgl-workspace/squidward/python:\${PYTHONPATH} python3 -m sglang.bench_serving \
        --backend sglang \
        --port 8000 \
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

    echo "Running: ${CMD}"
    echo "Log: ${LOG_FILE}"

    eval ${CMD} 2>&1 | tee "${LOG_FILE}"
done