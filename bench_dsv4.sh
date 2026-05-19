#!/bin/bash
# DSV4 serving benchmark.
# Sweeps concurrency 2/4/8/16/32 against a running server on --port 8000.
# Run inside the docker container after launching the server with python/run_dsv4.sh.
#
# Usage:
#   ./bench_dsv4.sh [INPUT_LEN] [OUTPUT_LEN] [ENABLE_PROFILE]
#   ./bench_dsv4.sh 8192 1024 0
#   OUT_DIR=/data/my_results ./bench_dsv4.sh 4096 512 1

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

INPUT_LEN=${1:-8192}
OUTPUT_LEN=${2:-1024}
ENABLE_PROFILE=${3:-0}

OUT_DIR=${OUT_DIR:-/data/results/v4-bench}
mkdir -p "${OUT_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-${OUT_DIR}/${TIMESTAMP}_traces}"
if [ "${ENABLE_PROFILE}" -eq 1 ]; then
    mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "INPUT_LEN=${INPUT_LEN}"
echo "OUTPUT_LEN=${OUTPUT_LEN}"
echo "PROFILE=${ENABLE_PROFILE}"
echo "TIMESTAMP=${TIMESTAMP}"
echo "OUT_DIR=${OUT_DIR}"
[ "${ENABLE_PROFILE}" -eq 1 ] && echo "PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR}"

for concurrency in 2 4 8
do
    prompt=$((concurrency * 4))

    LOG_FILE="${OUT_DIR}/dsv4_${INPUT_LEN}_${OUTPUT_LEN}_tp8_c-${concurrency}_${TIMESTAMP}.log"

    CMD="PYTHONPATH=${REPO_ROOT}/python:\${PYTHONPATH} python3 -m sglang.bench_serving \
        --port 8000 \
        --dataset-name random \
        --random-input ${INPUT_LEN} \
        --random-output ${OUTPUT_LEN} \
        --random-range-ratio 1 \
        --max-concurrency ${concurrency} \
        --num-prompt ${prompt}"

    if [ "${ENABLE_PROFILE}" -eq 1 ]; then
        CMD="${CMD} --profile --profile-num-steps 4 --profile-by-stage"
    fi

    echo "Running: ${CMD}"
    echo "Log: ${LOG_FILE}"

    eval ${CMD} 2>&1 | tee "${LOG_FILE}"
done
