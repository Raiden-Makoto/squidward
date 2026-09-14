#!/usr/bin/env bash
# Consolidated E2E throughput/latency sweep for GLM-5.3-Flash TP4 or TP8.
# Launch the matching checkpoint with run_glm53_flash.sh first.
#
# Usage:
#   bash utilities/e2e_glm53_flash.sh [INPUT_LEN] [OUTPUT_LEN] [ENABLE_PROFILE]
#   MODEL_VARIANT=fp8 bash utilities/e2e_glm53_flash.sh
#   TP=8 MODEL_VARIANT=fp8 REPS=2 bash utilities/e2e_glm53_flash.sh
#   CONCURRENCY="1 2 4 8 16 32 64" REPS=4 bash utilities/e2e_glm53_flash.sh
#
# MODEL_VARIANT labels outputs only; the running server determines the checkpoint.
# --profile requires the server to have SGLANG_TORCH_PROFILER_DIR configured.

set -euo pipefail

INPUT_LEN=${1:-8192}
OUTPUT_LEN=${2:-1024}
ENABLE_PROFILE=${3:-0}

PORT=${PORT:-8554}
TP=${TP:-4}
MODEL_VARIANT=${MODEL_VARIANT:-mxfp4}
CONCURRENCY=${CONCURRENCY:-"4 4 8 16 32 64"}
REPS=${REPS:-1}

case "${MODEL_VARIANT}" in
  mxfp4|fp8) ;;
  *)
    echo "MODEL_VARIANT must be 'mxfp4' or 'fp8'." >&2
    exit 2
    ;;
esac

case "${TP}" in
  4|8) ;;
  *)
    echo "TP must be 4 or 8, got ${TP}." >&2
    exit 2
    ;;
esac

OUT_DIR=${OUT_DIR:-/sgl-workspace/squidward/results/glm53-flash-${MODEL_VARIANT}-bench}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "${OUT_DIR}"

if ! curl --fail --silent --show-error --max-time 5 \
  "http://127.0.0.1:${PORT}/health" >/dev/null; then
  echo "No healthy SGLang server on port ${PORT}; launch run_glm53_flash.sh first." >&2
  exit 1
fi

export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH:-}
export SGLANG_TORCH_PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR:-${OUT_DIR}/${TIMESTAMP}_traces}
if [[ "${ENABLE_PROFILE}" == "1" ]]; then
  mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"
elif [[ "${ENABLE_PROFILE}" != "0" ]]; then
  echo "ENABLE_PROFILE must be 0 or 1." >&2
  exit 2
fi

echo "MODEL=GLM-5.3-Flash (${MODEL_VARIANT}, TP${TP})"
echo "PORT=${PORT}"
echo "INPUT_LEN=${INPUT_LEN}"
echo "OUTPUT_LEN=${OUTPUT_LEN}"
echo "CONCURRENCY=${CONCURRENCY}"
echo "REPS=${REPS}"
echo "PROFILE=${ENABLE_PROFILE}"
echo "TIMESTAMP=${TIMESTAMP}"
echo "OUT_DIR=${OUT_DIR}"
[[ "${ENABLE_PROFILE}" == "1" ]] && echo "PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR}"

run_index=0
for rep in $(seq 1 "${REPS}"); do
  for concurrency in ${CONCURRENCY}; do
    run_index=$((run_index + 1))
    num_prompts=$((concurrency * 4))
    log_file="${OUT_DIR}/glm53_flash_${MODEL_VARIANT}_${INPUT_LEN}_${OUTPUT_LEN}_tp${TP}_c-${concurrency}_run-${run_index}_${TIMESTAMP}.log"

    cmd=(
      python3 -m sglang.bench_serving
      --backend sglang
      --port "${PORT}"
      --dataset-name random
      --random-input-len "${INPUT_LEN}"
      --random-output-len "${OUTPUT_LEN}"
      --random-range-ratio 1
      --max-concurrency "${concurrency}"
      --num-prompts "${num_prompts}"
    )
    if [[ "${ENABLE_PROFILE}" == "1" ]]; then
      cmd+=(--profile --profile-num-steps 4 --profile-by-stage)
    fi

    printf 'Running [rep %s/%s]:' "${rep}" "${REPS}"
    printf ' %q' "${cmd[@]}"
    printf '\nLog: %s\n' "${log_file}"
    "${cmd[@]}" 2>&1 | tee "${log_file}"
  done
done
