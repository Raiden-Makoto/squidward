#!/usr/bin/env bash
# Reproducible GSM8K evaluation for GLM-5.3-Flash MXFP4 or block FP8.
# Launch the selected checkpoint with run_glm53_flash.sh first.
#
# Usage:
#   bash utilities/e2e_glm53_flash_gsm8k.sh mxfp4
#   bash utilities/e2e_glm53_flash_gsm8k.sh fp8
#   NUM_EXAMPLES=20 NUM_THREADS=8 bash utilities/e2e_glm53_flash_gsm8k.sh fp8

set -euo pipefail

VARIANT=${1:-${MODEL_VARIANT:-mxfp4}}
PORT=${PORT:-8554}
NUM_EXAMPLES=${NUM_EXAMPLES:-1319}
NUM_THREADS=${NUM_THREADS:-64}
MAX_TOKENS=${MAX_TOKENS:-32768}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
THINKING=${THINKING:-1}

case "${VARIANT}" in
  mxfp4)
    DEFAULT_MODEL=/data2/hf_home/hub/models--amd--GLM-5.3-Flash-Quark-MXFP4/snapshots/b5688f25491202978c19c4d036eef579f61bbe07
    ;;
  fp8)
    DEFAULT_MODEL=/data2/hf_home/hub/models--zai-org--GLM-5.3-Flash/snapshots/03eb5366286afd40d2221b1d9c63a6dd1ba4832e
    ;;
  *)
    echo "Variant must be 'mxfp4' or 'fp8'." >&2
    exit 2
    ;;
esac

MODEL=${MODEL_PATH:-${DEFAULT_MODEL}}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR=${OUT_DIR:-/sgl-workspace/squidward/results/glm53-flash-${VARIANT}-gsm8k/${TIMESTAMP}}

if ! curl --fail --silent --show-error --max-time 5 \
  "http://127.0.0.1:${PORT}/health" >/dev/null; then
  echo "No healthy SGLang server on port ${PORT}; launch run_glm53_flash.sh first." >&2
  exit 1
fi

if ! python3 -c "import sgl_eval" >/dev/null 2>&1; then
  echo "sgl-eval is not installed in the active Python environment." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

cmd=(
  python3 -m sgl_eval.cli run gsm8k
  --base-url "http://127.0.0.1:${PORT}/v1"
  --model "${MODEL}"
  --num-examples "${NUM_EXAMPLES}"
  --num-threads "${NUM_THREADS}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --out-dir "${OUT_DIR}"
)
if [[ "${THINKING}" == "1" ]]; then
  cmd+=(--thinking)
elif [[ "${THINKING}" != "0" ]]; then
  echo "THINKING must be 0 or 1." >&2
  exit 2
fi

echo "MODEL=GLM-5.3-Flash (${VARIANT}, TP4)"
echo "MODEL_PATH=${MODEL}"
echo "PORT=${PORT}"
echo "NUM_EXAMPLES=${NUM_EXAMPLES}"
echo "NUM_THREADS=${NUM_THREADS}"
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "THINKING=${THINKING}"
echo "OUT_DIR=${OUT_DIR}"
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
