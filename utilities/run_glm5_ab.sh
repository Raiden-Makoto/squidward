#!/usr/bin/env bash
# A/B launcher for GLM-5.2-MXFP4: two independent TP4 servers on disjoint GPU
# sets so a flag-gated feature can be compared against baseline side-by-side
# (no serial model reloads, identical hardware/thermals, same wall-clock window).
# Server config matches run_glm52.sh (canonical GLM-5.2 launcher).
#
#   Server A (baseline) -> GPUs 0-3, port 8552
#   Server B (feature)  -> GPUs 4-7, port 8553  + any args passed to this script
#
# Usage:
#   bash run_glm5_ab.sh --enable-aiter-allreduce-fusion
#   bash run_glm5_ab.sh --dsa-decode-backend aiter --dsa-prefill-backend tilelang
#   LOG_DIR=/root bash run_glm5_ab.sh            # baseline vs baseline (noise floor)
#
# Then drive both ports and compare, e.g.:
#   python3 -m sglang.bench_serving --backend sglang --port 8552 ... &
#   python3 -m sglang.bench_serving --backend sglang --port 8553 ...
#
# Requires 8 visible GPUs. Co-tenancy can perturb ABSOLUTE latency, so use this
# for RELATIVE A/B (A vs B under identical co-running conditions).
set -u

export PYTHONPATH=/sgl-workspace/squidward/python:${PYTHONPATH:-}
export SAFETENSORS_FAST_GPU=1
# Canonical GLM-5.2 gfx950 tuning (matches run_glm52.sh).
export SGLANG_DSA_TRITON_PREFILL=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

MODEL=${HF_HOME:-/root/hf_home}/hub/models--amd--GLM-5.2-MXFP4/snapshots/386bd0e4ec821f7b07975701cec3c3b953a5576a
LOG_DIR=${LOG_DIR:-/root}
PORT_A=${PORT_A:-8552}
PORT_B=${PORT_B:-8553}
GPUS_A=${GPUS_A:-0,1,2,3}
GPUS_B=${GPUS_B:-4,5,6,7}
FEATURE_ARGS=("$@")

mkdir -p "$LOG_DIR"

launch() {
  local label="$1" gpus="$2" port="$3"; shift 3
  local log="$LOG_DIR/glm5_ab_${label}.log"
  HIP_VISIBLE_DEVICES="$gpus" \
  SGLANG_TORCH_PROFILER_DIR="$LOG_DIR/prof_${label}" \
  env ${LAUNCH_ENV:-} \
  setsid sglang serve \
    --model-path "$MODEL" \
    --tp 4 \
    --host localhost \
    --port "$port" \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --watchdog-timeout 1200 \
    --mem-fraction-static 0.85 \
    --disable-radix-cache \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --kv-cache-dtype fp8_e4m3 \
    --tokenizer-worker-num 8 \
    --chunked-prefill-size 131072 \
    "$@" > "$log" 2>&1 < /dev/null &
  echo "  $label: pid=$! gpus=$gpus port=$port log=$log"
}

echo "Launching GLM-5.2-MXFP4 A/B servers"
echo "  feature flags on B (${PORT_B}): ${FEATURE_ARGS[*]:-(none -> baseline vs baseline)}"
# Server A = baseline. Server B gets FEATURE_ARGS (CLI flags) and, for env-gated
# features, FEATURE_ENV (e.g. FEATURE_ENV="SGLANG_DSA_FUSE_HADAMARD_QUANT=1").
launch A "$GPUS_A" "$PORT_A"
LAUNCH_ENV="${FEATURE_ENV:-}" launch B "$GPUS_B" "$PORT_B" "${FEATURE_ARGS[@]}"
echo
echo "Monitor:  tail -f $LOG_DIR/glm5_ab_A.log $LOG_DIR/glm5_ab_B.log"
echo "Health :  curl -s localhost:$PORT_A/health ; curl -s localhost:$PORT_B/health"
echo "Stop   :  pkill -9 -f '[s]glang serve'"
