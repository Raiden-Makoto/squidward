# GLM-5.3-Flash AgentX c64: validated commands

The exact commands behind the validated AgentX concurrency-64 A/Bs on MI355X (#48806 FP8 KV, #49036 BF16 KDA state). Every arm of those A/Bs passed the InferenceX health check with 0.0% errors, and repeats agreed within 1.1% on throughput.

## Tuple

| Field | Value |
| --- | --- |
| Hardware | MI355X (gfx950), m12, GPUs 4-7, TP4 |
| Image | `rocm/sgl-dev:v0.5.23-rocm724-mi35x-20261102` |
| Container | privileged, `--network host --ipc host --shm-size 32g -v /data2:/data2 -v /data:/data`, `psmisc` installed (the driver needs `fuser` and `cap_sys_nice`) |
| Driver | `Jacob0226/SGLang-benchmarks` branch `GLM5_InferenceX` at `4e3b460` (`Agentic/ix_agentx_glm53flash.sh`) |
| InferenceX | `bee3405ca` with the `utils/aiperf` submodule |
| AITER | `3015e3639` (`285579b62` with GLM-5.3 MXFP4 MoE changes) |
| Checkpoint | `/data2/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4` |
| Source | #49036: `a8e0045395` with #48112 applied |
| HiCache | `write_back`, kernel I/O, `page_first`, 216 GB/rank (driver default at conc 64) |
| Acceptance | simulated, length 3.56 (driver default) |
| Window | 3,600 s profiling (driver default; do not pass `--quick`) |
| Repeats | two runs per arm, interleaved A1 B1 A2 B2 |

## Container setup

```bash
docker run -d --name <container> --privileged --network host --ipc host --shm-size 32g \
  -v /data2:/data2 -v /data:/data rocm/sgl-dev:v0.5.23-rocm724-mi35x-20261102 sleep infinity
docker exec <container> bash -lc "apt-get update -qq && apt-get install -y -qq psmisc"
```

## Harness invocation

Run inside the container, one arm at a time on GPUs 4-7. The two arms differ only in the arm flag.

```bash
cd /tmp/SGLang-benchmarks/Agentic && \
HIP_VISIBLE_DEVICES=4,5,6,7 \
PYTHONPATH=<source>/python:<aiter> \
AITER_JIT_DIR=<aiter>/aiter/jit \
IX=/tmp/InferenceX \
BENCH_HOME=<bench_home> \
AIPERF_RUNTIME_DIR=<runtime>/<arm>-wb-r<rep> \
HF_HUB_CACHE=/data2/huggingface/hub \
DOCKER=rocm/sgl-dev:v0.5.23-rocm724-mi35x-20261102 \
./ix_agentx_glm53flash.sh --platform mi355x --gpus 4,5,6,7 \
  --ckpt /data2/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 \
  --conc 64 --dsa-backend triton --tag <TAG>-<arm>-c64-wb-r<rep> \
  --server-arg --skip-server-warmup \
  --server-arg "--kv-cache-dtype bfloat16" \
  <arm flag>
```

| A/B | Arm A flag | Arm B flag |
| --- | --- | --- |
| #49036 BF16 KDA state | none | `--server-arg --mamba-ssm-dtype=bfloat16` |
| #48806 FP8 KV cache | `--server-arg "--kv-cache-dtype bfloat16"` | `--server-arg "--kv-cache-dtype fp8_e4m3"` |

The driver at `4e3b460` defaults the KV cache to FP8, so a BF16 KV arm must pass `--kv-cache-dtype bfloat16` explicitly. Without it, sources without FP8 Triton DSA support fail at startup.

## Resulting server command

#49036 arm A (`sglang_command.txt`); arm B appends `--mamba-ssm-dtype=bfloat16`:

```bash
python3 -m sglang.launch_server --model-path /data2/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 --served-model-name amd/GLM-5.3-Flash-Quark-MXFP4 --host 0.0.0.0 --port 28964 --trust-remote-code --tp 4 --ep-size 1 --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton --moe-runner-backend aiter --mamba-radix-cache-strategy extra_buffer --tool-call-parser glm47 --reasoning-parser glm45 --chunked-prefill-size 16384 --mem-fraction-static 0.85 --max-running-requests 80 --cuda-graph-max-bs-decode 80 --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 2560 --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' --mamba-full-memory-ratio 0.5 --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --enable-hierarchical-cache --hicache-write-policy write_back --hicache-io-backend kernel --hicache-mem-layout page_first --hicache-size 216 --watchdog-timeout 1800 --enable-metrics --skip-server-warmup --kv-cache-dtype bfloat16
```

The later `--kv-cache-dtype bfloat16` overrides the driver's FP8 default. Environment exported by the driver:

```bash
AITER_QUICK_REDUCE_QUANTIZATION=INT4
ROCM_QUICK_REDUCE_QUANTIZATION=INT4
SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=0.04
SGLANG_SIMULATE_ACC_LEN=3.56
SGLANG_SIMULATE_ACC_METHOD=match-expected
SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
SGLANG_TIMEOUT_KEEP_ALIVE=900
```

## Resulting benchmark command

```bash
<runtime>/venv/bin/aiperf profile --scenario inferencex-agentx-mvp --url http://localhost:28964 --endpoint /v1/chat/completions --endpoint-type chat --streaming --model amd/GLM-5.3-Flash-Quark-MXFP4 --tokenizer amd/GLM-5.3-Flash-Quark-MXFP4 --concurrency 64 --benchmark-duration 3600 --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code --num-dataset-entries 393 --slice-duration 1.0 --output-artifact-dir <result_dir>/aiperf_artifacts --public-dataset semianalysis_cc_traces_weka_062126 --server-metrics http://localhost:28964/metrics
```

## Checks per run

- `helper/check_health.py <result_dir>`: 0 crashes, 0 OOM warnings, 0.0% errors.
- The driver prints `conc=64 exit=0`; it can exit 0 after a failed server start, so check that line rather than the exit code.
- `server.log` shows the intended `KV Cache is allocated. dtype:` and `mamba_ssm_dtype`.
- Both arms' `sglang_command.txt` differ only by the arm flag.

## This branch

This branch merges #48806 (FP8 KV with Triton DSA), #49036 (BF16 KDA state), #47336, #44661, #38764, #48303, #44824 and #48112, and adds the FlyDSL KDA MTP verify kernel, the DSA k-pool indexer, MoE router and BF16 GEMM dispatch, breakable prefill graphs and sort-free top-p. The FlyDSL verify kernel engages only with ReplaySSM spec and 6 or 8 draft tokens, so a run of this branch keeps the driver's FP8 KV default and adds the BF16 state, ReplaySSM spec and the opt-in selectors:

```bash
cd <SGLang-benchmarks>/Agentic && \
HIP_VISIBLE_DEVICES=4,5,6,7 \
PYTHONPATH=<this branch>/python:<aiter> \
IX=<InferenceX> BENCH_HOME=<bench_home> AIPERF_RUNTIME_DIR=<runtime>/<arm>-r<rep> \
HF_HUB_CACHE=/data2/huggingface/hub \
DOCKER=rocm/sgl-dev:v0.5.23-rocm724-mi35x-20261102 \
SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj \
SGLANG_DSA_FUSE_HADAMARD_QUANT=1 \
AITER_MXFP4_INTERMEDIATE=1 \
./ix_agentx_glm53flash.sh --platform mi355x --gpus 4,5,6,7 \
  --ckpt /data2/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 \
  --conc 64 --dsa-backend triton --tag <TAG>-<arm>-c64-wb-r<rep> \
  --server-arg --skip-server-warmup \
  --server-arg "--mamba-ssm-dtype bfloat16" \
  --server-arg "--enable-linear-replayssm-spec" \
  <arm flag>
```

The DSA k-pool top-k, grouped-top-k router and small-K GEMM dispatch are on by default on ROCm; the driver already passes `--cuda-graph-backend-prefill breakable`.

The stage-2 runs in `/data2/macui/glm53_agentx_stage2_ab` used this configuration but are not valid A/Bs: they passed `--quick` (600 s window instead of 3,600 s) and `--acc real` (instead of the simulated 3.56), started every run with empty AITER, Triton and TileLang caches, and shared one aiperf runtime directory between arms.
