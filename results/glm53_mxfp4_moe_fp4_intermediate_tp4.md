# GLM-5.3-Flash MXFP4 MoE FP4 scatter intermediate — TP4 — MI355X

`AITER_MXFP4_INTERMEDIATE=1` makes the AITER FlyDSL MoE stage-2 `scatter` epilogue (`flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_scatter_sbm128`) write its per-expert partials as MXFP4 (`_f4out` GEMM2) and sum them with `scatter_reduce_mxfp4_kernel`, instead of writing BF16 partials for `scatter_reduce`. No code change; the selector is an AITER environment variable. It only affects token buckets whose tuned row uses the `scatter` epilogue, which for GLM-5.3-Flash are the 16384-token rows (prefill batches of 8,193–16,384 tokens).

## Configuration

| Field | Value |
| --- | --- |
| Hardware | MI355X/gfx950, `smci355-ccs-aus-m12-17`, GPUs 4–7 |
| Image | `rocm/sgl-dev:v0.5.22-rocm724-mi35x-20261019`, container `macui-glm53-new` |
| Source | `RM/glm53` `f320188a6a`, clean, run from `/sgl-workspace/squidward` |
| AITER | `/sgl-workspace/aiter` branch `validation/glm53-mxfp4-moe` `27b274d332` (image AITER `2bb549102` plus the [#6435](https://github.com/ROCm/aiter/pull/6435) GLM-5.3-Flash MXFP4 MoE configs, aux-kernel shapes and 16384-token rows), clean |
| Model | `amd/GLM-5.3-Flash-Quark-MXFP4@b5688f25491202978c19c4d036eef579f61bbe07` |
| Launcher | `utilities/run_glm53_flash.sh --mxfp4` (TP4 recipe: 65,536 context, 64 max running requests, INT4 QuickReduce, full decode CUDA graphs) |
| Scripts | `utilities/e2e_glm53_flash.sh` and `utilities/e2e_glm53_flash_gsm8k.sh`, snapshot `6aef5d5fc2` |
| A/B | Baseline `AITER_MXFP4_INTERMEDIATE=0`, candidate `AITER_MXFP4_INTERMEDIATE=1`; every other field identical; fresh AITER JIT, Triton and TileLang caches per arm |

## Accuracy

GSM8K, `utilities/e2e_glm53_flash_gsm8k.sh mxfp4`, 1,319 examples, temperature 1.0, top-p 0.95, thinking on, candidate only.

| TP | Threads | Max tokens | Score | 16384-token bucket reached |
| --- | ---: | ---: | ---: | --- |
| TP4 | 64 | 32,768 | 97.73% | No: largest prefill batch 6,124 tokens (64-request cap) |
| TP8 | 1,200 | 4,096 | 97.35% | Yes: 7 of 37 prefill batches above 8,192 tokens |

The TP4 GSM8K therefore does not exercise the FP4 path; the TP8 run does. Both ran on AITER `2bb549102` with the same uncommitted changes later committed as `27b274d332` (identical code diff). TP8 used the TP8 recipe (`TP=8`, 131,072 context, all eight GPUs).

## TP4 end-to-end serving

`utilities/e2e_glm53_flash.sh 8192 1024 0`, random 8,192-token input, 1,024-token output, `--random-range-ratio 1`, `num_prompts = 4 × concurrency`, `REPS=2`, script default `CONCURRENCY="4 4 8 16 32 64"` with the first concurrency-4 run of each repetition excluded as warmup. Values average the two measured repetitions. All requests succeeded in both arms.

| Concurrency | Output tok/s A | Output tok/s B | Δ | Median TTFT ms A | Median TTFT ms B | Δ | Median ITL ms A | Median ITL ms B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 590.6 | 590.4 | −0.0% | 578.9 | 575.0 | −0.7% | 6.12 | 6.12 |
| 8 | 988.3 | 989.9 | +0.2% | 887.0 | 877.0 | −1.1% | 6.83 | 6.84 |
| 16 | 1528.8 | 1532.8 | +0.3% | 1499.8 | 1478.6 | −1.4% | 8.00 | 8.01 |
| 32 | 2233.0 | 2242.8 | +0.4% | 2722.5 | 2682.4 | −1.5% | 9.50 | 9.51 |
| 64 | 2994.1 | 3015.6 | +0.7% | 5179.0 | 5096.4 | −1.6% | 11.75 | 11.75 |

A = `AITER_MXFP4_INTERMEDIATE=0` (BF16 scatter partials), B = `AITER_MXFP4_INTERMEDIATE=1` (MXFP4 scatter partials). Throughput gains at concurrency 32 and 64 exceed the repetition-to-repetition spread; at 16 and below they are within it. Decode latency is unchanged because decode batches do not use the 16384-token rows.

## Kernel-level effect

AITER `gemm_moe_tune.py --run_config` on the six GLM-5.3-Flash 16384-token `scatter` rows (hidden 4096), GPU 4, two rounds in alternating order, AITER `main` `41d4d8d7e`. Times are the MoE stage total in µs.

| Inter dim / experts / top-k | BF16 scatter | FP4 scatter | Δ |
| --- | ---: | ---: | ---: |
| 512 / 288 / 8 (TP4) | 985 | 861 | −12.6% |
| 512 / 289 / 9 (TP4, shared-expert fused) | 1083 | 966 | −10.8% |
| 256 / 288 / 8 (TP8) | 787 | 648 | −17.7% |
| 256 / 289 / 9 (TP8, shared-expert fused) | 878 | 760 | −13.5% |
