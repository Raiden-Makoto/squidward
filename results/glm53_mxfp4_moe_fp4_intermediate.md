# GLM-5.3-Flash MXFP4 MoE FP4 scatter intermediate — TP4 and TP8 — MI355X

`AITER_MXFP4_INTERMEDIATE=1` makes the AITER FlyDSL MoE stage-2 `scatter` epilogue (`flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_scatter_sbm128`) write its per-expert partials as MXFP4 (`_f4out` GEMM2) and sum them with `scatter_reduce_mxfp4_kernel`, instead of writing BF16 partials for `scatter_reduce`. No code change; the selector is an AITER environment variable. It only affects token buckets whose tuned row uses the `scatter` epilogue, which for GLM-5.3-Flash are the 16384-token rows (prefill batches of 8,193–16,384 tokens).

## Configuration

| Field | Value |
| --- | --- |
| Hardware | MI355X/gfx950, `smci355-ccs-aus-m12-17`; TP4 on GPUs 4–7, TP8 on GPUs 0–7 |
| Image | `rocm/sgl-dev:v0.5.22-rocm724-mi35x-20261019`, container `macui-glm53-new` |
| Source | `RM/glm53` `f320188a6a`, clean, run from `/sgl-workspace/squidward` |
| AITER | `/sgl-workspace/aiter` branch `validation/glm53-mxfp4-moe` `27b274d332` (image AITER `2bb549102` plus the [#6435](https://github.com/ROCm/aiter/pull/6435) GLM-5.3-Flash MXFP4 MoE configs, aux-kernel shapes and 16384-token rows), clean |
| Model | `amd/GLM-5.3-Flash-Quark-MXFP4@b5688f25491202978c19c4d036eef579f61bbe07` |
| Launcher | `utilities/run_glm53_flash.sh --mxfp4`; TP4 recipe (65,536 context, 64 max running requests, INT4 QuickReduce, full decode CUDA graphs) and `TP=8` recipe (131,072 context) |
| Scripts | `utilities/e2e_glm53_flash.sh` and `utilities/e2e_glm53_flash_gsm8k.sh`, snapshot `6aef5d5fc2` |
| A/B | Baseline `AITER_MXFP4_INTERMEDIATE=0`, candidate `AITER_MXFP4_INTERMEDIATE=1`; every other field identical; fresh AITER JIT, Triton and TileLang caches per arm |

## Accuracy

GSM8K, `utilities/e2e_glm53_flash_gsm8k.sh mxfp4`, 1,319 examples, temperature 1.0, top-p 0.95, thinking on, candidate only.

| TP | Threads | Max tokens | Score | 16384-token bucket reached |
| --- | ---: | ---: | ---: | --- |
| TP4 | 64 | 32,768 | 97.73% | No: largest prefill batch 6,124 tokens (64-request cap) |
| TP8 | 1,200 | 4,096 | 97.35% | Yes: 7 of 37 prefill batches above 8,192 tokens |

The TP4 GSM8K therefore does not exercise the FP4 path; the TP8 run does. Both ran on AITER `2bb549102` with the same uncommitted changes later committed as `27b274d332` (identical code diff).

### RULER2 long-context check (TP4)

Because GSM8K at TP4 never reaches the 16384-token bucket, accuracy on the FP4 path at TP4 was also checked with `sgl_eval` RULER2 at a 32,768-token sequence length: 12 subtasks × 100 samples, thinking off, 4 concurrent requests (eval defaults), temperature 1.0, top-p 0.95, `max_tokens` unset. It ran as an A/B on `smci355-ccs-aus-m15-21` with the TP4 launcher recipe, source `RM/glm53` `bb1a9c3718` and AITER `validation/glm53-mxfp4-moe-6672` `9135a3d3a` (tuned CSV identical to `27b274d332`), differing only by `AITER_MXFP4_INTERMEDIATE`. 2,393 of 2,417 candidate prefill batches were 16,384-token chunks on the FP4 path. Both arms had zero request errors and zero truncation.

| Subtask | A | B |
| --- | ---: | ---: |
| mk_niah_basic | 100.00% | 100.00% |
| mk_niah_easy | 97.78% | 96.28% |
| mk_niah_medium | 91.00% | 91.00% |
| mk_niah_hard | 94.00% | 95.00% |
| mv_niah_basic | 99.75% | 99.75% |
| mv_niah_easy | 81.92% | 81.77% |
| mv_niah_medium | 89.88% | 89.76% |
| mv_niah_hard | 95.03% | 89.94% |
| qa_basic | 100.00% | 100.00% |
| qa_easy | 97.00% | 96.00% |
| qa_medium | 89.00% | 88.00% |
| qa_hard | 85.00% | 91.00% |
| **Score** | **93.36%** | **93.21%** |

A = `AITER_MXFP4_INTERMEDIATE=0`, B = `AITER_MXFP4_INTERMEDIATE=1`. Per-subtask differences go both ways and are within the sampling noise of 100 samples at temperature 1.0; the headline differs by 0.15 points.

MMLU-Pro (12,032 questions, thinking off, TP8, 1,200 threads, 4,096 max tokens, same A/B and source) scored 76.31% (A) and 76.12% (B), but only 24 of about 6,600 prefill batches reached the 16384-token bucket, so it exercises the FP4 path very little. GPQA could not run because its dataset is gated.

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

## TP8 end-to-end serving

`TP=8 utilities/e2e_glm53_flash.sh 8192 1024 0` against the `TP=8` launcher recipe, random 8,192-token input, 1,024-token output, `--random-range-ratio 1`, `num_prompts = 4 × concurrency`, `REPS=2`, script default `CONCURRENCY="4 4 8 16 32 64"` with the first concurrency-4 run of each repetition excluded as warmup. Values average the two measured repetitions. All requests succeeded in both arms.

| Concurrency | Output tok/s A | Output tok/s B | Δ | Median TTFT ms A | Median TTFT ms B | Δ | Median ITL ms A | Median ITL ms B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 656.2 | 656.2 | −0.0% | 477.0 | 468.7 | −1.7% | 5.59 | 5.59 |
| 8 | 1125.8 | 1129.7 | +0.4% | 725.2 | 709.0 | −2.2% | 6.10 | 6.10 |
| 16 | 1851.1 | 1863.3 | +0.7% | 1215.6 | 1186.5 | −2.4% | 6.67 | 6.66 |
| 32 | 2763.9 | 2785.5 | +0.8% | 2195.3 | 2142.1 | −2.4% | 7.71 | 7.72 |
| 64 | 3811.0 | 3852.7 | +1.1% | 4154.7 | 4057.3 | −2.3% | 9.09 | 9.09 |

A = `AITER_MXFP4_INTERMEDIATE=0` (BF16 scatter partials), B = `AITER_MXFP4_INTERMEDIATE=1` (MXFP4 scatter partials). From concurrency 8 upward every measured repetition of B exceeds every measured repetition of A; at concurrency 4 throughput is unchanged.

## Kernel-level effect

AITER `gemm_moe_tune.py --run_config` on the six GLM-5.3-Flash 16384-token `scatter` rows (hidden 4096), GPU 4, two rounds in alternating order, AITER `main` `41d4d8d7e`. Times are the MoE stage total in µs.

| Inter dim / experts / top-k | BF16 scatter | FP4 scatter | Δ |
| --- | ---: | ---: | ---: |
| 512 / 288 / 8 (TP4) | 985 | 861 | −12.6% |
| 512 / 289 / 9 (TP4, shared-expert fused) | 1083 | 966 | −10.8% |
| 256 / 288 / 8 (TP8) | 787 | 648 | −17.7% |
| 256 / 289 / 9 (TP8, shared-expert fused) | 878 | 760 | −13.5% |

## FP8 reduce intermediate (not enabled)

`AITER_FLYDSL_STAGE2_FP8=1` makes the stage-2 `reduce` epilogue write FP8 per-expert partials (`_fp8out` GEMM2) and sum them with `moe_reduction_fp8_bf16`, instead of BF16 partials and `moe_reduction_bf16_bf16`. For GLM-5.3-Flash it covers the `reduce` rows: 2048–8192 tokens at TP4 and 512/1024–8192 tokens at TP8. It speeds up every `reduce` kernel row but does not change serving throughput beyond run-to-run noise, so it is not enabled.

| Field | Value |
| --- | --- |
| Hardware | MI355X/gfx950, `smci355-ccs-aus-m15-21`; TP4 on GPUs 4–7, TP8 on GPUs 0–7 |
| Image | `rocm/sgl-dev:v0.5.22-rocm724-mi35x-20261019`, container `macui-glm53-new` |
| Source | `RM/glm53` `bb1a9c3718` (python/ and utilities/ identical to `origin/RM/glm53` `8e3a6a1c01`), clean |
| AITER | `/sgl-workspace/aiter` branch `validation/glm53-mxfp4-moe-6672` `9135a3d3a` (AITER `313f86a7d`, the #6672 head, plus the [#6435](https://github.com/ROCm/aiter/pull/6435) GLM-5.3-Flash MXFP4 MoE configs, aux-kernel shapes and 16384-token rows; tuned CSV identical to `27b274d332`), clean |
| Launcher and scripts | As above; snapshot `6aef5d5fc2` |
| A/B | `AITER_MXFP4_INTERMEDIATE=1` in both arms; baseline `AITER_FLYDSL_STAGE2_FP8=0`, candidate `AITER_FLYDSL_STAGE2_FP8=1`; fresh AITER JIT, Triton and TileLang caches per arm |

### Kernel-level effect

AITER `gemm_moe_tune.py --run_config` on the 15 GLM-5.3-Flash `reduce` rows (hidden 4096), GPU 4 on m15-21, two rounds in alternating order. Times are the MoE stage total in µs. A kernel trace of the 8192-token 512/289/9 row confirms the candidate runs `gemm2_a4w4_port_..._fp8out_...` and `moe_reduction_fp8_bf16`.

| Inter dim / experts / tokens | BF16 | FP8 | Δ |
| --- | ---: | ---: | ---: |
| 512 / 288 / 2048 (TP4) | 255.6 | 229.2 | −10.3% |
| 512 / 288 / 4096 (TP4) | 349.1 | 292.3 | −16.3% |
| 512 / 288 / 8192 (TP4) | 602.0 | 471.4 | −21.7% |
| 512 / 289 / 2048 (TP4) | 276.4 | 254.2 | −8.1% |
| 512 / 289 / 4096 (TP4) | 394.4 | 336.1 | −14.8% |
| 512 / 289 / 8192 (TP4) | 657.5 | 531.2 | −19.2% |
| 256 / 288 / 1024 (TP8) | 131.1 | 117.9 | −10.1% |
| 256 / 288 / 2048 (TP8) | 171.0 | 151.7 | −11.3% |
| 256 / 288 / 4096 (TP8) | 257.8 | 206.2 | −20.0% |
| 256 / 288 / 8192 (TP8) | 450.1 | 321.3 | −28.6% |
| 256 / 289 / 512 (TP8) | 115.9 | 110.5 | −4.8% |
| 256 / 289 / 1024 (TP8) | 133.7 | 121.9 | −8.9% |
| 256 / 289 / 2048 (TP8) | 189.1 | 163.9 | −13.3% |
| 256 / 289 / 4096 (TP8) | 286.0 | 234.8 | −17.9% |
| 256 / 289 / 8192 (TP8) | 500.7 | 361.0 | −27.9% |

### Accuracy

GSM8K, candidate only, same evaluator settings as above. TP4 scored 97.27% (109 prefill batches in the 1,025–8,192-token `reduce` range). TP8 scored 96.89%; its first attempt failed in the evaluator client before any request completed (`OSError: Too many open files`, container soft `nofile` limit 1,024 on m15-21) and was rerun on a fresh server with `ulimit -n 524288`, all other settings unchanged.

### TP4 end-to-end serving

`utilities/e2e_glm53_flash.sh 8192 1024 0`, random 8,192-token input, 1,024-token output, `--random-range-ratio 1`, `num_prompts = 4 × concurrency`, `REPS=2`, script default `CONCURRENCY="4 4 8 16 32 64"` with the first concurrency-4 run of each repetition excluded as warmup. Values average the two measured repetitions. All requests succeeded in both arms.

| Concurrency | Output tok/s A | Output tok/s B | Δ | Median TTFT ms A | Median TTFT ms B | Δ | Median ITL ms A | Median ITL ms B |
| −--: | −--: | −--: | −--: | −--: | −--: | −--: | −--: | −--: |
| 4 | 596.0 | 596.2 | +0.0% | 570.6 | 560.3 | −1.8% | 6.07 | 6.08 |
| 8 | 1000.7 | 1000.0 | −0.1% | 869.6 | 857.2 | −1.4% | 6.76 | 6.79 |
| 16 | 1553.6 | 1551.2 | −0.2% | 1461.5 | 1450.6 | −0.7% | 7.91 | 7.94 |
| 32 | 2273.7 | 2271.9 | −0.1% | 2644.1 | 2634.1 | −0.4% | 9.39 | 9.42 |
| 64 | 3059.1 | 3062.5 | +0.1% | 5014.5 | 5001.4 | −0.3% | 11.59 | 11.59 |

### TP8 end-to-end serving

`TP=8 utilities/e2e_glm53_flash.sh 8192 1024 0`, random 8,192-token input, 1,024-token output, `--random-range-ratio 1`, `num_prompts = 4 × concurrency`, `REPS=2`, script default `CONCURRENCY="4 4 8 16 32 64"` with the first concurrency-4 run of each repetition excluded as warmup. Values average the two measured repetitions. All requests succeeded in both arms.

| Concurrency | Output tok/s A | Output tok/s B | Δ | Median TTFT ms A | Median TTFT ms B | Δ | Median ITL ms A | Median ITL ms B |
| −--: | −--: | −--: | −--: | −--: | −--: | −--: | −--: | −--: |
| 4 | 662.4 | 663.5 | +0.2% | 458.6 | 460.1 | +0.3% | 5.54 | 5.54 |
| 8 | 1138.5 | 1139.3 | +0.1% | 698.3 | 698.8 | +0.1% | 6.06 | 6.06 |
| 16 | 1882.8 | 1881.7 | −0.1% | 1171.1 | 1172.2 | +0.1% | 6.60 | 6.61 |
| 32 | 2818.1 | 2817.0 | −0.0% | 2106.7 | 2107.1 | +0.0% | 7.64 | 7.64 |
| 64 | 3906.9 | 3906.1 | −0.0% | 3990.9 | 3993.3 | +0.1% | 8.98 | 8.98 |

A = `AITER_FLYDSL_STAGE2_FP8=0`, B = `AITER_FLYDSL_STAGE2_FP8=1`, both with `AITER_MXFP4_INTERMEDIATE=1`. With 8,192-token prompts most prefill chunks are 16,384-token `scatter` batches, so the `reduce` rows run rarely in this workload. TP4 throughput changes by −0.2% to +0.1% (within repetition spread), and TP8 is unchanged.
