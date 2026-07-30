# GLM-5.2-MXFP4 fp8 dense proj GEMM split-K — e2e 1k/1k

Box `smci355-ccs-aus-m12-17`, container `glm5`, GPUs 4-7, TP4, `--kv-cache-dtype fp8_e4m3`.
sglang `RM/glm51` @ `0987fff7e`, aiter `/sgl-workspace/aiter_dev` (split-K build, `PYTHONPATH` override).
`utilities/e2e_glm5.sh 1024 1024`, `CONCURRENCY="4 4 8 16 32 64"`, sequential on one server.
Feature = `SGLANG_DSA_FP8_PROJ_GEMM=1` with split-K rows in the tuned CSV
`utilities/glm5_a8w8_blockscale_bpreshuffle_tuned.csv` (13 rows: q_b_proj 4096×2048 M 1–16,
o_proj 6144×4096 M 1–24, splitK 2–3). conc 4 = mean of 2 reps; conc 8/16/32/64 = 1 rep.

A. Baseline, bf16 dense proj (`SGLANG_DSA_FP8_PROJ_GEMM=0`):

| concurrency | TTFT (ms) | ITL (ms) | TPOT (ms) | E2EL (ms) | output tok/s |
| ----------- | --------- | -------- | --------- | --------- | ------------ |
| 4 | 196.9 | 11.94 | 11.94 | 12416.9 | 327.1 |
| 8 | 253.2 | 13.59 | 13.62 | 14234.8 | 574.3 |
| 16 | 526.3 | 16.03 | 16.13 | 16981.4 | 960.7 |
| 32 | 747.5 | 20.03 | 20.29 | 21532.8 | 1524.8 |
| 64 | 1104.5 | 26.81 | 27.41 | 29178.6 | 2244.6 |

B. Feature on, fp8 dense proj + split-K (Δ vs baseline):

| concurrency | TTFT (ms) | Δ | ITL (ms) | Δ | TPOT (ms) | Δ | E2EL (ms) | Δ | output tok/s | Δ |
| ----------- | --------- | ----- | -------- | ----- | --------- | ----- | --------- | ----- | ------------ | ----- |
| 4 | 209.9 | +6.6% | 11.88 | −0.5% | 11.89 | −0.4% | 12369.1 | −0.4% | 328.8 | +0.5% |
| 8 | 515.1 | +103.4% | 13.56 | −0.2% | 13.59 | −0.2% | 14465.6 | +1.6% | 566.7 | −1.3% |
| 16 | 494.8 | −6.0% | 15.98 | −0.3% | 16.02 | −0.7% | 16865.6 | −0.7% | 965.4 | +0.5% |
| 32 | 736.9 | −1.4% | 19.99 | −0.2% | 20.27 | −0.1% | 21463.8 | −0.3% | 1528.5 | +0.2% |
| 64 | 1114.7 | +0.9% | 26.60 | −0.8% | 27.25 | −0.6% | 28984.9 | −0.7% | 2255.1 | +0.5% |

Same server, `SGLANG_DSA_FP8_PROJ_GEMM=1` without split-K rows (tuned CSV @ `HEAD~2`),
ITL / TPOT / output tok/s per concurrency: 4 = 12.23 / 12.24 / 320.3, 8 = 13.94 / 13.94 / 561.1,
16 = 16.11 / 16.26 / 955.5, 32 = 19.99 / 20.26 / 1530.7, 64 = 26.59 / 27.23 / 2248.2.
