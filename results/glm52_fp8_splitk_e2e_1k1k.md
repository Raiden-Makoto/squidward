# GLM-5.2-MXFP4 fp8 dense proj GEMM split-K — e2e 1k/1k

Box `smci355-ccs-aus-m12-17`, container `glm5`, GPUs 4-7, TP4, `--kv-cache-dtype fp8_e4m3`.
sglang `RM/glm51` @ `0987fff7e`, aiter `/sgl-workspace/aiter_dev` (split-K build, `PYTHONPATH` override).
`utilities/e2e_glm5.sh 1024 1024`, `CONCURRENCY="4 4 8 16 32 64"`, sequential A/B on one server.
Both arms `SGLANG_DSA_FP8_PROJ_GEMM=1`; split-K rows come from the tuned CSV
`utilities/glm5_a8w8_blockscale_bpreshuffle_tuned.csv` (13 rows: q_b_proj 4096×2048 M 1–16,
o_proj 6144×4096 M 1–24, splitK 2–3). conc 4 = mean of 2 reps; conc 8/16/32/64 = 1 rep.

A. Baseline, tuned CSV without split-K rows:

| concurrency | TTFT (ms) | ITL (ms) | TPOT (ms) | E2EL (ms) | output tok/s |
| ----------- | --------- | -------- | --------- | --------- | ------------ |
| 4 | 195.6 | 12.23 | 12.24 | 12723.7 | 320.3 |
| 8 | 252.3 | 13.94 | 13.94 | 14539.3 | 561.1 |
| 16 | 485.6 | 16.11 | 16.26 | 17059.7 | 955.5 |
| 32 | 687.9 | 19.99 | 20.26 | 21405.1 | 1530.7 |
| 64 | 1373.1 | 26.59 | 27.23 | 29117.9 | 2248.2 |

B. Split-K rows on (Δ vs baseline):

| concurrency | TTFT (ms) | Δ | ITL (ms) | Δ | TPOT (ms) | Δ | E2EL (ms) | Δ | output tok/s | Δ |
| ----------- | --------- | ----- | -------- | ----- | --------- | ----- | --------- | ----- | ------------ | ----- |
| 4 | 209.9 | +7.3% | 11.88 | −2.9% | 11.89 | −2.9% | 12369.1 | −2.8% | 328.8 | +2.6% |
| 8 | 515.1 | +104.2% | 13.56 | −2.7% | 13.59 | −2.5% | 14465.6 | −0.5% | 566.7 | +1.0% |
| 16 | 494.8 | +1.9% | 15.98 | −0.8% | 16.02 | −1.5% | 16865.6 | −1.1% | 965.4 | +1.0% |
| 32 | 736.9 | +7.1% | 19.99 | +0.0% | 20.27 | +0.0% | 21463.8 | +0.3% | 1528.5 | −0.1% |
| 64 | 1114.7 | −18.8% | 26.60 | +0.0% | 27.25 | +0.1% | 28984.9 | −0.5% | 2255.1 | +0.3% |

bf16 reference (`SGLANG_DSA_FP8_PROJ_GEMM=0`), same server, ITL / TPOT / output tok/s per
concurrency: 4 = 11.94 / 11.94 / 327.1, 8 = 13.59 / 13.62 / 574.3, 16 = 16.03 / 16.13 / 960.7,
32 = 20.03 / 20.29 / 1524.8, 64 = 26.81 / 27.41 / 2244.6.
