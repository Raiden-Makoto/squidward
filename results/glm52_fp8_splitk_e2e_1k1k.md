# GLM-5.2-MXFP4 fp8 dense proj GEMM split-K — e2e 1k/1k

Box `smci355-ccs-aus-m12-17`, container `glm5`, GPUs 4-7, TP4, `--kv-cache-dtype fp8_e4m3`.
sglang `RM/glm51` @ `0987fff7e`, aiter `/sgl-workspace/aiter_dev` (split-K build, `PYTHONPATH` override).
`utilities/e2e_glm5.sh 1024 1024`, `CONCURRENCY="4 4 8 16 32 64"`, sequential A/B/C on one server.
conc 4 = mean of 2 reps; conc 8/16/32/64 = 1 rep.
Split-K rows: tuned CSV `utilities/glm5_a8w8_blockscale_bpreshuffle_tuned.csv` (13 split-K rows at
the time of this run).

A. Baseline, bf16 dense proj (`SGLANG_DSA_FP8_PROJ_GEMM=0`):

| concurrency | TTFT (ms) | ITL (ms) | E2EL (ms) | output tok/s |
| ----------- | --------- | -------- | --------- | ------------ |
| 4 | 196.9 | 11.94 | 12416.9 | 327.1 |
| 8 | 253.2 | 13.59 | 14234.8 | 574.3 |
| 16 | 526.3 | 16.03 | 16981.4 | 960.7 |
| 32 | 747.5 | 20.03 | 21532.8 | 1524.8 |
| 64 | 1104.5 | 26.81 | 29178.6 | 2244.6 |

B. fp8 dense proj, no split-K rows (`SGLANG_DSA_FP8_PROJ_GEMM=1`, tuned CSV @ `HEAD~2`), Δ vs A:

| concurrency | TTFT (ms) | Δ | ITL (ms) | Δ | E2EL (ms) | Δ | output tok/s | Δ |
| ----------- | --------- | ----- | -------- | ----- | --------- | ----- | ------------ | ----- |
| 4 | 195.6 | −0.7% | 12.23 | +2.5% | 12723.7 | +2.5% | 320.3 | −2.1% |
| 8 | 252.3 | −0.4% | 13.94 | +2.6% | 14539.3 | +2.1% | 561.1 | −2.3% |
| 16 | 485.6 | −7.7% | 16.11 | +0.5% | 17059.7 | +0.5% | 955.5 | −0.5% |
| 32 | 687.9 | −8.0% | 19.99 | −0.2% | 21405.1 | −0.6% | 1530.7 | +0.4% |
| 64 | 1373.1 | +24.3% | 26.59 | −0.8% | 29117.9 | −0.2% | 2248.2 | +0.2% |

C. fp8 dense proj + split-K (`SGLANG_DSA_FP8_PROJ_GEMM=1`, tuned CSV @ `HEAD`), Δ vs A:

| concurrency | TTFT (ms) | Δ | ITL (ms) | Δ | E2EL (ms) | Δ | output tok/s | Δ |
| ----------- | --------- | ----- | -------- | ----- | --------- | ----- | ------------ | ----- |
| 4 | 209.9 | +6.6% | 11.88 | −0.5% | 12369.1 | −0.4% | 328.8 | +0.5% |
| 8 | 515.1 | +103.4% | 13.56 | −0.2% | 14465.6 | +1.6% | 566.7 | −1.3% |
| 16 | 494.8 | −6.0% | 15.98 | −0.3% | 16865.6 | −0.7% | 965.4 | +0.5% |
| 32 | 736.9 | −1.4% | 19.99 | −0.2% | 21463.8 | −0.3% | 1528.5 | +0.2% |
| 64 | 1114.7 | +0.9% | 26.60 | −0.8% | 28984.9 | −0.7% | 2255.1 | +0.5% |

D. Mean TPOT (ms):

| concurrency | A bf16 | B fp8 | C fp8+splitK | C vs B | C vs A | B vs A |
| ----------- | ------ | ----- | ------------ | ------ | ------ | ------ |
| 4 | 11.94 | 12.24 | 11.89 | −2.9% | −0.5% | +2.5% |
| 8 | 13.62 | 13.94 | 13.59 | −2.5% | −0.2% | +2.3% |
| 16 | 16.13 | 16.26 | 16.02 | −1.5% | −0.7% | +0.8% |
| 32 | 20.29 | 20.26 | 20.27 | +0.0% | −0.1% | −0.1% |
| 64 | 27.41 | 27.23 | 27.25 | +0.1% | −0.6% | −0.7% |

E. Kernel-level split-K coverage:

| shape | split-K M range | splitK |
| ----- | --------------- | ------ |
| q_b_proj 4096×2048 | 1–16 | 2–3 |
| o_proj 6144×4096 | 1–24 | 2–3 |
