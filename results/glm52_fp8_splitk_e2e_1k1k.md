# GLM-5.2-MXFP4 fp8 dense proj GEMM split-K — e2e 1k/1k

Box `smci355-ccs-aus-m12-17`, container `macui-glm5`, GPUs 4-7, TP4, `--kv-cache-dtype fp8_e4m3`.
sglang `RM/glm51` @ `7a5364140`, aiter `/sgl-workspace/aiter_dev` (split-K build, `AITER_PATH`).
`CONCURRENCY="4 16 64" REPS=3 bash utilities/e2e_glm5.sh 1024 1024`, sequential A/B on one server,
every value the mean of 3 reps. Feature = `SGLANG_DSA_FP8_PROJ_GEMM=1` with split-K rows in
`utilities/glm5_a8w8_blockscale_bpreshuffle_tuned.csv` (q_b_proj 4096×2048 M 1–16,
o_proj 6144×4096 M 1–24, splitK 2–3).

A. Baseline, bf16 dense proj (`SGLANG_DSA_FP8_PROJ_GEMM=0`):

| concurrency | TTFT (ms) | ITL (ms) | TPOT (ms) | E2EL (ms) | output tok/s |
| ----------- | --------- | -------- | --------- | --------- | ------------ |
| 4 | 196.1 | 11.94 | 11.94 | 12411.7 | 328.8 |
| 16 | 493.8 | 16.04 | 16.13 | 16941.9 | 965.4 |
| 64 | 1130.1 | 26.78 | 27.37 | 29121.0 | 2246.9 |

B. Feature on, fp8 dense proj + split-K (Δ vs baseline):

| concurrency | TTFT (ms) | Δ | ITL (ms) | Δ | TPOT (ms) | Δ | E2EL (ms) | Δ | output tok/s | Δ |
| ----------- | --------- | ----- | -------- | ----- | --------- | ----- | --------- | ----- | ------------ | ----- |
| 4 | 195.3 | −0.4% | 11.85 | −0.7% | 11.85 | −0.7% | 12324.7 | −0.7% | 331.1 | +0.7% |
| 16 | 496.2 | +0.5% | 15.95 | −0.5% | 16.03 | −0.6% | 16847.2 | −0.6% | 971.0 | +0.6% |
| 64 | 1164.6 | +3.1% | 26.58 | −0.7% | 27.24 | −0.5% | 28993.9 | −0.4% | 2257.0 | +0.5% |

Same server, `SGLANG_DSA_FP8_PROJ_GEMM=1` without the split-K aiter build (stock aiter, split-K
rows inert), 3-rep means — ITL / TPOT / output tok/s: 4 = 12.34 / 12.34 / 318.3,
16 = 16.14 / 16.26 / 955.7, 64 = 26.57 / 27.23 / 2257.1.
