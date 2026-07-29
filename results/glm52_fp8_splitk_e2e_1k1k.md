# GLM-5.2-MXFP4 fp8 dense proj GEMM split-K — e2e 1k/1k

Box `smci355-ccs-aus-m12-17`, container `glm5`, GPUs 4-7, TP4, `--kv-cache-dtype fp8_e4m3`.
sglang `RM/glm51` @ `0987fff7e`, aiter `/sgl-workspace/aiter_dev` (split-K build, `PYTHONPATH` override).
`utilities/e2e_glm5.sh 1024 1024`, `CONCURRENCY="4 4 8 16 32 64"`, sequential A/B on one server.

Arms:
- A `SGLANG_DSA_FP8_PROJ_GEMM=1`, tuned CSV `utilities/glm5_a8w8_blockscale_bpreshuffle_tuned.csv` (13 split-K rows)
- B `SGLANG_DSA_FP8_PROJ_GEMM=1`, tuned CSV @ `HEAD~2` (no split-K rows)
- C `SGLANG_DSA_FP8_PROJ_GEMM=0` (bf16 dense proj)

## Mean TPOT (ms)

| conc | C bf16 | B fp8 | A fp8+splitK | A vs B | A vs C | B vs C |
|---|---|---|---|---|---|---|
| 4 | 11.94 | 12.24 | 11.89 | −2.9% | −0.5% | +2.5% |
| 8 | 13.62 | 13.94 | 13.59 | −2.5% | −0.2% | +2.3% |
| 16 | 16.13 | 16.26 | 16.02 | −1.5% | −0.7% | +0.8% |
| 32 | 20.29 | 20.26 | 20.27 | +0.0% | −0.1% | −0.1% |
| 64 | 27.41 | 27.23 | 27.25 | +0.1% | −0.6% | −0.7% |

## Output token throughput (tok/s)

| conc | C bf16 | B fp8 | A fp8+splitK | A vs B | A vs C |
|---|---|---|---|---|---|
| 4 | 327.1 | 320.3 | 328.8 | +2.7% | +0.5% |
| 8 | 574.3 | 561.1 | 566.7 | +1.0% | −1.3% |
| 16 | 960.7 | 955.5 | 965.4 | +1.0% | +0.5% |
| 32 | 1524.8 | 1530.7 | 1528.5 | −0.1% | +0.2% |
| 64 | 2244.6 | 2248.2 | 2255.1 | +0.3% | +0.5% |

conc-4 rows average the two repeat runs.

## Kernel-level split-K coverage

| shape | split-K M range | splitK |
|---|---|---|
| q_b_proj 4096×2048 | 1–16 | 2–3 |
| o_proj 6144×4096 | 1–24 | 2–3 |
