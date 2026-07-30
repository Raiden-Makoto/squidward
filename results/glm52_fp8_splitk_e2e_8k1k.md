# GLM-5.2-MXFP4 fp8 dense proj GEMM split-K — e2e 8k/1k

Box `smci355-ccs-aus-m12-17`, container `macui-glm5`, image
`raidenmakoto916/raidenmakoto:glm52-fork-aiter0724-ckaf7118-ckgemm1`, GPUs 4-7, TP4,
`--kv-cache-dtype fp8_e4m3`. aiter `/sgl-workspace/aiter_dev` (split-K build, `AITER_PATH`).
`CONCURRENCY="16 64" REPS=3 bash utilities/e2e_glm5.sh 8192 1024`, one server per arm,
every value the mean over 3 reps of that rep's median. Feature = `SGLANG_DSA_FP8_PROJ_GEMM=1`
with split-K rows in `utilities/glm5_a8w8_blockscale_bpreshuffle_tuned.csv` (q_b_proj 4096×2048
M 1–16, o_proj 6144×4096 M 1–24, splitK 2–3).

Arms A and C: sglang `RM/glm51` @ `9be13849b7`. Arm B: @ `12c2f247e4` (adds the prefill o_proj
pre-quant and drops the bpreshuffle scale materialize; both gated on the feature flag, so the
A baseline is unchanged by it and is not re-measured).

A. Baseline, bf16 dense proj (`SGLANG_DSA_FP8_PROJ_GEMM=0`):


| concurrency | TTFT (ms) | ITL (ms) | TPOT (ms) | E2EL (ms) | output tok/s |
| ----------- | --------- | -------- | --------- | --------- | ------------ |
| 16          | 3962      | 16.47    | 19.49     | 23904     | 685.3        |
| 64          | 14567     | 27.01    | 40.13     | 55597     | 1178.5       |


B. Feature on, fp8 dense proj + split-K + prefill o_proj pre-quant (Δ vs baseline):


| concurrency | TTFT (ms) | Δ     | ITL (ms) | Δ     | TPOT (ms) | Δ     | E2EL (ms) | Δ     | output tok/s | Δ     |
| ----------- | --------- | ----- | -------- | ----- | --------- | ----- | --------- | ----- | ------------ | ----- |
| 16          | 3966      | +0.1% | 16.28    | −1.2% | 19.28     | −1.1% | 23685     | −0.9% | 691.2        | +0.9% |
| 64          | 14511     | −0.4% | 26.96    | −0.2% | 40.01     | −0.3% | 55397     | −0.4% | 1182.6       | +0.3% |


C. Feature on, before the prefill o_proj pre-quant (`9be13849b7`), same arms (Δ vs baseline):


| concurrency | TTFT (ms) | Δ     | ITL (ms) | Δ     | TPOT (ms) | Δ     | E2EL (ms) | Δ     | output tok/s | Δ     |
| ----------- | --------- | ----- | -------- | ----- | --------- | ----- | --------- | ----- | ------------ | ----- |
| 16          | 4024      | +1.6% | 16.33    | −0.9% | 19.31     | −0.9% | 23734     | −0.7% | 690.2        | +0.7% |
| 64          | 14445     | −0.8% | 26.98    | −0.1% | 40.02     | −0.3% | 55415     | −0.3% | 1182.0       | +0.3% |


GSM8K (1319 questions, 5-shot, parallel 200) at `12c2f247e4`, feature on: 0.928, invalid 0.000.