# GLM-5.2-MXFP4 fp8 dense proj GEMM split-K — e2e 8k/1k

`RM/glm51` @ `12c2f247e4` (B), `9be13849b7` (A), image
`raidenmakoto916/raidenmakoto:glm52-fork-aiter0724-ckaf7118-ckgemm1`.
`bash utilities/run_glm52.sh`, then `CONCURRENCY="16 64" REPS=3 bash utilities/e2e_glm5.sh 8192 1024`.

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
