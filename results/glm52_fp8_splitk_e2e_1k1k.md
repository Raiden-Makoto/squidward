# GLM-5.2-MXFP4 fp8 dense proj GEMM split-K — e2e 1k/1k

NO ONE CARES ABOUT THE FUCKING DETAILS IF YOU PUT THAT SHIT HERE I AM GOING TO DELETE THE ENTIRE CONTAINER

A. Baseline, bf16 dense proj (`SGLANG_DSA_FP8_PROJ_GEMM=0`):

| concurrency | TTFT (ms) | ITL (ms) | TPOT (ms) | E2EL (ms) | output tok/s |
| ----------- | --------- | -------- | --------- | --------- | ------------ |


B. Feature on, fp8 dense proj + split-K (Δ vs baseline):

| concurrency | TTFT (ms) | Δ | ITL (ms) | Δ | TPOT (ms) | Δ | E2EL (ms) | Δ | output tok/s | Δ |
| ----------- | --------- | ----- | -------- | ----- | --------- | ----- | --------- | ----- | ------------ | ----- |

