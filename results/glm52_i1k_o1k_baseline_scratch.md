# GLM-5.2-MXFP4 i1k/o1k (scratch)

TP4 MI355X GPUs 4-7, port 8552. `bash utilities/e2e_glm5.sh 1024 1024 0`
(random, input=1024, output=1024, num-prompts=conc*4), graphs-on, median.

## Baseline — fp8-proj off (bf16 dense-fallback-on, post-#30808)

| concurrency | TTFT (ms) | ITL (ms) | E2EL (ms) | output tok/s |
|-------------|-----------|----------|-----------|--------------|
| 4           | 196.9     | 11.31    | 11775     | 346.9        |
| 8           | 366.8     | 13.00    | 13660     | 582.1        |
| 16          | 526.6     | 16.46    | 17432     | 935.0        |
| 32          | 783.2     | 20.48    | 21895     | 1498.13      |
| 64          | 1125.6    | 26.10    | 29352     | 2227.3       |

## Feature — fp8-proj prefill-only gated (`SGLANG_DSA_FP8_PROJ_GEMM=1`, decode bf16)

Prefill (M>512) → tuned fp8 CK GEMM; decode (M<=512) → bf16. GSM8K 200q = 0.930.

| concurrency | TTFT (ms) | ITL (ms) | E2EL (ms) | output tok/s |
|-------------|-----------|----------|-----------|--------------|
| 4           | 197.6     | 11.84    | 12318     | 332.5        |
| 8           | 255.4     | 13.72    | 14402     | 569.1        |
| 16          | 499.5     | 16.39    | 17296     | 946.1        |
| 32          | 678.4     | 20.28    | 21778     | 1503.3       |
| 64          | 1123.5    | 27.10    | 29667     | 2209.3       |
