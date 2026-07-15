# GLM-5.2-MXFP4 i1k/o1k (scratch)

TP4 MI355X GPUs 4-7, port 8552. `bash utilities/e2e_glm5.sh 1024 1024 0`
(random, input=1024, output=1024, num-prompts=conc*4), graphs-on, median.
c4/c8 = 4-rep avg, c64 = 6-rep avg (obvious outlier dropped); c16/c32 = single run.
c64 TTFT is queue-variance-dominated (rep spread ~1.1-1.5s), so its Δ is noise.

## Baseline — fp8-proj off (bf16 dense-fallback-on, post-#30808)

| concurrency | TTFT (ms) | ITL (ms) | E2EL (ms) | output tok/s |
|-------------|-----------|----------|-----------|--------------|
| 4           | 198.9     | 11.82    | 12300     | 331.96       |
| 8           | 255.7     | 13.68    | 14337     | 570.97       |
| 16          | 526.6     | 16.46    | 17432     | 935.0        |
| 32          | 783.2     | 20.48    | 21895     | 1498.13      |
| 64          | 1102.0    | 27.17    | 29740     | 2201.9       |

## Feature — fp8-proj prefill-only gated (`SGLANG_DSA_FP8_PROJ_GEMM=1`, decode bf16)

Prefill (M>512) → tuned fp8 CK GEMM; decode (M<=512) → bf16. GSM8K 200q = 0.930.
Δ = vs baseline.

| concurrency | TTFT (ms) | Δ | ITL (ms) | Δ | E2EL (ms) | Δ | output tok/s | Δ |
|-------------|-----------|---|----------|---|-----------|---|--------------|---|
| 4           | 197.8     | −0.5%  | 11.83 | +0.1% | 12305 | +0.0% | 332.58 | +0.2% |
| 8           | 255.0     | −0.3%  | 13.74 | +0.4% | 14354 | +0.1% | 570.31 | −0.1% |
| 16          | 499.5     | −5.1%  | 16.39 | −0.4% | 17296 | −0.8% | 946.1  | +1.2% |
| 32          | 678.4     | −13.4% | 20.28 | −1.0% | 21778 | −0.5% | 1503.3 | +0.3% |
| 64          | 1137.4    | +3.2%  | 27.08 | −0.3% | 29658 | −0.3% | 2208.4 | +0.3% |
