# GLM-5.2-MXFP4 i1k/o1k baseline (scratch)

Stock aiter (no `SGLANG_MOE2_PIPE`), TP4 MI355X GPUs 4-7, port 8552.
`bash utilities/e2e_glm5.sh 1024 1024 0` (random, input=1024, output=1024, num-prompts=conc*4).

| concurrency | TTFT (ms) | ITL (ms) | E2EL (ms) | output tok/s |
|-------------|-----------|----------|-----------|--------------|
| 4           | 196.9     | 11.31    | 11775     | 346.9        |
| 8           | 366.8     | 13.00    | 13660     | 582.1        |
| 16          | 526.6     | 16.46    | 17432     | 935.0        |
| 32          | 783.2     | 20.48    | 21895     | 1498.13      |
| 64          | 1125.6    | 26.10    | 29352     | 2227.3       |