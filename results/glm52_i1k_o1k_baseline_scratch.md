# GLM-5.2-MXFP4 i1k/o1k baseline (scratch)

Stock aiter (no `SGLANG_MOE2_PIPE`), TP4 MI355X GPUs 4-7, port 8552.
`bash utilities/e2e_glm5.sh 1024 1024 0` (random, input=1024, output=1024, num-prompts=conc*4).

| conc | out tok/s | med TTFT (ms) | med TPOT (ms) | med ITL (ms) | med E2E (ms) | req/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 332.85 | 198.02 | 11.82 | 11.83 | 12292.47 | 0.33 |
| 8 | 565.87 | 406.80 | 13.74 | 13.72 | 14469.77 | 0.55 |
| 16 | 935.04 | 526.60 | 16.53 | 16.46 | 17432.07 | 0.91 |
| 32 | 1498.13 | 783.24 | 20.61 | 20.33 | 21894.54 | 1.46 |
| 64 | 2198.15 | 1125.57 | 28.02 | 27.19 | 29812.57 | 2.15 |
