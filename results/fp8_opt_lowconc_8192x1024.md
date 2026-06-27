gbt MI35x gfx950 (v4-pro), TP8,DP1 | random 8192 in / 1024 out | bench_serving, steady-state
delta% = fp8 optimized vs baseline (- = faster/lower/worse-tput, + = higher)

baseline `2294c4ca2` (bf16 unified-kv, SGLANG_UNIFIED_KV_FP8=0)

| Conc | TTT (tok/s) | E2EL (ms) | TTFT (ms) | ITL (ms) |
|---|---|---|---|---|
| 2  | 107.10 | 19114.53 | 636.47  | 18.06 |
| 4  | 207.12 | 19766.97 | 1090.84 | 18.26 |
| 8  | 370.69 | 22089.01 | 1805.55 | 19.83 |
| 16 | 604.77 | 27078.06 | 3190.73 | 23.35 |

fp8 optimized `2294c4ca2` (vectorized MXFP8 pack kernels over rows, SGLANG_UNIFIED_KV_FP8=1)

| Conc | TTT (tok/s) | E2EL (ms) | TTFT (ms) | ITL (ms) | TTT% | E2E% | TTFT% | ITL% |
|---|---|---|---|---|---|---|---|---|
| 2  | 90.31  | 22668.25 | 674.89  | 21.50 | -15.7 | +18.6 | +6.0 | +19.0 |
| 4  | 174.90 | 23409.61 | 1157.14 | 21.75 | -15.6 | +18.4 | +6.1 | +19.1 |
| 8  | 316.57 | 25865.54 | 1901.82 | 23.42 | -14.6 | +17.1 | +5.3 | +18.1 |
| 16 | 522.69 | 31331.28 | 3365.46 | 27.34 | -13.6 | +15.7 | +5.5 | +17.1 |
