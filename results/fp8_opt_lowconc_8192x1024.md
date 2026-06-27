gbt MI35x gfx950 (v4-pro), TP8,DP1 | random 8192 in / 1024 out | bench_serving, steady-state
delta% = fp8 optimized vs baseline (- = faster/lower/worse-tput, + = higher)

baseline `e57e064b4c` (bf16 unified-kv)

| Conc | TTT (tok/s) | E2EL (ms) | TTFT (ms) | ITL (ms) |
|---|---|---|---|---|
| 2  | 108.00 | 18940.12 | 630.74  | 17.90 |
| 4  | 209.55 | 19534.37 | 1193.81 | 17.93 |
| 8  | 361.88 | 22568.12 | 1847.02 | 20.28 |
| 16 | 608.64 | 26881.90 | 3185.98 | 23.14 |

fp8 optimized `2294c4ca2` (vectorized MXFP8 pack kernels over rows)

| Conc | TTT (tok/s) | E2EL (ms) | TTFT (ms) | ITL (ms) | TTT% | E2E% | TTFT% | ITL% |
|---|---|---|---|---|---|---|---|---|
| 2  | 90.31  | 22668.25 | 674.89  | 21.50 | -16.4 | +19.7 | +7.0 | +20.1 |
| 4  | 174.90 | 23409.61 | 1157.14 | 21.75 | -16.5 | +19.8 | -3.1 | +21.3 |
| 8  | 316.57 | 25865.54 | 1901.82 | 23.42 | -12.5 | +14.6 | +3.0 | +15.5 |
| 16 | 522.69 | 31331.28 | 3365.46 | 27.34 | -14.1 | +16.5 | +5.6 | +18.2 |
