gbt MI35x gfx950 (v4-pro), TP8,DP1 | random 8192 in / 1024 out | bench_serving, steady-state
delta% = fp8 optimized vs baseline (- = faster/lower/worse-tput, + = higher)

baseline `2294c4ca2` (bf16 unified-kv, SGLANG_UNIFIED_KV_FP8=0)

| Conc | TTT (tok/s) | E2EL (ms) | TTFT (ms) | ITL (ms) |
|---|---|---|---|---|
| 2  | 107.10 | 19114.53 | 636.47  | 18.06 |
| 4  | 207.12 | 19766.97 | 1090.84 | 18.26 |
| 8  | 370.69 | 22089.01 | 1805.55 | 19.83 |
| 16 | 604.77 | 27078.06 | 3190.73 | 23.35 |

fp8 optimized `905951e96` (vectorized MXFP8 packs + cached swa_loc decode store, SGLANG_UNIFIED_KV_FP8=1)

| Conc | TTT (tok/s) | E2EL (ms) | TTFT (ms) | ITL (ms) | TTT% | E2E% | TTFT% | ITL% |
|---|---|---|---|---|---|---|---|---|
| 2  | 93.72  | 21842.72 | 666.60  | 20.70 | -12.5 | +14.3 | +4.7 | +14.6 |
| 4  | 178.78 | 22902.01 | 1169.09 | 21.24 | -13.7 | +15.9 | +7.2 | +16.3 |
| 8  | 323.21 | 25335.17 | 1898.82 | 22.91 | -12.8 | +14.7 | +5.2 | +15.5 |
| 16 | 532.90 | 30730.70 | 3360.50 | 26.75 | -11.9 | +13.5 | +5.3 | +14.6 |
