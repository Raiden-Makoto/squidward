fp8 unified-kv attn (fused norm+rope+fp8+scatter store): squidward `9f8c18ac0` + aiter `0b2c5b69a`
random 8192 in / 1024 out, MI35x (v4-pro); low-conc DP off, high-conc DP on
ratio = bf16 / fp8 for latency, fp8 / bf16 for TTT; ratio < 1 = fp8 regression

bf16 baseline (squidward `main c328b4278`)

| Concurrency | TP, DP | TTT (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) |
|---|---|---|---|---|---|
| 2  | 8    | 1166.66  | 15820.17 | 652.37  | 14.67 |
| 4  | 8    | 2221.21  | 16764.26 | 1222.08 | 14.90 |
| 8  | 8    | 3793.58  | 19452.31 | 1942.04 | 16.00 |
| 16 | 8    | 5995.11  | 24643.31 | 3295.71 | 18.24 |
| 32 | 8, 8 | 7586.16  | 39294.32 | 4448.41 | 26.43 |
| 64 | 8, 8 | 11822.19 | 49489.13 | 7737.26 | 31.23 |

fp8 unified-kv (fused store)

| Concurrency | TP, DP | TTT (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | TTT perf | E2E | TTFT | ITL |
|---|---|---|---|---|---|---|---|---|---|
| 2  | 8    | 1044.82 | 17678.23 | 701.17   | 15.52 | 0.90 | 0.89 | 0.93 | 0.95 |
| 4  | 8    | 2030.22 | 18077.60 | 1277.79  | 15.97 | 0.91 | 0.93 | 0.96 | 0.93 |
| 8  | 8    | 3473.08 | 21294.76 | 1958.50  | 17.35 | 0.92 | 0.91 | 0.99 | 0.92 |
| 16 | 8    | 5520.62 | 26827.08 | 3368.73  | 19.92 | 0.92 | 0.92 | 0.98 | 0.92 |
| 32 | 8, 8 | 4449.31 | 64946.71 | 12262.77 | 32.12 | 0.59 | 0.60 | 0.36 | 0.82 |
| 64 | 8, 8 | 6049.46 | 96016.42 | 20244.93 | 38.33 | 0.51 | 0.52 | 0.38 | 0.81 |
