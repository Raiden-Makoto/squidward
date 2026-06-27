gbt MI35x gfx950 (v4-pro), TP8,DP1 | random 8192 in / 1024 out | e2e_low_conc.sh, bench_serving, steady-state
clean re-baseline: GPUs verified clear (rocm-smi ~0.3GB/GPU) before each launch, both halves on commit `78fc750f0` via run_dsv4.sh, differ only by SGLANG_UNIFIED_KV_FP8
delta% = fp8 vs bf16 baseline (- = lower, + = higher)

bf16 baseline `78fc750f0` (SGLANG_UNIFIED_KV_FP8=0)

| Conc | TTT (tok/s) | E2EL (ms) | TTFT (ms) | ITL (ms) |
|---|---|---|---|---|
| 2  | 107.49 | 19045.77 | 716.44  | 17.92 |
| 4  | 206.81 | 19797.24 | 1096.84 | 18.28 |
| 8  | 371.74 | 22026.79 | 1796.23 | 19.78 |
| 16 | 610.86 | 26807.74 | 3172.08 | 23.10 |

fp8 `78fc750f0` (in-kernel MXFP8 decode store, SGLANG_UNIFIED_KV_FP8=1) | GSM8K 8-shot acc 0.953

| Conc | TTT (tok/s) | E2EL (ms) | TTFT (ms) | ITL (ms) | TTT% | E2E% | TTFT% | ITL% |
|---|---|---|---|---|---|---|---|---|
| 2  | 94.28  | 21714.85 | 677.52  | 20.56 | -12.3 | +14.0 | -5.4 | +14.7 |
| 4  | 180.54 | 22677.63 | 1157.80 | 21.04 | -12.7 | +14.5 | +5.6 | +15.1 |
| 8  | 323.48 | 25313.19 | 1896.74 | 22.89 | -13.0 | +14.9 | +5.6 | +15.7 |
| 16 | 541.19 | 30259.69 | 3345.79 | 26.31 | -11.4 | +12.9 | +5.5 | +13.9 |
