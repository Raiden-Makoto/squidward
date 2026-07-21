# GLM-5.2 MoE gemm1 (gate-up) — CK vs FlyDSL small-M analysis

- Shape: prod per-rank a4w4 mxfp4, `dim 6144,512`, E257, topk9, gfx950 / cu256
- Bench: `op_tests/test_moe_2stage.py --kernel` (kernel_bench, isolated stage1), GPU4, per-call µs
- CK forced via `AITER_BYPASS_TUNE_CONFIG=1 AITER_FLYDSL_FORCE=0` (kernel `kernel_moe_mxgemm_2lds`); FlyDSL = tuned `mfma_moe1`

## Crossover — CK (optimal block_m) vs FlyDSL

| M | tok/expert | CK µs | FlyDSL µs | Δ |
| ----- | --- | ---- | ---- | ---- |
| 1024  | 35  | 177  | 151  | +17% |
| 4096  | 143 | 307  | 270  | +14% |
| 8192  | 286 | 399  | 407  | −2%  |
| 16384 | 573 | 615  | 670  | −8%  |

Two independent causes.

## Cause 1 — `block_m` heuristic picks 128 at all M

CK stage1 µs vs forced `block_m`:

| M | bm32 | bm64 | bm128 | optimal |
| ----- | ---- | ---- | ----- | ------- |
| 1024  | 188  | 177  | 200   | 64  |
| 4096  | 388  | 304  | 311   | 64  |
| 8192  | 653  | 430  | 398   | 128 |
| 16384 | 1149 | 720  | 618   | 128 |

`get_block_size_M` sorts candidates by wave-count `(rnd, empty)`, which structurally always picks the largest `block_m` (128). It ignores intra-tile padding: at low tok/expert each expert's rows pad into a mostly-empty 128-row tile (waste ∝ `block_m / m_per_expert`). Optimal is bm64 ≤ ~256 tok/expert, bm128 above (bm32 never wins).

Fix (in `aiter_dev` scratch; needs an aiter PR to land): select `block_m` by per-expert occupancy instead of wave-count → **−11.5% @M1024, −1.3% @M4096, neutral ≥ M8192**.

## Cause 2 — VGPR pressure caps occupancy (residual small-M gap)

Even at optimal `block_m`, CK trails FlyDSL at M ≤ 4096. Code-object (`amdhsa.kernels`) metadata:

| kernel | VGPR/thread | LDS | occ limiter | ~waves/SIMD |
| ------ | ---- | ----- | ----- | ---- |
| CK gemm1 `kernel_moe_mxgemm_2lds` | 249 | 32 KB | VGPR | ~2 |
| FlyDSL `mfma_moe1` t64x64 | 100 | 40 KB | LDS | ~4–5 |

CK's 249 VGPRs cap occupancy at ~2 waves/SIMD. At small M the CUs are under-filled (few expert-tiles) and ~2 waves can't hide memory/MFMA latency → +17%. At large M the CUs saturate regardless and CK's heavier register blocking wins on per-wave MFMA throughput → −8%. Classic occupancy-vs-register-intensity crossover; VGPR=249 is the driver.

Fix: lower-VGPR CK gemm1 tiling (kernel-template change) to raise small-M occupancy, else keep the M-gated split.

## Deployment

- Mixed CK-gemm1 + FlyDSL-gemm2 is shelved: CK-stage1 sort layout ≠ FlyDSL-stage2 expected sort → wrong outputs at M ≥ 12288 (logits_diff 0.997).
- CK is only a win at M ≥ 8192; there the current heuristic already picks the optimal bm128, so Cause-1's fix does not change the deploy calculus — it only helps the M ≤ 4096 range, which stays FlyDSL regardless.
