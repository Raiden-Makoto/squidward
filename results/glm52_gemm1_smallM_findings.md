# GLM-5.2 MoE gemm1 (gate-up) — CK (ck2stages_v3) vs FlyDSL small-M analysis

- Shape: prod per-rank a4w4 mxfp4, `dim 6144,512`, E257, topk9, gfx950 / cu256
- CK candidate: `moe_ck2stages_gemm1_*_v3` — the deployable path, selected via a tuned
  `AITER_CONFIG_FMOE` `kernelName1`. NOT `AITER_BYPASS_TUNE_CONFIG` (that runs the legacy
  `kernel_moe_mxgemm_2lds` and is disregarded).
- gemm1 µs from aiter online-tune: best CK tile per M vs the auto-tuned best (deployed).

## Best CK tile vs FlyDSL (isolated gemm1 µs)

| M | best CK tile | CK µs | FlyDSL µs | winner |
| ----- | ------------------ | ----- | ----- | ------------- |
| 1024  | 256x64x128x128_1x4 | 209.6 | 185.9 | FlyDSL (+12.8%) |
| 4096  | 256x64x128x128_1x4 | 380.8 | 327.2 | FlyDSL (+16%) |
| 8192  | 256x64x128x128_1x4 | 549.8 | 493.8 | FlyDSL (+11%) |
| 16384 | 256x128x128x128_1x4| 843.9 | >843.9 | CK |

The auto-tuner deploys FlyDSL gemm1 for M ≤ 8192 and `ck2stages_gemm1_256x128x128x128_1x4`
only at M = 16384. Crossover is 8192↔16384 — higher, and small-M losses larger (+11–16%),
than the legacy bypass kernel appeared to show.

## Why CK loses at small M

The CK tile family's small/mid-M pick is `256x64x128x128_`**`1x4`** — 4 N-accumulator XDL
tiles per wave → high VGPR → low occupancy. At small M the CUs are under-filled (few
expert-tiles) so low occupancy cannot hide latency → CK trails FlyDSL. The lower-VGPR tiles
that are codegen'd (`64x32x32x128_`**`1x1`**) raise occupancy but are too small (low
arithmetic intensity) and the tuner only selects them at tiny M. CK's ck2stages tile set has
no small-M configuration that beats FlyDSL; only the large `256x128_1x4` tile's throughput
wins, and only at M = 16384. Occupancy-vs-register-intensity tradeoff.

## block_m

`get_block_size_M` (adaptive fix prototyped in aiter_dev) governs the mxfp4 fallback/bypass
path only. For the deployable ck2stages path the per-M tile — including block_m — comes from
the tuned `AITER_CONFIG_FMOE`, not `get_block_size_M`.

## Deployment

CK gemm1 is a win only at M ≥ 16384 (`256x128_1x4`). Mixed CK-gemm1 + FlyDSL-gemm2 is shelved
(CK-stage1 sort layout ≠ FlyDSL-stage2, wrong outputs at M ≥ 12288). Net: FlyDSL remains best
for GLM-5.2 gemm1 across the prefill range that matters; there is no compelling CK gemm1
deployment win except the largest chunks.
