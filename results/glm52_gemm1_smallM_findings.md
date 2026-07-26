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

## K=256 tile investigation (2026-07-25) — WHY CK loses small-M = KPerBlock hardcoded 128

Root cause of the small-M loss is that the a4w4 gufusion path only ships **K=128** tiles, while
FlyDSL small-M uses **K=256** (`t64x64x256`, `t128x128x256`) → half the K-loop iterations →
less prefetch-prologue overhead. Two blockers to enabling a K=256 CK tile:

1. `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_mxfp4.cuh` lines 106 (stage1) &
   298 (stage2): the DeviceGemm instantiation hardcodes `KPer=128` literally (ignores the
   `KPerBlock` template arg). Prior "KPerBlock inert" note was because only the codegen field was
   changed, never this literal. Fix: `128` → `KPerBlock`.
2. `a4w4_gemm1_kernels_list` in `gemm_moe_ck2stages_common.py` had NO K=256 tile. Added
   `(256,64,128,256,1,4,3)` and `(256,128,128,256,1,4,3)`.

With both applied, a K=256 a4w4 gufusion tile **compiles clean** and is **much faster** (isolated
gemm1 us_stage1, `--kernel` GPU0, 6144,512, same-harness vs FlyDSL):

| M    | K=256 CK | FlyDSL | K=128 CK |
| ---- | -------- | ------ | -------- |
| 512  | 65.9     | 139.2  | —        |
| 1024 | 123.2    | 154.8  | 209.6    |
| 4096 | 242.9    | 272.1  | 380.8    |
| 8192 | 391.0    | 408.5  | 549.8    |

BUT it is **numerically WRONG** (logits_diff ≈ 0.97–1.01, err ≈ 0.997 — output uncorrelated with
reference; NOT fp4 precision, which is ~1e-3..1e-6). So K=128 was the band-aid for a real
correctness bug. Localized: weights are K-agnostic (`shuffle_weight (16,16)` MFMA-fragment, fine);
the breaker is the **e8m0 B-scale n32k4 layout** (`aiter/ops/shuffle.py shuffle_scale`), which is
built on fixed **WMMA-K=128 super-blocks** (`col = remain_k*128 + row32*4 + r`, asserts `K%128==0`).
The kernel scale traversal (`ScalesPerKBlockSize=KPerBlock/32`, `KRepeat=KPerBlock/(KLane·KPack)`,
`KXdlPack=2` fixed at gridwise lines 297–299, `b_block_slice_copy_step=(0,0,0,KRepeat,0)`) walks
the n32k4 buffer a multi-`remain_k` path at K=256 that K=128 never exercises, so every block gets
mis-scaled → garbage.

Making K=256 correct = aligning the K=256 scale traversal to the n32k4 `remain_k*128` stride
(kernel-side `KRepeat`/scale-descriptor factoring, or a K-256-aware scale shuffle). Bounded but an
empirical build/test loop (recompile module ~2 min per cycle). Sub-1% E2E, so ROI is marginal.
On-box edits live in `/sgl-workspace/aiter` on `glm5`@m12-17 (`.bak` saved for the header); nothing
committed.
