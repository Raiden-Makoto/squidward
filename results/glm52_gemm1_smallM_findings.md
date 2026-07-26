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

### EXACT fix site (2026-07-25, ground truth)

MFMA `mfma_scale_f32_16x16x128f8f6f4`: `k_per_blk=32`, `num_input_blks=4`. Derived:
`APackedSize=2`, `KPack=max(16,32/2)=16`, `NLane=16`, `KLane=64/16=4`,
`KRepeat=KPerBlock/(KLane*KPack)=KPerBlock/64` → **K=128→2, K=256→4**; `KXdlPack=2`.

The scale-consume loop is in `blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_gufusion_v3.hpp`
(b_scale prefetch ~L418-437, plus the two `b_scale_thread_copy_up` copies and the hotloop repeats):
```
for n0 in NRepeat/NXdlPack:
  for k0 in KRepeat/KXdlPack:         // =1 at K=128, =2 at K=256
     b_scale_thread_copy.Run(..., make_tuple(n0,k0,I0), ...)
     b_scale_thread_copy.MoveSrcSliceWindow(desc, make_multi_index(0, I1, 0))  // midK += 1
  b_scale_thread_copy.MoveSrcSliceWindow(desc, make_multi_index(NWaves, -KRepeat/KXdlPack, 0))
```
The inner `k0` loop steps the descriptor midK dim by **+1** per iteration. `b_scale_grid_desc_bn_ak`
(gridwise ~L1267) has midK stride = `64*KXdlPack*NXdlPack/scale_pack_size_b = 256/spb`, while the
n32k4 host layout's `remain_k` stride is **128**. They match only if `spb==2`. If `spb==1` (B-scale
is raw `e8m0_bexp_t`, 1 byte — likely), midK step = 256 skips every other `remain_k`, so at K=256
(k0 loops twice) the 2nd super-block is read from `remain_k=2` instead of `1` → wrong scales →
logits_diff ~1.0. K=128 (k0 loops once) never takes the second step, so it's unaffected — matches the
symptom exactly.

NEXT: confirm `spb` (BScaleDataType size), then fix the midK step to advance by the true `remain_k`
stride (or fix the descriptor midK stride) in the b_scale/b_scale_up prefetch AND hotloop repeats of
the gufusion_v3 pipeline. Then rebuild the fp4-silu ck2stages module in `aiter_dev` and check
logits_diff at M=1024 (expect ~1e-3 fp4 noise on success).

WORKFLOW (corrected): edits belong in **our fork** `/home/macui/aiter`
(`RM/glm52-mxfp4-moe-ck-gemm1`); test on box via `/sgl-workspace/aiter_dev` scratch (copy of image
aiter + fork edits, `PYTHONPATH=/sgl-workspace/aiter_dev`), leaving `/sgl-workspace/aiter` pristine.
Fork rebased onto upstream `6c48c5fa0` + CK submodule checked out `af7118e34`.

### VALIDATION SIGNAL (2026-07-25, critical)

The microbench `test_moe_2stage.py` csv-path `logits_diff`/`checkAllclose` is **INVALID for the
ck2stages path** — the known-good K=128 CK kernel (production-correct, e2e GSM8K 0.929) ALSO shows
`logits_diff ~1.0`, err~0.997, `checkAllclose failed` in that harness. So `logits_diff` from the
csv-path cannot distinguish correct from broken ck2stages kernels. **Do NOT use it.** The only
trustworthy signal is **e2e GSM8K** on the live server (K=128 baseline = 0.929).

### K=256 STATUS: confirmed BROKEN e2e

Deployed the K=256 tile e2e (server on `aiter_dev` via `PYTHONPATH`, `AITER_CONFIG_FMOE` routing the
CK gemm1 buckets to `...256x128x128x256...`, GPUs 4-7). **GSM8K = 0.118 / Invalid 0.348** (vs K=128
0.929). So K=256 gemm1 is genuinely numerically wrong at prod scale — the perf win (4-53% isolated)
is not usable until fixed.

ELIMINATED as the cause (do not re-chase):
- **Scales:** gfx950 uses arch-agnostic `shuffle_scale` (permute path, NOT the gfx1250 n32k4). With
  scale-context `BPackedSize=1`: descriptor midK count=24/stride=256, dim0 stride=6144 = per-super-row
  — descriptor matches the shuffle exactly. Scale path is correct at K=256.
- **async_vmcnt waitcnt** (`gufusion_v3.hpp` L167-169, L463/679/879): forcing full `s_waitcnt(0)`
  did NOT change the result. Not the bug.

REMAINING suspects for the `KRepeat` 2→4 (K=128→256) break: B-weight bpreshuffle read
(`b_blockwise_copy`, KRepeat in dim3) at KRepeat=4; A-block LDS double-buffering at KPerBlock=256;
or the MFMA accumulation order across the extra KRepeat. Iterate via: edit fork → rsync/base64 to
`aiter_dev` → rebuild fp4-silu ck2stages module → redeploy server → GSM8K (~8 min/cycle; slow but the
only valid signal until a K256-vs-K128 output-diff microbench is built).

Staged: fork WIP `a899075d0` (KPerBlock + K=256 tiles); notes `a16fc70898`.

### FAST valid repro (2026-07-25) — use this to iterate, NOT the bogus allclose

The harness kernel-vs-reference is bogus, but kernel-vs-**kernel** is valid: K=128 CK is
production-correct, so compare K=256 output to K=128 output on identical seeded inputs.
Instrumented `op_tests/test_moe_2stage.py` (in `aiter_dev`): `torch.manual_seed(MB_SEED)` before inputs,
and when `MB_DUMP` set + shape==(6144,512) print `MBSTAT tok=.. sum=.. absmean=.. std=.. nan=..` for
`out2_ck`. Run WITHOUT `--kernel` (kernel-bench mode returns before out2_ck), WITH
`--csv-filter moe_ck2stages`, `AITER_CONFIG_FMOE=<csv routing 6144/512 rows to the CK kernel>`,
`HIP_VISIBLE_DEVICES=0`, `MB_SEED=1234`. ~10s/run, no server.

Result at tok=1024 (6144,512, seed 1234):
- K=128 (correct): sum=`4.593536e7`, absmean=`2.755e4`, std=`3.467e4`
- K=256 (broken):  sum=`-1.411e8`,   absmean=`3.169e4`, std=`3.990e4`

So K=256 gemm1 is numerically wrong even at M=1024 (not just large M), consistent with GSM8K 0.118.
gemm2 is flydsl (unchanged) so the divergence is purely the K=256 gemm1. Target: K=256 MBSTAT must
match the K=128 numbers. Kernel edits still need the ~108s fp4-module rebuild in `aiter_dev`, but the
test itself is ~10s. Remaining suspects (scales+waitcnt already eliminated): B-weight bpreshuffle
read / a_block LDS / MFMA loop at KRepeat=4 (K=256 halves num_loop 48→24, KRepeat 2→4).

### BISECTION RESULTS (2026-07-25) — narrowed to shared A operand

Using the raw-accumulator MBSTAT diagnostic (patch epilogue L1652: `c_thread_buf_fp32 =
c_thread_buf[cidx]` for raw gate / `c_thread_buf_up[cidx]` for raw up), M=1024 6144,512 seed 1234:
- Raw GATE: K128 sum=2.07e6 absmean=506.9; K256 sum=3.61e5 absmean=537.1 → **differ** → bug is in
  the GEMM reduction, NOT the silu/up epilogue.
- Raw UP:   K128 sum=-8.64e5 absmean=505.2; K256 sum=2.55e6 absmean=536.9 → **also differ**.
Both gate AND up wrong, same ~6% absmean inflation, scrambled sums. gate/up share the same
`a_thread_vec` (A operand) + the same K-accumulation loop but use different B → the symmetric error
implicates the **shared A operand read / K-accumulation double-buffer state machine** at KRepeat=4
(a_block LDS `scale_comp_buf`/`scale_mem_buf` toggle, `SwitchM` reload, `LocalPrefetchStages=2`),
NOT a per-B issue. model_dim=256 (num_loop=1) is unsupported so can't isolate single-block.
Next: inspect/perturb the A double-buffer + SwitchM logic for a KRepeat=4 off-by-one.

Diagnostic is in `aiter_dev` gridwise (MBDIAG comment, 2 spots) — revert before any real build.

### DEVICE PRINTF WORKS (2026-07-25) — earlier failures were placement/guard/cache, not a limit

Standalone HIP `printf` prints fine; and an unconditional `printf` at the gridwise `Run` entry
prints in the aiter kernel too (`KERNEL_RAN KPerBlock=256 KRepeat=4`). Earlier "no output" was
because probes sat after an early `return`, had a non-matching `blockIdx.y` guard, or hit a stale
module. So the kernel CAN be instrumented — use `printf` guarded by
`threadIdx.x==0 && blockIdx.x==0 && KPerBlock==256`, placed BEFORE the `expert_block_id ... return`.

### READ-SIDE PROVEN CORRECT via printf — bug is in the KRepeat 2,3 operand FILL

At K=256 (KRepeat=4, KPack=16, KPerXdlops=128, K1PerXdlops=32, num_input_blks=4):
- A LDS read offsets (local prefetch `a_k_step_chunk`): k=0→0, 1→64, 2→128, 3→192, aBlkK=256 —
  evenly span the 256-K block, no overflow/alias.
- a/b thread-buffer offsets per KRepeat: 0/16/32/48 — distinct, no alias.
So indexing, buffer sizing, and scales are all correct. The symmetric gate+up error therefore comes
from the DATA loaded into a_block LDS (a_blockwise_copy A global→LDS) or b_thread_buf
(b_blockwise_copy B global→thread) for the 2nd 128-K super-block (KRepeat 2,3).

### ROOT CAUSE (2026-07-25): A-activation global→LDS fill at KPerBlock=256

Const-operand bisection (raw-gate MBSTAT, M=1024, seed 1234), forcing the MFMA operands to fixed
fp4 bytes to isolate which input is wrong:
- **const-B only** (real A): K128 sum=-1.62e6 vs K256 sum=-1.14e5 → **DIFFER** → not (solely) B.
- **const-A + const-B** (only scales left live): K128 sum=-4.677e7 vs K256 sum=-4.628e7,
  absmean 9937 vs 9931 → **MATCH** (~1% = fp/e8m0 rounding) → scales + KRepeat=4 accumulation are
  CORRECT.
Therefore the K=256 correctness bug is the **A operand CONTENT** — `a_block` LDS is filled with wrong
A activations for the 2nd 128-K super-block at KPerBlock=256 (the `a_thread_copy` read offsets
0/64/128/192 are correct, the a_block contents at 64+ are not). Locus: `a_blockwise_copy` (A global→
LDS, ThreadGroup transfer with `ABlockTransferThreadClusterLengths S<K0_A,K0_M_A,1>`; K0_A=16 at
K256) and/or the a_block LDS descriptor `GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1`. B content,
scales, MFMA accumulation, thread/LDS read offsets, buffer sizing all VERIFIED CORRECT and eliminated.

FIX DIRECTION: correct the A activation LDS staging for KPerBlock=256 (a_blockwise_copy cluster/vector
or a_block descriptor). May be a genuine rework since this MoE gridwise's A-LDS staging appears
designed/validated only for K=128 blocks. Sub-1% E2E, so weigh vs shipping the mixed-M routing.
