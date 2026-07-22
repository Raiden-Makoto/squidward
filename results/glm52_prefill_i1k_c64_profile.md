# GLM-5.2 prefill — i1k/o1k conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4). MI355X ~376 (fp8-proj prefill-only gated ‡) vs B200 331 (0.88x). Graphs-off eager, dense-MHA at i1k, `bench_serving` conc64 num256 (i1k in, o16 — prefill only) `--profile-num-steps 4 --profile-by-stage`. Reprofiled 2026-07-15 on the current stack (`RM/glm51`: prefill-only fp8-proj gating + fused hadamard-quant #30715 + fp8 MLA absorbed-bmm #30519, tuned MoE confirmed executing), image `v0.5.15-rocm720-mi35x-20260711`. All-reduce carried from prior clean capture (fresh profiled QuickReduce ~108ms is profiler-perturbed).


| Section             | MI355X kernel                                  | MI355X ms  | B200 kernel                                   | B200 ms   | B200/MI355X |
| ------------------- | ---------------------------------------------- | ---------- | --------------------------------------------- | --------- | ----------- |
| Attention           | `ck_tile FmhaFwd` dense MHA                    | 27.3       | `fmhaSm100f ...H256...Causal`                 | 9.9       |             |
| Attention           | `set_mla_kv_buffer` + concat/cast              | 6.2        | `set_mla_kv_buffer` + `float8_copy` + concat  | 11.0      |             |
| Attention           | q/k-norm + rope                                | 5.9        | `LayerNorm`+`RMSNorm`+`fused_rope`            | 4.2       |             |
| Attention           | DSA indexer                                    | 0.3        | `fast_hadamard`+`fused_store_indexer_cache`   | 1.1       |             |
| **Attention**       |                                                | **39.7**   |                                               | **26.3**  | **0.66x**   |
| MoE gate-up (gemm1) | `mfma_moe1 t128x128x256`                       | 38.0       | `bmm_E2m1`                                    | 32.1      | 0.85x       |
| MoE down (gemm2)    | `mfma_moe2 t64x128x256`                        | 47.5       | `bmm_Bfloat16_E2m1`                           | 26.2      | 0.55x       |
| MoE combine         | `moe_reduction_kernel`                         | 22.3       | `finalizeKernelVecLoad`                       | 15.8      | 0.71x       |
| MoE act quant       | `dynamic_per_group_scaled_quant`               | 4.0        | `NVFP4Quantize`                               | 2.6       | 0.65x       |
| **MoE**             |                                                | **111.8**  |                                               | **76.7**  | **0.69x**   |
| Dense GEMM          | o_proj `a8w8_blockscale_bpreshuffle` (fp8 ‡)   | 26.9       | `nvjet_sm100 128x256`                         | ≈40       |             |
| Dense GEMM          | q_a+kv_a Tensile `MT224x256/240x256` (bf16)    | 25.2       | `nvjet_sm100 176x128`                         | ≈22       |             |
| Dense GEMM          | q_b_proj `a8w8_blockscale_bpreshuffle` (fp8 ‡) | 12.0       | `nvjet_sm100 256x128`                         | ≈12       |             |
| Dense GEMM          | kv_b Tensile `MT256x256/256x240` (bf16)        | 7.8        | —                                             | —         |             |
| Dense GEMM          | router `hgemm`+`MT32x32` (bf16)                | 4.7        | —                                             | —         |             |
| Dense GEMM          | DenseMLP L0-2 (bf16)                           | 2.9        | —                                             | —         |             |
| **Dense GEMM**      |                                                | **79.5**   |                                               | **73.7**  | **0.93x**   |
| All-reduce          | `quickreduce twoshot` (INT4)                   | 109.9      | `ncclDevKernel ...RING_LL`                    | 68.6      |             |
| All-reduce          | `aiter::cross_device_reduce_2stage`            | 20.8       | `mnnvl twoshot` + `rmsNormLamport` + one-shot | 70.9      |             |
| **All-reduce**      |                                                | **130.7**  |                                               | **139.5** | **1.07x**   |
| RMSNorm             | `aiter::add_rmsnorm_quant`                     | 14.0       | `fused_add_rmsnorm`                           | 14.7      | 1.05x       |
| **TOTAL**           |                                                | **~376** ‡ |                                               | **331**   | **0.88x**   |


‡ Dense GEMM is the **fp8-proj prefill-only gated config**: `SGLANG_DSA_FP8_PROJ_GEMM=1` runs the 128-aligned `q_b_proj`/`o_proj` on fp8 `a8w8_blockscale_bpreshuffle` **for prefill (M>512)** while decode (M≤512, cuda-graph batch sizes) stays bf16 (`unquant.py` M-gate). Prefill here uses the tuned fp8 (q_b_proj 4096,2048 / o_proj 6144,4096): o_proj 26.9, q_b_proj 12.0. +~2ms fp8 act-quant on the o_proj input (not in the GEMM total). Tuning is mandatory (untuned fp8 = regression). Gating is decode-only, so the prefill profile is unchanged from the pre-gating fp8-tuned capture (Dense GEMM 83.3→79.5 here is measurement variance).

## Status / levers


| Area                                      | MI vs B200 (ms) | ratio | status                                   | result / next                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------------- | --------------- | ----- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dense GEMM `MT224x256`+`MT256x240` (bf16) | 33.0 vs ≈22     | 0.67x | **AT VENDOR FLOOR — no lever**           | q_a+kv_a (N2624/K6144, 25.2) + kv_b (7.8). Full re-tune @M16384 (all backends + hipBLASLt): best = hipBLASLt `MT256x256` 408us/1294 TFLOPS/783 GB/s = the deployed torch-native. N-padding 2624→2816 is WORSE (450 vs 396us). flydsl broken here (19 solns error). q_a+kv_a can't go fp8 (N2624 not 128-aligned). Gap is vendor-library/architectural; only path = from-scratch kernel beating hipBLASLt (large, uncertain).                                     |
| MoE gemm2                                 | 47.5 vs 26.2    | 0.55x | no CK lever (CK port ≈ FlyDSL)           | Real CK candidate = port `flydsl_mxmoe_g2` (`mfma_moe2`) ≈ FlyDSL (@16k 1167 vs 1176us; +15/−1/+4% @8k/16k/32k). No CK speedup for gemm2. glm5 deploys FlyDSL only because CK-port aux-quant not codegen'd for `(257,6144,512,9)` (missing from `moe_aux/gen_instances.py SHAPES`) — codegen gap, not perf. 0.55x is the FlyDSL-vs-B200 gap (architectural), unchanged by CK. Prior 2.4×/2855us/occupancy-rewrite = LEGACY bypass kernel, disregard. See detail. |
| MoE gemm2↔combine fusion                  | —               | —     | DEAD                                     | no CDNA4 PDL overlap; fused megakernel 5.6x slower                                                                                                                                                                                                                                                                                                                                                                                                               |
| MoE combine                               | 22.3 vs 15.8    | 0.71x | HBM-bound, no lever                      | `moe_reduction` 366us@M16384 ~5.5 TB/s; combine fusion DEAD (gemm2+combine megakernel 5.6x slower)                                                                                                                                                                                                                                                                                                                                                               |
| MoE act quant                             | 4.0 vs 2.6      | 0.65x | HBM-bound, no lever                      | `dynamic_per_group_scaled_quant` 30us@M16384 ~6.3 TB/s. Fusion into gemm1 TESTED = net LOSS: fp4q stage1 1615.7us vs bf16 1569.5 (+46us epilogue quant) > standalone quant 30us. gemm1 is VALU-saturated so in-epilogue group-max+pack has no free cycles. Tuner's bf16+separate-quant is correct.                                                                                                                                                               |
| Attention FA (kernel)                     | 27.3 vs 9.9     | 0.36x | CLOSED                                   | pin D256 `(128,128)` new-CK = 421us (−5.4%); rest architectural (B200 TMEM); DSA indexer = required decode-cache, not a lever — see detail                                                                                                                                                                                                                                                                                                                       |
| MoE gate-up (gemm1)                       | 38.0 vs 32.1    | 0.85x | no occupancy lever (VGPR disproven)      | new-CK gemm1 beats FlyDSL −5→−7.6% at M≥8192, loses M≤4096; numerically ==FlyDSL. **Small-M loss is NOT occupancy: forced 2/3/4-wave tiles all ≈232us @1024 — VGPR lever disproven, low-VGPR tiles dead.** See detail                                                                                                                                                                                                                                            |
| Dense GEMM fp8 proj                       | —               | —     | **WIN — prefill-only gate, upstreaming** | Tuned a8w8_blockscale_bpreshuffle q_b_proj(4096,2048)+o_proj(6144,4096): Dense GEMM −8.8ms/−9.5% (o_proj 45.9→27.9). Prefill-only M-gate (M>512 fp8, decode bf16); GSM8K 0.943. Untuned = regression (tuning mandatory). Upstreaming: sglang PR `RM/fp8-proj-gemm-prefill-gate` + tuned config aiter#4243.                                                                                                                                                       |
| Router GEMM                               | —               | —     | no lever                                 | B200 fuses; standalone 547→1196 TF                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Dense GEMM `MT256x256`                    | 27.7 vs ≈40     | 1.4x  | MI faster                                | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| All-reduce                                | 130.7 vs 139.5  | 1.07x | parity                                   | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| RMSNorm                                   | 14.0 vs 14.7    | 1.05x | parity                                   | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Dense-attn fallback                       | —               | —     | DONE `3235a7d271`                        | sparse≈764→dense≈28 ms/fwd; GSM8K 0.955; TTFT −14–32%                                                                                                                                                                                                                                                                                                                                                                                                            |
| Tuned MoE dispatch                        | —               | —     | DONE (box-local)                         | `glm5_fp4` gfx950/cu256 rows injected; lost on rebuild                                                                                                                                                                                                                                                                                                                                                                                                           |




### Attention FA detail

D256 tile sweep, new CK `cb859854a`, b16 s1024 nh16, `module_mha_varlen_fwd`:


| tile · warps·mfma               | VGPR | waves | fwd_us  |
| ------------------------------- | ---- | ----- | ------- |
| **(128,128) 4w 32×32** (pinned) | 236  | ~2    | **421** |
| (128,64) 4w 32×32 (new-CK auto) | 208  | ~2    | 490     |
| (128,64) 8w 16×16               | 84   | 3+    | 620     |
| (128,32) 8w 16×16               | 76   | 3+    | 762     |
| (128,128) old-CK baseline       | 256  | ~2    | 444.9   |


MFMA-throughput bound: biggest valid tile wins, more waves = slower (occupancy disproven). `(128,128)` is max valid (`bn0≥192`/`bk0≥64` don't compile). Pinned in `aiter_dev` CK submodule.

**DSA indexer (0.3ms) is NOT a lever — it's required decode-cache work.** Under the dense-MHA fallback the indexer runs with `return_indices=False` and auto-skips top-k (dense regime `max_kv_len ≤ 2048` == `dsa_index_topk` 2048 → top-k is identity). What remains is `_store_index_k_cache`: it writes the compressed index-K cache for the prefilled tokens, which is mandatory because DECODE uses sparse MLA and scores every decode query against the index-K of all prior tokens (incl. these). So the 0.3ms is irreducible cache population, not sparse-selection waste.

**Attention section: CLOSED.** Dense FA kernel at its floor (`(128,128)` 421us, MFMA-bound, occupancy disproven; B200 gap architectural = TMEM). Sparse/DSA path irrelevant at i1k (dense fallback, 18× faster than the old sparse `_sparse_mla_fwd` 515us). Indexer is required decode-cache work, not a lever.

### MoE gemm2 detail (nk=2 per-rank, TP4 inter_dim→512)

**Real CK gemm2 candidate = the PORT kernel** `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` **(**`mfma_moe2`**), and it ≈ FlyDSL** (`test_moe_2stage --kernel`, prod 6144,512 a4w4, GPU0, logits_diff 5.9e-4):


| M     | CK port gemm2 (`mfma_moe2`) | FlyDSL gemm2 | Δ    | CK legacy bypass (non-candidate) |
| ----- | --------------------------- | ------------ | ---- | -------------------------------- |
| 8192  | 708                         | 615          | +15% | 1433                             |
| 16384 | 1167                        | 1176         | −1%  | 2855                             |
| 32768 | 2257                        | 2165         | +4%  | 5689                             |


CK port gemm2 is at parity with FlyDSL (−1% @16k). Companion port gemm1 @16k = 684us ≈ FlyDSL 678 (the −7% gufusion_v3 win needs the tuned gemm1 kname; the throwaway CSV above used a non-optimal one).

**Why glm5 deploys FlyDSL, not the CK port = codegen gap, not perf.** The CK port (`output_aux`) needs a per-shape codegen'd aux-quant instance; glm5 `(NE257, H6144, INTER512, TOPK9)` is absent from `SHAPES` in `csrc/kernels/mxfp4_moe/moe_aux/codegen/gen_instances.py` (has kimik2/dsv4/qwen, not glm5) → `RuntimeError: no codegen'd instance aux_quant_NE257_TOPK9_MB128_H6144`. Adding `(257,6144,512,9)` + rebuilding the aux module makes the port run at the numbers above.

**CORRECTION (supersedes any earlier 2.4×/2855us/21%-occupancy/short-K/persistent-rewrite notes):** all of that profiled the LEGACY bypass kernel (`kernel_moe_mxgemm_2lds`, only reached via `AITER_BYPASS_TUNE_CONFIG`) — never deployed, not the candidate. Disregard. No CK gemm2 rewrite needed: the CK port already matches FlyDSL. The split-K `KBatch` TODO lives in that legacy path and is irrelevant to the deployed/candidate kernels.

### MoE gemm1 gate-up detail (prod shape 6144,512 per-rank, a4w4, per-call µs)

new-CK `gufusion_v3` vs FlyDSL `mfma_moe1 t128x128x256`, gemm1 only (`test_moe_2stage.py --kernel`, GPU0):


| M      | FlyDSL | new-CK | Δ     |
| ------ | ------ | ------ | ----- |
| 1024   | 146.8  | 198.9  | +35%  |
| 4096   | 273.7  | 305.9  | +12%  |
| 8192   | 417.0  | 396.2  | −5.0% |
| 16384  | 678.7  | 626.8  | −7.6% |
| 32768  | 1175.3 | 1089.7 | −7.3% |
| 131072 | 4233.2 | 4037.6 | −4.6% |


new-CK wins −5→−7.6% at prefill M≥8192 (numerically ==FlyDSL, logits_diff 1e-5), loses M≤4096. FlyDSL gemm1 itself VALU-bound: MFMA 37% / VALU 100% @16384; native mxfp4 MFMA (`mfma_scale_f32_16x16x128_f8f6f4`) already used, cost = per-MFMA fp4 packing + silu_mul.

#### CK gemm1 dispatch — HOW TO ACTUALLY RUN EACH PATH (do NOT re-derive — cost real hours twice)

a4w4 (fp4×fp4, per-1x32) has THREE gemm1 paths in `aiter/fused_moe.py`; the wrong invocation silently runs the wrong kernel:

1. **FlyDSL `mfma_moe1`** — the DEFAULT. `_flydsl_force = AITER_FLYDSL_FORCE` **defaults to `"1"`**, so `use_mxfp4_flydsl` (line ~2163) is True for a4w4 regardless of activation → FlyDSL wins before any CK branch. **Any "forced CK" run that does not set `AITER_FLYDSL_FORCE=0` silently runs FlyDSL.**
2. **CK-tile `moe_cktile2stages_gemm1`** (`csrc/ck_tile_gemm_moe_2stages`) — a8w8/a16w4 only; **raises `RuntimeError: Unsupported scales/output dtype!` for a4w4.** `AITER_FORCE_CK_GEMM1=1` mis-routes HERE (line ~2245) → that error. NOT the deployable CK; never use force-ck to benchmark a4w4.
3. **CK gufusion `ck_moe_stage1` → `aiter.ck_moe_stage1_fwd`** (`csrc/ck_gemm_moe_2stages_codegen`, `moe_ck2stages_gemm1_..._v3`) — the REAL a4w4 CK gemm1 (the numbers above). Reached at line ~2316 when `kernelName1` contains `ck2stages`.

**To dispatch real gufusion CK gemm1:** `AITER_FLYDSL_FORCE=0` + `AITER_CONFIG_FMOE` row with `kernelName1=moe_ck2stages_gemm1_...` + activation Silu; do NOT set `AITER_FORCE_CK_GEMM1`. **Verify with rocprofv3 `--kernel-trace`:** real CK shows an `mxgemm`/`ck2stages` kernel; if the trace shows only `mfma_moe1_...`, forcing failed and it ran FlyDSL. Pitfalls: (a) `AITER_CONFIG_FMOE` token must match the microbench's padded tier (`-t 1536` pads to 2048 → a `token=1536` row misses → default dispatch); (b) `--kernel` KBENCH µs are meaningless under rocprof (perftest perturbed) — use kernel-trace durations.

#### CK gufusion_v3 gemm1 tile VGPR (valid — real `_v3_` object metadata, gfx950 512 VGPR/SIMD, spill=0)


| tile (BLOCK×M×N×K_MxN) | accum fp32/lane = M·N/BLOCK | VGPR | waves/SIMD |
| ---------------------- | --------------------------- | ---- | ---------- |
| 256×32×128×128_1x4     | 16                          | 132  | 3          |
| 256×64×128×128_1x4     | 32                          | 186  | 2          |
| 256×128×128×128_1x4    | 64                          | 249  | 2          |
| 64×32×32×128_1x1       | 16                          | 134  | 3          |


VGPR ≈ 93 + 2.44·(accum/lane); occupancy cliffs ≤170=3w, ≤128=4w.

**INVALID — measured FlyDSL, not CK** (run WITHOUT `AITER_FLYDSL_FORCE=0`; kernel-trace showed `mfma_moe1_t32x128x256` only, mxgemm=0). Kept as a warning; re-run via the gufusion invocation above:


| M     | 256×64×128 (2w) | 256×32×128 (3w) | 256×32×64 (4w) |
| ----- | --------------- | --------------- | -------------- |
| 1024  | 235.6           | 231.5           | 232.1          |
| 4096  | 313.4           | 323.0           | 319.4          |
| 16384 | 687.9           | 676.1           | 686.1          |


All three "tiles" gave ≈232µs @1024 **because they were the same FlyDSL kernel, not CK**.

#### REAL CK gufusion gemm1 small-M tile sweep (trace-verified: mxgemm=50, mfma_moe1=0; `AITER_FLYDSL_FORCE=0`, GPU0)

| M    | CK 256×32×128 (132 VGPR, 3w) | CK 256×64×128 (186 VGPR, 2w) | FlyDSL |
| ---- | ---------------------------- | ---------------------------- | ------ |
| 512  | 169.2 | 171.5 | ~145  |
| 1024 | 224.1 | **171.2** | 146.8 |
| 2048 | 272.9 | 251.7 | ~193  |

Real CK tiles are NOT identical (unlike the invalid FlyDSL sweep). **Occupancy is NOT the small-M lever — now confirmed on the real kernel:** the higher-occupancy `256×32×128` (3w) is *slower* (224) than the lower-occupancy `256×64×128` (2w, 171) @1024. Bigger per-wave tile wins even at small M (matches the Attention-FA result). CK's best small-M tile `256×64×128` = 171µs @1024 vs FlyDSL 147 (**+17%**); the residual gap is CK's fixed pipeline/packing cost, not occupancy/tiling. Lowering VGPR / adding low-N tiles is dead. (`256×128×128` @M512 produced no KBENCH — block_m=128 at tiny M.)

#### root-cause = VALU-bound fp4 unpack; ALL config levers exhausted (2026-07-21)

PMC on real CK gemm1 `256×64×128` (GPU0, single-metric passes): **VALUUtilization 99% / MfmaUtil 8.6% @1024 (18% @8192) / MemUnitStalled ~1% / Occupancy ~29%.** Hard VALU-bound, MFMA idle, not memory/occupancy limited. `is_scale_mfma=true` → e8m0 scale applied in-HW by the scaled MFMA (scale is FREE). The VALU is the **fp4x2→MFMA-operand unpack** in the CK **v3 blockwise mx-moe pipeline** (`3rdparty/composable_kernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_selector.hpp`, via `gridwise_moe_mx_gemm_bpreshuffle.hpp`). FlyDSL wins small-M because its `pm1_async` unpack path is leaner.

Config/codegen levers — **all exhausted, do not retry:**
- Occupancy/VGPR tiles — disproven (higher-occupancy is slower).
- `KPerBlock` — the wrapper `gemm_moe_ck2stages_common_mxfp4.cuh` **hardcodes `128`** (the `KPerBlock` field is ignored); a `…x256` instance silently builds as K=128. Inert.
- Pipeline `v1` — **does not compile** for a4w4 gufusion (only `v3` builds; a `v1` instance breaks the whole module build). Closed.

**Only remaining lever = CK submodule source:** reduce the per-MFMA fp4 unpack VALU in the v3 blockwise mx-moe pipeline (LDS-stage operands / fewer unpack ops, matching FlyDSL `pm1_async`). Vendor-CK-kernel surgery, not a config change.