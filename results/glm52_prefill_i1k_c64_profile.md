# GLM-5.2 prefill — i1k/o1k conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

ms/prefill forward. MI355X ~383 (fp8-tuned proj ‡, was ~390 bf16) vs B200 331 (0.86x). Graphs-off eager, dense-MHA at i1k, `bench_serving` conc64 num256 `--profile-num-steps 4 --profile-by-stage`, EXTEND trace n_fwd=4. MI355X = `v0.5.15-rocm720-mi35x-20260711`, tuned MoE confirmed executing. All-reduce row carried over from prior clean capture (fresh QuickReduce profiler-perturbed).


| Section             | MI355X kernel                                  | MI355X ms  | B200 kernel                                   | B200 ms   | B200/MI355X |
| ------------------- | ---------------------------------------------- | ---------- | --------------------------------------------- | --------- | ----------- |
| Attention           | `ck_tile FmhaFwd` dense MHA                    | 27.9       | `fmhaSm100f ...H256...Causal`                 | 9.9       |             |
| Attention           | `set_mla_kv_buffer` + concat/cast              | 10.1       | `set_mla_kv_buffer` + `float8_copy` + concat  | 11.0      |             |
| Attention           | q/k-norm                                       | 2.0        | `LayerNorm`+`RMSNorm`+`fused_rope`            | 4.2       |             |
| Attention           | DSA indexer                                    | 0.3        | `fast_hadamard`+`fused_store_indexer_cache`   | 1.1       |             |
| **Attention**       |                                                | **40.4**   |                                               | **26.3**  | **0.65x**   |
| MoE gate-up (gemm1) | `mfma_moe1 t128x128x256`                       | 38.6       | `bmm_E2m1`                                    | 32.1      | 0.83x       |
| MoE down (gemm2)    | `mfma_moe2 t64x256`                            | 46.7       | `bmm_Bfloat16_E2m1`                           | 26.2      | 0.56x       |
| MoE combine         | `moe_reduction_kernel`                         | 22.6       | `finalizeKernelVecLoad`                       | 15.8      | 0.70x       |
| MoE act quant       | `dynamic_per_group_scaled_quant`               | 4.2        | `NVFP4Quantize`                               | 2.6       | 0.62x       |
| **MoE**             |                                                | **112.2**  |                                               | **76.7**  | **0.68x**   |
| Dense GEMM          | o_proj `a8w8_blockscale_bpreshuffle` (fp8 ‡)   | 27.9       | `nvjet_sm100 128x256`                         | ≈40       |             |
| Dense GEMM          | q_a+kv_a Tensile `MT224x256/240x256` (bf16)    | 26.1       | `nvjet_sm100 176x128`                         | ≈22       |             |
| Dense GEMM          | q_b_proj `a8w8_blockscale_bpreshuffle` (fp8 ‡) | 12.2       | `nvjet_sm100 256x128`                         | ≈12       |             |
| Dense GEMM          | kv_b Tensile `MT256x256/256x240` (bf16)        | 8.6        | —                                             | —         |             |
| Dense GEMM          | router `hgemm`+`MT256x256` (bf16)              | 5.6        | —                                             | —         |             |
| Dense GEMM          | DenseMLP L0-2 (bf16)                           | 3.0        | —                                             | —         |             |
| **Dense GEMM**      |                                                | **83.3**   |                                               | **73.7**  | **0.89x**   |
| All-reduce          | `quickreduce twoshot` (INT4)                   | 109.9      | `ncclDevKernel ...RING_LL`                    | 68.6      |             |
| All-reduce          | `aiter::cross_device_reduce_2stage`            | 20.8       | `mnnvl twoshot` + `rmsNormLamport` + one-shot | 70.9      |             |
| **All-reduce**      |                                                | **130.7**  |                                               | **139.5** | **1.07x**   |
| RMSNorm             | `aiter::add_rmsnorm_quant`                     | 14.3       | `fused_add_rmsnorm`                           | 14.7      | 1.03x       |
| **TOTAL**           |                                                | **~383** ‡ |                                               | **331**   | **0.86x**   |


‡ Dense GEMM section is the **fp8-tuned proj config**: `SGLANG_DSA_FP8_PROJ_GEMM=1` converts the 128-aligned `q_b_proj`/`o_proj` to fp8 `a8w8_blockscale_bpreshuffle`, **tuned** for GLM shapes (q_b_proj 4096,2048 / o_proj 6144,4096). vs bf16-off: o_proj 45.9→27.9, Dense GEMM total 92.1→83.3 (−8.8ms/−9.5%). Adds +2.3ms fp8 act-quant (o_proj input, not in GEMM total) → net dense-side −6.5ms. Tuning is mandatory (untuned = +8.8ms regression). Other sections are the fp8-off baseline capture.

## Status / levers


| Area                                   | MI vs B200 (ms) | ratio | status                                 | result / next                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------------------------------- | --------------- | ----- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dense GEMM `MT256x240/240x256/224x256` | 46.0 vs ≈22     | 0.48x | **UNEXPLORED — biggest untouched gap** | ragged Tensile tiles (240/224 = non-256-aligned N); try tile/pad/backend                                                                                                                                                                                                                                                                                                                                                                                                         |
| MoE gemm2                              | 46.7 vs 26.2    | 0.56x | CK re-test BLOCKED (upstream CK bug)   | new mxfp4 MoE pipeline + its compile-breaking loader landed in same CK commit #8260; won't build thru develop tip. FlyDSL wins/deployed; see detail                                                                                                                                                                                                                                                                                                                              |
| MoE gemm2↔combine fusion               | —               | —     | DEAD                                   | no CDNA4 PDL overlap; fused megakernel 5.6x slower                                                                                                                                                                                                                                                                                                                                                                                                                               |
| MoE combine                            | 22.6 vs 15.8    | 0.70x | HBM-bound, no lever                    | `moe_reduction` 366us@M16384 ~5.5 TB/s; combine fusion DEAD (gemm2+combine megakernel 5.6x slower)                                                                                                                                                                                                                                                                                                                                                                               |
| MoE act quant                          | 4.2 vs 2.6      | 0.62x | HBM-bound, no lever                    | `dynamic_per_group_scaled_quant` 30us@M16384 ~6.3 TB/s. Fusion into gemm1 TESTED = net LOSS: fp4q stage1 1615.7us vs bf16 1569.5 (+46us epilogue quant) > standalone quant 30us. gemm1 is VALU-saturated so in-epilogue group-max+pack has no free cycles. Tuner's bf16+separate-quant is correct.                                                                                                                                                                               |
| Attention FA (kernel)                  | 27.9 vs 9.9     | 0.35x | CLOSED                                 | pin D256 `(128,128)` new-CK = 421us (−5.4%); rest architectural (B200 TMEM); DSA indexer = required decode-cache, not a lever — see detail                                                                                                                                                                                                                                                                                                                                       |
| MoE gate-up (gemm1)                    | 38.6 vs 32.1    | 0.83x | VALU-bound, not MFMA                   | MFMA 37% / VALU 100% @ M16384 (637.9us); native mxfp4 MFMA already used; bottleneck = per-MFMA fp4 packing + silu_mul; see detail                                                                                                                                                                                                                                                                                                                                                |
| Dense GEMM fp8 proj                    | —               | —     | **WIN after tuning**                   | Tuned a8w8_blockscale_bpreshuffle for q_b_proj(4096,2048)+o_proj(6144,4096) M∈{1k..64k}, then fp8-ON e2e (GPU4-7): Dense GEMM **83.3ms vs 92.1 bf16 = −8.8ms/−9.5%**. o_proj 45.9→**27.9** (−39%, the whole swing); q_b_proj flat 12.2. +2.3ms fp8 act-quant → still −6.5ms net. Untuned it was +8.8ms REGRESSION — tuning is mandatory. DEPLOY: `SGLANG_DSA_FP8_PROJ_GEMM=1` + carry tuned config (box-local in model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv). |
| Router GEMM                            | —               | —     | no lever                               | B200 fuses; standalone 547→1196 TF                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Dense GEMM `MT256x256`                 | 27.7 vs ≈40     | 1.4x  | MI faster                              | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| All-reduce                             | 130.7 vs 139.5  | 1.07x | parity                                 | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| RMSNorm                                | 14.3 vs 14.7    | 1.03x | parity                                 | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Dense-attn fallback                    | —               | —     | DONE `3235a7d271`                      | sparse≈764→dense≈28 ms/fwd; GSM8K 0.955; TTFT −14–32%                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Tuned MoE dispatch                     | —               | —     | DONE (box-local)                       | `glm5_fp4` gfx950/cu256 rows injected; lost on rebuild                                                                                                                                                                                                                                                                                                                                                                                                                           |




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

Latency-bound ~888us: MFMA 115 + dequant 231 + `[M,topk,D]` write 226; ~316us unhidable stall (13% MFMA util). FlyDSL levers wash/regress (pipe3 inert nk<3, full-K +31%, atomic +63%).
New-CK re-test blocked: upstream CK bug (rocm-libraries #9400) — #8260 shipped the reworked mxfp4 MoE pipeline + a compile-breaking static-only loader in one commit; broken thru develop tip. Last compilable CK = old pipeline = prior verdict (FlyDSL wins all buckets). A win needs the upstream fix or a from-scratch CK device-op.

### MoE gemm1 gate-up detail (K=6144, fused silu_mul, M=16384 microbench)

`mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd` = 637.9us, VGPR=160 (~3 waves), LDS=82KB. Kernel = `mixed_moe_gemm_2stage.py`.
**MFMA util 37% / VALU util 100%** → VALU-bound, not MFMA-peak-bound (MFMA idle 63%).
Already uses gfx950 **native mxfp4 MFMA** (`mfma_scale_f32_16x16x128_f8f6f4`, K=128, e8m0 scale applied in hardware) — so the VALU cost is NOT dequant or scale-apply. VALU bottleneck = per-MFMA fp4 input packing (`pack()` → i128 bit-ops every MFMA in the K-loop) + fused silu_mul epilogue. Confirmed structural: gemm2 (down) is even more VALU-bound (13% MFMA) with no activation → common cost = fp4 packing/marshalling around the native MFMA.
Lever = cut/overlap the per-MFMA fp4 packing + silu_mul VALU work; FlyDSL tuning levers already washed (see gemm2). Gap is structural vs B200 tensor-core-integrated fp4 epilogue.

**MoE GEMM section: NOT fully closed.** No viable near-term lever. Both remaining avenues are out of near-term reach: (1) a from-scratch CK mxfp4 MoE device-op (large effort), and (2) the CK native path, which is blocked on upstream bug #9400 with no fix ETA. FlyDSL is at its floor (native MFMA already used; VALU-marshalling-bound). Revisit if #9400 is fixed upstream or a from-scratch CK op is prioritized.