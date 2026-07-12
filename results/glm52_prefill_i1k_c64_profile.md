# GLM-5.2 prefill — i1k/o1k conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

GPU-busy per prefill forward: MI355X ~390 ms vs B200 331 ms → 0.85x. All kernel times are **ms per prefill forward pass** (graphs-off eager, dense-MHA at i1k, `--mem-fraction-static 0.85 --chunked-prefill-size 131072`; `bench_serving` i1k/o1k conc64 num256 `--profile-num-steps 4 --profile-by-stage`, EXTEND trace, n_fwd=4). MI355X = **fresh capture** on the `v0.5.15-rocm720-mi35x-20260711` image (`run_glm52.sh --profile`). Tuned MoE now **confirmed executing** on the workers (upstream `glm5_fp4` tiles injected into the box aiter default config — the run-script `AITER_CONFIG_FMOE` env does not reach SGLang scheduler workers): `mfma_moe1 t128x128x256`, `mfma_moe2 t64x256` (tuned tiles, not the untuned t64x128). **All-reduce row is carried over from the prior clean capture** — the fresh graphs-off capture's QuickReduce two-shot is profiler-perturbed (spin-wait; ~251 ms/fwd, ~1.9x the compute — excluded); clean re-capture pending. B200 = apples-to-apples capture (GLM-5.2-NVFP4, tp4 modelopt_fp4, kv fp8_e4m3, trtllm DSA, flashinfer_trtllm MoE, dense `fmhaSm100f...H256...Causal`).

| Section | MI355X kernel | MI355X ms | B200 kernel | B200 ms | B200/MI355X |
| --- | --- | ---: | --- | ---: | ---: |
| Attention | `ck_tile FmhaFwd` dense MHA (gfx950) | 27.9 | `fmhaSm100f ...H256...Causal` dense MHA | 9.9 | |
| Attention | `set_mla_kv_buffer` + concat/cast (KV-cache write) | 10.1 | `set_mla_kv_buffer` + `float8_copy` + concat `direct_copy` | 11.0 | |
| Attention | q/k-norm (`add_rmsnorm` + ck_tile `LayerNorm2d`) | 2.0 | flashinfer `LayerNorm`+`RMSNorm` + `fused_rope_kernel` | 4.2 | |
| Attention | DSA indexer (`fast_hadamard` + `indexer_k_quant`) | 0.3 | `fast_hadamard` + `fused_store_indexer_cache` | 1.1 | |
| **Attention subtotal** | | **40.4** | | **26.3** | **0.65x** |
| MoE gate-up GEMM (gemm1) | `mfma_moe1_silu_mul_afp4_wfp4 t128x128x256` | 38.6 | `bmm_E2m1` (swiGlu) | 32.1 | 0.83x |
| MoE down GEMM (gemm2) | `mfma_moe2_afp4_wfp4 cshuffle t64x256` | 46.7 | `bmm_Bfloat16_E2m1` | 26.2 | 0.56x |
| MoE combine/finalize | `moe_reduction_kernel` | 22.6 | `finalizeKernelVecLoad` (fuses routed-scale) | 15.8 | 0.70x |
| MoE act quant | `dynamic_per_group_scaled_quant` (mxfp4) | 4.2 | `NVFP4Quantize` | 2.6 | 0.62x |
| **MoE subtotal** | | **112.2** | | **76.7** | **0.68x** |
| Dense GEMM | Tensile `MT256x256` (proj) | 27.7 | `nvjet_sm100 128x256` | ≈40 | |
| Dense GEMM | Tensile `MT256x240 / 240x256 / 224x256` (proj) | 46.0 | `nvjet_sm100 176x128` | ≈22 | |
| Dense GEMM | `aiter::bf16gemm_256x256` (q_b_proj ‡) | 8.4 | `nvjet_sm100 256x128` | ≈12 | |
| Dense GEMM | `hgemm_bf16_128x128` (router) + shared-expert/misc tiles | 10.0 | — | — | |
| **Dense GEMM subtotal** | | **92.1** | | **73.7** | **0.80x** |
| All-reduce (carried over) | `quickreduce::allreduce_prototype_twoshot` (INT4 CodecQ4) | 109.9 | `ncclDevKernel_AllReduce ...RING_LL` | 68.6 | |
| All-reduce (carried over) | `aiter::cross_device_reduce_2stage` | 20.8 | `mnnvl twoshotAllreduce` + `rmsNormLamport` + one-shot | 70.9 | |
| **All-reduce subtotal** | | **130.7** | | **139.5** | **1.07x** |
| RMSNorm | `aiter::add_rmsnorm_quant` (norm+quant fused) | 14.3 | `fused_add_rmsnorm` (norm only; quant in MoE) | 14.7 | |
| **RMSNorm subtotal** | | **14.3** | | **14.7** | **1.03x** |
| **TOTAL prefill** | | **~390** | | **331** | **0.85x** |

‡ `q_b_proj` (ColumnParallelLinear, M×2048→4096/rank) and `o_proj` (RowParallelLinear, M×4096→6144/rank) are the two 128-aligned dense projections that `SGLANG_DSA_FP8_PROJ_GEMM=1` (gfx950, default off) converts from bf16 to the aiter FP8 CK GEMM. `fused_qkv_a` (M×6144→2624, not 128-aligned) and dense-MLP (layers 0-2) stay bf16.

## Levers

- **Dense-attention fallback (IMPLEMENTED, commit `3235a7d271`).** Extended the `use_mha` gate to gfx950; short prefill (`max_kv_len ≤ 2048`) routes to ck_tile dense FA instead of sparse-MLA. Gated by `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`. Sparse ≈764 ms/fwd → dense ≈28; GSM8K parity (0.955). TTFT −14–32%:

| conc | sparse TTFT | dense TTFT | Δ |
| --- | ---: | ---: | ---: |
| 4  | 277.3  | 196.9  | −29% |
| 8  | 467.6  | 366.8  | −22% |
| 16 | 872.9  | 593.9  | −32% |
| 32 | 1371.1 | 1183.6 | −14% |
| 64 | 1920.5 | 1640.6 | −15% |

- **Tuned MoE config dispatch (fixed):** `AITER_CONFIG_FMOE` set in `run_glm52.sh` does **not** reach SGLang scheduler workers (they get a curated env), so the workers staged aiter's default `configs/tuned_fmoe.csv` (no GLM gfx950 rows) and ran untuned. Fix: injected the upstream `glm5_fp4` gfx950/cu256 rows (explicit `gfx` col) into the box aiter default → workers now dispatch tuned tiles `mfma_moe1 t128x128x256` / `mfma_moe2 t64x256`. Box-local (lost on container rebuild).
- **All-reduce (1.07x):** MI355X INT4 QuickReduce 130.7 vs B200 139.5 — MI355X faster; not the gap.
- **Dense GEMM (0.80x):** MI355X 92.1 vs B200 73.7. FP8 proj lever (`SGLANG_DSA_FP8_PROJ_GEMM=1`, ‡) A/B run had an anomalous all-reduce; needs a clean redo.
- **Router GEMM: not an anomaly.** Standalone bench of the exact shape hits 547 TF (M=8k) → ~1196 TF (M=65k, ~peak); B200 fuses it into another GEMM. No lever.
- **RMSNorm+quant (fused):** parity (14.3 vs 14.7).
- **Attention (0.65x, dense FA 27.9 vs 9.9): register-pressure wall on CDNA4, no fork lever.** ck_tile FA = 256 VGPR / 0 AGPR → ~2 waves; latency-bound (14% compute / 44% BW). The `[BM,256]` fp32 accumulator needs ~256 VGPR (B200 offloads to TMEM, CDNA4 can't). Config levers all neutral-or-worse: tiling 1.00x, backend swap 1.01x, `waves_per_eu` 1→8 (`4811ac0906`) spills at wpe≥3.
  - *fp8 batch-prefill (`SGLANG_DSA_FP8_DENSE_ATTN`, default off; `c410a22111`/`9c966c21a0`/`6cea8561a2`): dead.* Kernel faster (1.13–1.40x), accuracy parity (0.928 vs 0.929), but per-forward q/k/v quant (bf16 `kv_b_proj`) makes e2e +6/+42/+25% at conc 4/8/16 and OOM at conc32+. Fused fp8-K/V needs MXFP4 `kv_b_proj`, structurally excluded for GLM (`SGLANG_FORCE_MXFP4_KVB`).
- **MoE (0.68x):** fresh tuned capture (upstream `glm5_fp4` tiles, confirmed executing).
  - **Down GEMM (46.7 vs 26.2, 0.56x) — NOT roofline-bound.** ~512 tokens/expert (16.4k tok × 8 / 256), K=512/rank → small grouped GEMM at only ~13% of the ~10 PFLOP/s MXFP4 compute peak (61.9 TFLOP/fwd/rank ÷ 46.7 ms ≈ 1.33 PFLOP/s); B200 ~26% of its NVFP4 peak (2.36 PFLOP/s). Memory-bound on the bf16 intermediate (`[tok×topk, 6144]` in HBM, ~80% of bytes): MI355X ≈41% of 8 TB/s HBM vs B200 ≈74%. The gap is kernel efficiency, not hardware — B200 round-trips the same intermediate but its cutlass grouped `bmm` hits far higher effective BW.
  - **Expert combine (22.6 vs 15.8, 0.70x):** the read side of that same HBM round-trip — gemm2 → HBM → reduce. B200 does **not** fuse it either (`finalizeKernelVecLoad` also reads the intermediate from HBM); it wins by (1) 128-bit vectorized packed loads + smem-staged scales/idx, and (2) PDL grid-dependency launch that overlaps the finalize with the gemm2 tail. Near-parity once B200's separate shared-add is counted (15.8 + 5.4 = 21.2). aiter FlyDSL-JIT (aiter-side); reference source vendored (`trtllm_fused_moe_dev_kernel.cu`).
  - gate-up GEMM (38.6 vs 32.1, 0.83x) / act-quant (4.2 vs 2.6): minor.

## Remaining gaps (future work)

1. **MoE down-GEMM efficiency** — gemm2 46.7 vs B200 26.2 (the ~20 ms prize). aiter grouped `mfma_moe2` runs at ~41% of HBM BW vs B200 cutlass `bmm` ~74%, across 256 tiny (M≈512) per-expert GEMMs → grouped-GEMM scheduling/occupancy (weight-load latency, tile utilization, persistent CTAs). A better aiter kernel, not a tile pick; not transcribable from the SM100 cutlass source. Hard, aiter/upstream.
2. **Combine overlap** — combine 22.6 vs B200 15.8 (~7 ms). *Scoped: not a porting gap* — aiter's `moe_gather_reduce` is already the same 128-bit vectorized one-block-per-token f32 gather-reduce (no atomics) as B200's `finalizeKernelVecLoad`, ~66% HBM BW standalone. The gap is purely that B200 hides the finalize behind the gemm2 tail via **PDL** (`cudaGridDependencySynchronize`), for which CDNA4 has no primitive — would need a persistent producer-consumer megakernel. High-effort, low ROI; defer.
3. **Dense-attention rewrite** — ck_tile FA 27.9 vs 9.9 (0.35x). CDNA4 `[BM,256]` fp32-accumulator register-pressure wall; needs a register-pressure-engineered D=256 kernel.

At/above parity (no further work): all-reduce (1.07x, MI355X faster), RMSNorm+quant (fused, ~parity). Gate-up GEMM (0.83x) is compute-efficient; not a fusion target.
