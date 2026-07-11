# GLM-5.2 prefill — i1k/o1k conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

GPU-busy per prefill forward: MI355X 363 ms vs B200 331 ms → 0.91x. All kernel times are **ms per prefill forward pass** (both captured 4 forwards, graphs-off eager, dense-MHA at i1k, `--mem-fraction-static 0.85 --chunked-prefill-size 131072`). MI355X = single dense+tuned capture (`run_glm52.sh --profile` + `AITER_CONFIG_FMOE=utilities/glm_fp8fp4_tuned_fmoe.csv`, INT4 QuickReduce). B200 = fresh apples-to-apples capture (GLM-5.2-NVFP4, tp4 modelopt_fp4, kv fp8_e4m3, trtllm DSA, flashinfer_trtllm MoE, dense `fmhaSm100f...H256...Causal`).

| Section | MI355X kernel | MI355X ms | B200 kernel | B200 ms | B200/MI355X |
| --- | --- | ---: | --- | ---: | ---: |
| Attention | `ck_tile FmhaFwd` dense MHA (gfx950) | 27.0 | `fmhaSm100f ...H256...Causal` dense MHA | 9.9 | |
| Attention | `set_mla_kv_buffer` + concat/cast (KV-cache write) | 6.2 | `set_mla_kv_buffer` + `float8_copy` + concat `direct_copy` | 11.0 | |
| Attention | `fused_qk_rmsnorm` (q/k norm) | 4.3 | flashinfer `LayerNorm`+`RMSNorm` | 3.1 | |
| Attention | `RotaryEmbedding` (rope) | 1.3 | `fused_rope_kernel` | 1.1 | |
| Attention | DSA indexer (`fast_hadamard` + `indexer_k_quant`) | 0.4 | `fast_hadamard` + `fused_store_indexer_cache` | 1.1 | |
| **Attention subtotal** | | **39.2** | | **26.3** | **0.67x** |
| MoE gate-up GEMM (gemm1) | `ck_moe_mxgemm` stage1 | 31.1 | `bmm_E2m1` (swiGlu) | 32.1 | 1.03x |
| MoE down GEMM (gemm2, tuned `t64x256`) | `mfma_moe2_afp4_wfp4` | 41.4 | `bmm_Bfloat16_E2m1` | 26.2 | 0.63x |
| MoE combine/finalize | `moe_reduction_kernel` | 22.3 | `finalizeKernelVecLoad` (fuses routed-scale) | 15.8 | 0.71x |
| MoE act quant | `dynamic_per_group_scaled_quant` (mxfp4) | 3.9 | `NVFP4Quantize` | 2.6 | 0.67x |
| **MoE subtotal** | | **98.7** | | **76.7** | **0.78x** |
| Dense GEMM | Tensile `MT256x256` (`unquant apply`) | 41.3 | `nvjet_sm100 128x256` | ≈40 | |
| Dense GEMM | Tensile `MT224x256` | 23.2 | `nvjet_sm100 176x128` | ≈22 | |
| Dense GEMM | `aiter::bf16gemm_256x256` (q_b_proj ‡) | 11.6 | `nvjet_sm100 256x128` | ≈12 | |
| Dense GEMM | `hgemm_bf16_128x128` (router gemm) | 4.1 | — | — | |
| **Dense GEMM subtotal** | | **80.2** | | **73.7** | **0.92x** |
| All-reduce | `quickreduce::allreduce_prototype_twoshot` (INT4 CodecQ4) | 109.9 | `ncclDevKernel_AllReduce ...RING_LL` | 68.6 | |
| All-reduce | `aiter::cross_device_reduce_2stage` | 20.8 | `mnnvl twoshotAllreduce` + `rmsNormLamport` + one-shot | 70.9 | |
| **All-reduce subtotal** | | **130.7** | | **139.5** | **1.07x** |
| RMSNorm | `aiter::add_rmsnorm_quant` (norm+quant fused) | 13.9 | `fused_add_rmsnorm` (norm only; quant in MoE) | 14.7 | |
| **RMSNorm subtotal** | | **13.9** | | **14.7** | **1.06x** |
| **TOTAL prefill** | | **363** | | **331** | **0.91x** |

‡ `q_b_proj` (ColumnParallelLinear, M×2048→4096/rank) and `o_proj` (RowParallelLinear, M×4096→6144/rank) are the two 128-aligned dense projections that `SGLANG_DSA_FP8_PROJ_GEMM=1` (gfx950, default off) converts from bf16 to the aiter FP8 CK GEMM. `fused_qkv_a` (M×6144→2624, not 128-aligned) and dense-MLP (layers 0-2) stay bf16.

## Levers

- **Dense-attention fallback (IMPLEMENTED, commit `3235a7d271`).** Extended the `use_mha` gate to gfx950; short prefill (`max_kv_len ≤ 2048`) routes to ck_tile dense FA instead of sparse-MLA. Gated by `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`. Sparse ≈764 ms/fwd → dense ≈27; GSM8K parity (0.955). TTFT −14–32%:

| conc | sparse TTFT | dense TTFT | Δ |
| --- | ---: | ---: | ---: |
| 4  | 277.3  | 196.9  | −29% |
| 8  | 467.6  | 366.8  | −22% |
| 16 | 872.9  | 593.9  | −32% |
| 32 | 1371.1 | 1183.6 | −14% |
| 64 | 1920.5 | 1640.6 | −15% |

- **Tuned MoE tile CSV (`AITER_CONFIG_FMOE=utilities/glm_fp8fp4_tuned_fmoe.csv`):** `mfma_moe2 t64x256` (vs default `t64x128`).
- **All-reduce (1.07x):** MI355X INT4 QuickReduce 130.7 vs B200 139.5 — MI355X faster; not the gap.
- **Dense GEMM (0.92x):** near-parity. FP8 proj lever (`SGLANG_DSA_FP8_PROJ_GEMM=1`, ‡) A/B run had an anomalous all-reduce; needs a clean redo.
- **Router GEMM: not an anomaly.** Standalone bench of the exact shape hits 547 TF (M=8k) → ~1196 TF (M=65k, ~peak); B200 fuses it into another GEMM. No lever.
- **RMSNorm (1.06x):** parity.
- **Attention (0.67x, dense FA 27 vs 9.9): register-pressure wall on CDNA4, no fork lever.** ck_tile FA = 256 VGPR / 0 AGPR → ~2 waves; latency-bound (14% compute / 44% BW). The `[BM,256]` fp32 accumulator needs ~256 VGPR (B200 offloads to TMEM, CDNA4 can't). Config levers all neutral-or-worse: tiling 1.00x, backend swap 1.01x, `waves_per_eu` 1→8 (`4811ac0906`) spills at wpe≥3.
  - *fp8 batch-prefill (`SGLANG_DSA_FP8_DENSE_ATTN`, default off; `c410a22111`/`9c966c21a0`/`6cea8561a2`): dead.* Kernel faster (1.13–1.40x), accuracy parity (0.928 vs 0.929), but per-forward q/k/v quant (bf16 `kv_b_proj`) makes e2e +6/+42/+25% at conc 4/8/16 and OOM at conc32+. Fused fp8-K/V needs MXFP4 `kv_b_proj`, structurally excluded for GLM (`SGLANG_FORCE_MXFP4_KVB`).
- **MoE (0.78x):**
  - **Down GEMM (41.4 vs 26.2):** already tuned `t64x256`, near CDNA4 mxfp4 roofline — needs a kernel rewrite (aiter/upstream), not re-tuning.
  - **Expert combine (22.3):** near-parity once B200's shared-add is counted (15.8 + 5.4 = 21.2); memory-bound at roofline; kernel is aiter FlyDSL-JIT (aiter-side only).
  - gate-up GEMM / all-reduce / dense-GEMM / RMSNorm / router at/above parity.
