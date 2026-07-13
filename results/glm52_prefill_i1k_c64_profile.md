# GLM-5.2 prefill — i1k/o1k conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

ms/prefill forward. MI355X ~390 vs B200 331 (0.85x). Graphs-off eager, dense-MHA at i1k, `bench_serving` conc64 num256 `--profile-num-steps 4 --profile-by-stage`, EXTEND trace n_fwd=4. MI355X = `v0.5.15-rocm720-mi35x-20260711`, tuned MoE confirmed executing. All-reduce row carried over from prior clean capture (fresh QuickReduce profiler-perturbed).

| Section | MI355X kernel | MI355X ms | B200 kernel | B200 ms | B200/MI355X |
| --- | --- | ---: | --- | ---: | ---: |
| Attention | `ck_tile FmhaFwd` dense MHA | 27.9 | `fmhaSm100f ...H256...Causal` | 9.9 | |
| Attention | `set_mla_kv_buffer` + concat/cast | 10.1 | `set_mla_kv_buffer` + `float8_copy` + concat | 11.0 | |
| Attention | q/k-norm | 2.0 | `LayerNorm`+`RMSNorm`+`fused_rope` | 4.2 | |
| Attention | DSA indexer | 0.3 | `fast_hadamard`+`fused_store_indexer_cache` | 1.1 | |
| **Attention** | | **40.4** | | **26.3** | **0.65x** |
| MoE gate-up (gemm1) | `mfma_moe1 t128x128x256` | 38.6 | `bmm_E2m1` | 32.1 | 0.83x |
| MoE down (gemm2) | `mfma_moe2 t64x256` | 46.7 | `bmm_Bfloat16_E2m1` | 26.2 | 0.56x |
| MoE combine | `moe_reduction_kernel` | 22.6 | `finalizeKernelVecLoad` | 15.8 | 0.70x |
| MoE act quant | `dynamic_per_group_scaled_quant` | 4.2 | `NVFP4Quantize` | 2.6 | 0.62x |
| **MoE** | | **112.2** | | **76.7** | **0.68x** |
| Dense GEMM | Tensile `MT256x256` | 27.7 | `nvjet_sm100 128x256` | ≈40 | |
| Dense GEMM | Tensile `MT256x240/240x256/224x256` | 46.0 | `nvjet_sm100 176x128` | ≈22 | |
| Dense GEMM | `aiter::bf16gemm_256x256` (q_b_proj ‡) | 8.4 | `nvjet_sm100 256x128` | ≈12 | |
| Dense GEMM | `hgemm_bf16_128x128` (router) + misc | 10.0 | — | — | |
| **Dense GEMM** | | **92.1** | | **73.7** | **0.80x** |
| All-reduce | `quickreduce twoshot` (INT4) | 109.9 | `ncclDevKernel ...RING_LL` | 68.6 | |
| All-reduce | `aiter::cross_device_reduce_2stage` | 20.8 | `mnnvl twoshot` + `rmsNormLamport` + one-shot | 70.9 | |
| **All-reduce** | | **130.7** | | **139.5** | **1.07x** |
| RMSNorm | `aiter::add_rmsnorm_quant` | 14.3 | `fused_add_rmsnorm` | 14.7 | 1.03x |
| **TOTAL** | | **~390** | | **331** | **0.85x** |

‡ `q_b_proj`/`o_proj` are the two 128-aligned dense projections `SGLANG_DSA_FP8_PROJ_GEMM=1` (default off) converts to fp8 CK GEMM.

## Levers

- **Dense-attn fallback (DONE, `3235a7d271`):** short prefill (`max_kv_len ≤ 2048`) → ck_tile dense FA, not sparse-MLA. Sparse ≈764 → dense ≈28 ms/fwd; GSM8K 0.955. TTFT −14–32% (conc4/8/16/32/64: −29/−22/−32/−14/−15%).
- **Tuned MoE dispatch (fixed):** `AITER_CONFIG_FMOE` env doesn't reach workers; injected `glm5_fp4` gfx950/cu256 rows into box aiter default. Box-local (lost on rebuild).
- **All-reduce (1.07x):** MI355X faster; not the gap.
- **Dense GEMM (0.80x):** fp8 proj lever A/B had anomalous all-reduce; needs clean redo.
- **Router GEMM:** not an anomaly (547→1196 TF standalone); B200 fuses it. No lever.
- **Attention (0.65x, FA 27.9 vs 9.9):** register-pressure wall — 256 VGPR/0 AGPR → ~2 waves; `[BM,256]` fp32 accum can't fit (B200 uses TMEM). Tiling/backend/`waves_per_eu` all neutral-or-worse. fp8 batch-prefill (`SGLANG_DSA_FP8_DENSE_ATTN`) dead: kernel 1.13–1.40x but per-fwd qkv quant → e2e +6/+42/+25% and OOM.

## Remaining gaps

1. **MoE gemm2 + combine (69 vs 42 ms) — REOPENED vs new CK.** gemm2 46.7 vs 26.2 (0.56x), combine 22.6 vs 15.8 (0.70x). At the deployed per-rank shape (TP4 shards inter_dim→512 ⇒ nk=2 K-tiles) gemm2 is latency-bound (~888us: 13% MFMA, ~316us unhidable stall; MFMA 115 + dequant 231 + [M,topk,D] write 226). Exhausted with data:
   - *FlyDSL tuning:* all levers wash/regress (pipe3 inert nk<3, full-K +31%, cu_num_mul/waves_per_eu/tile_m wash, atomic +63%); config pick optimal (862us).
   - *gemm2↔combine fusion:* dead — no CDNA4 PDL/CTA-retire overlap (persistent gemm2 hogs CUs); 2-stream ceiling ~0; fused megakernel built+measured 7112us (5.6x slower) + coherence bugs (aiter `glm52-moe2-pipe3`, `SGLANG_MOE2_FUSED`).
   - *CK:* tuner picked FlyDSL over CK at nk=2 — BUT that was **STALE CK**. Active CK (rocm-libraries) may have a better fp4 gemm2. **Re-test new-CK `moe_ck2stages_gemm2` before final.** If it also loses, a win needs a new-from-scratch CK device-op (frontier).
2. **Dense-attn FA (27.9 vs 9.9, 0.35x) — ACTIVE, CK-bound, improvable (not pure HW).** GLM dense prefill → aiter `flash_attn_varlen_func` → CK-tile `FmhaFwd` (hdim_q/v=256 causal varlen; not fmha_v3/FlyDSL). Fwd baseline **444.9us** (b16 s1024 nh16). `[BM,256]` fp32 O-accum → 256 VGPR/0 AGPR → ~2 waves. Box runs **STALE CK** (untuned gfx950 D256 `(128,128,...)`); active CK has a gfx950 D256 tune `(128,64,...)` (BN0 128→64) — cherry-pick won't compile (needs full new-CK). PLAN: new CK → CK_DIR → rebuild → A/B vs 444.9us (use CK-dev-blessed configs; hand-crafted BM0=64 hit CK-tile validity).

Both big gaps now hinge on the **new CK** (`ROCm/rocm-libraries/projects/composablekernel`; box submodule points at stale `ROCm/composable_kernel` cb859854, and box aiter itself is behind).

Parity (no work): all-reduce (1.07x), RMSNorm (1.03x).
Minor / possibly-mislabeled: gate-up 38.6 vs 32.1 (0.83x) — NOT parity; `mfma_moe1` vs B200 `bmm_E2m1` mapping may be mislabeled (verify before treating as a real gap).
