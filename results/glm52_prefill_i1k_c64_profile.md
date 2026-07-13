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

1. **Down-GEMM (46.7 vs 26.2) — HARD FLOOR, needs a new kernel.** TP4 shards inter_dim→512/rank ⇒ nk=2 K-tiles. Latency-bound (13% MFMA, 26% VALU, 0.6% mem-stall, occ 37% ≈3 waves; ~60% unhidable dependency stall). ALL fork-level levers wash/regress at the real per-rank shape (~888us): pipe3 inert (nk<3), full-K +31%, cu_num_mul 2/3/4 wash, waves_per_eu=8 wash, tile_m=128 wash, atomic +63%. The ~888us splits ~MFMA 115 + dequant 231 + [M,topk,D] write 226 + ~316us stall. Closing to B200's bmm (26.2) needs a fundamentally different cutlass-style grouped GEMM (async/warp-specialized pipelining), not tuning. Out of fork scope.
2. **gemm2↔combine overlap — CLOSED, not viable on gfx950.** B200's 42ms is a profiler *sum*, not wall (PDL hides finalize under gemm2 → ~26-32ms wall). Gap decomp: ~20.5ms better gemm2 cubin (75%), ~6.8ms faster finalize (25%); B200 does NOT epilogue-fuse finalize. 2-stream overlap ceiling probe (per-rank, reduction on side stream concurrent w/ gemm2): **~0 within ±80us noise** — gemm2 is a persistent full-grid kernel monopolizing all CUs, so combine can't get CU slots until it retires (no GDC/CTA-retire overlap on CDNA4). Real overlap would need a fused megakernel interleaving both on shared CUs — huge effort for ~0 ceiling. Remaining real lever = the gemm2 cubin itself (hard: nk=2, 13% MFMA).
3. **Dense-attn rewrite** — FA 27.9 vs 9.9 (0.35x); needs register-pressure-engineered D=256 kernel.

Parity (no work): all-reduce (1.07x), RMSNorm (1.03x), gate-up (0.83x).
