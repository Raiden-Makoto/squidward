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


| Area                                      | MI vs B200 (ms) | ratio | status                               | result / next                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------------- | --------------- | ----- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dense GEMM `MT224x256`+`MT256x240` (bf16) | 33.0 vs ≈22     | 0.67x | **AT VENDOR FLOOR — no lever**       | q_a+kv_a (N2624/K6144, 25.2) + kv_b (7.8). Full re-tune @M16384 (all backends + hipBLASLt): best = hipBLASLt `MT256x256` 408us/1294 TFLOPS/783 GB/s = the deployed torch-native. N-padding 2624→2816 is WORSE (450 vs 396us). flydsl broken here (19 solns error). q_a+kv_a can't go fp8 (N2624 not 128-aligned). Gap is vendor-library/architectural; only path = from-scratch kernel beating hipBLASLt (large, uncertain).                                                     |
| MoE gemm2                                 | 47.5 vs 26.2    | 0.55x | CK tested — loses, PARKED             | new-CK unblocked (CK IndexEval fix #9091); new-CK gemm2 v1 = FlyDSL +18→+41% (M8k→128k, prod shape). FlyDSL wins; CK gemm2 needs rewrite.                                                                                                                                                                                                                                                                                                                                        |
| MoE gemm2↔combine fusion                  | —               | —     | DEAD                                 | no CDNA4 PDL overlap; fused megakernel 5.6x slower                                                                                                                                                                                                                                                                                                                                                                                                                               |
| MoE combine                               | 22.3 vs 15.8    | 0.71x | HBM-bound, no lever                  | `moe_reduction` 366us@M16384 ~5.5 TB/s; combine fusion DEAD (gemm2+combine megakernel 5.6x slower)                                                                                                                                                                                                                                                                                                                                                                               |
| MoE act quant                             | 4.0 vs 2.6      | 0.65x | HBM-bound, no lever                  | `dynamic_per_group_scaled_quant` 30us@M16384 ~6.3 TB/s. Fusion into gemm1 TESTED = net LOSS: fp4q stage1 1615.7us vs bf16 1569.5 (+46us epilogue quant) > standalone quant 30us. gemm1 is VALU-saturated so in-epilogue group-max+pack has no free cycles. Tuner's bf16+separate-quant is correct.                                                                                                                                                                               |
| Attention FA (kernel)                     | 27.3 vs 9.9     | 0.36x | CLOSED                               | pin D256 `(128,128)` new-CK = 421us (−5.4%); rest architectural (B200 TMEM); DSA indexer = required decode-cache, not a lever — see detail                                                                                                                                                                                                                                                                                                                                       |
| MoE gate-up (gemm1)                       | 38.0 vs 32.1    | 0.85x | **LEVER: new-CK gufusion_v3**        | new-CK gemm1 beats FlyDSL −5→−7.6% at prefill M≥8192 (@16384 626.8 vs 678.7us), loses M≤4096; numerically ==FlyDSL. Capture via M-gated mixed dispatch (CK gemm1 M≥8192 + FlyDSL gemm2), unwired. See detail                                                                                                                                                                                                                                                                     |
| Dense GEMM fp8 proj                       | —               | —     | **WIN — prefill-only gate, upstreaming** | Tuned a8w8_blockscale_bpreshuffle q_b_proj(4096,2048)+o_proj(6144,4096): Dense GEMM −8.8ms/−9.5% (o_proj 45.9→27.9). Prefill-only M-gate (M>512 fp8, decode bf16); GSM8K 0.943. Untuned = regression (tuning mandatory). Upstreaming: sglang PR `RM/fp8-proj-gemm-prefill-gate` + tuned config aiter#4243. |
| Router GEMM                               | —               | —     | no lever                             | B200 fuses; standalone 547→1196 TF                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Dense GEMM `MT256x256`                    | 27.7 vs ≈40     | 1.4x  | MI faster                            | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| All-reduce                                | 130.7 vs 139.5  | 1.07x | parity                               | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| RMSNorm                                   | 14.0 vs 14.7    | 1.05x | parity                               | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Dense-attn fallback                       | —               | —     | DONE `3235a7d271`                    | sparse≈764→dense≈28 ms/fwd; GSM8K 0.955; TTFT −14–32%                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Tuned MoE dispatch                        | —               | —     | DONE (box-local)                     | `glm5_fp4` gfx950/cu256 rows injected; lost on rebuild                                                                                                                                                                                                                                                                                                                                                                                                                           |




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
New-CK re-tested (unblocked by CK IndexEval fix rocm-libraries #9091; our dup issue #9400 / PR #9477 closed). new-CK gemm2 v1 @prod shape 6144,512 = FlyDSL +18% (M8k) → +41% (M128k). FlyDSL wins; CK gemm2 needs a rewrite. PARKED.

### MoE gemm1 gate-up detail (prod shape 6144,512 per-rank, a4w4, per-call µs)

new-CK `gufusion_v3` vs FlyDSL `mfma_moe1 t128x128x256`, gemm1 only (`test_moe_2stage.py --kernel`, GPU0):

| M      | FlyDSL | new-CK |      Δ |
| ------ | -----: | -----: | -----: |
| 1024   |  146.8 |  198.9 |   +35% |
| 4096   |  273.7 |  305.9 |   +12% |
| 8192   |  417.0 |  396.2 |  −5.0% |
| 16384  |  678.7 |  626.8 |  −7.6% |
| 32768  | 1175.3 | 1089.7 |  −7.3% |
| 131072 | 4233.2 | 4037.6 |  −4.6% |

new-CK wins −5→−7.6% at prefill M≥8192 (numerically ==FlyDSL, logits_diff 1e-5), loses M≤4096. Lever = M-gated mixed dispatch (CK gemm1 M≥8192 + FlyDSL gemm2), unwired. FlyDSL gemm1 itself VALU-bound: MFMA 37% / VALU 100% @16384; native mxfp4 MFMA (`mfma_scale_f32_16x16x128_f8f6f4`) already used, cost = per-MFMA fp4 packing + silu_mul.

**MoE GEMM section:** gemm1 has a new-CK mixed-dispatch lever (−5→−8% prefill, unwired); gemm2 CK loses (PARKED, needs rewrite).