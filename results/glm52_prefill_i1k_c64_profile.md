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
| MoE | `mfma_moe2_afp4_wfp4 t64x256` (tuned) | 41.4 | `bmm_E2m1` + `bmm_Bfloat16_E2m1` expert GEMMs (sm100f) | 58.3 | |
| MoE | `ck_moe_mxgemm` stage1 | 31.1 | `finalizeKernelVecLoad` | 15.8 | |
| MoE | `moe_reduction_kernel` | 22.3 | `NVFP4Quantize` (act quant) | 2.6 | |
| MoE | `dynamic_per_group_scaled_quant` (act mxfp4) | 3.9 | — | — | |
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

- **Dense-attention fallback at short prefill context (IMPLEMENTED + validated, commit `3235a7d271`).** MI355X previously ran triton `_sparse_mla_fwd_split_dim` **unconditionally** (128.8 ms/forward, the #1 prefill kernel) because the dense-MHA fallback was hard-gated to NVIDIA SM90/SM100 in `dsa_backend.py`. We extended the `use_mha` device gate to gfx950 and routed `_forward_standard_mha` through aiter `flash_attn_varlen_func` (ck_tile dense FA). Threshold-gated by `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` (effective value = model `index_topk` = 2048); off = set 0. Dense triggers when `max_kv_len ≤ 2048` (short prefill); long context stays sparse-MLA.
  - **Result at i1k (GLM-5.2-MXFP4, MI355X TP4):** the ≈764 ms sparse attention stack collapses to dense FA ≈27 ms/forward (vs B200 dense ≈9.9). GSM8K parity (**0.955**, 200 ex). e2e median **TTFT drops 14-32%** vs the sparse path:

| conc | sparse TTFT (ms) | dense TTFT (ms) | Δ |
| --- | ---: | ---: | ---: |
| 4  | 277.3  | 196.9  | −29% |
| 8  | 467.6  | 366.8  | −22% |
| 16 | 872.9  | 593.9  | −32% |
| 32 | 1371.1 | 1183.6 | −14% |
| 64 | 1920.5 | 1640.6 | −15% |
- **Tuned MoE tile CSV (applied above via `AITER_CONFIG_FMOE=utilities/glm_fp8fp4_tuned_fmoe.csv`).** Tuned `mfma_moe2 t64x256` tile (vs default `t64x128`).
- **All-reduce is NOT the gap at conc64.** With the corrected apples-to-apples B200 capture, MI355X INT4 QuickReduce (130.7 ms) is actually **faster** than B200's mnnvl twoshot + rmsNormLamport all-reduce (139.5 ms) at i1k/conc64. (The earlier "B200 wins all-reduce" claim came from a mismatched 8192-workload reference; it does not hold here.) Combined all-reduce+RMSNorm: MI355X 144.6 vs B200 154.2 — MI355X ahead.
- **Dense GEMM is near-parity (0.92x), not 3-4x** — the earlier 3-4x was against the bogus old B200 column. FP8 lever (`SGLANG_DSA_FP8_PROJ_GEMM=1`, ‡) A/B was captured but that run showed an anomalous all-reduce (degraded run); needs a clean redo before drawing a conclusion.
- **Router GEMM is NOT an anomaly (microbench-verified).** MI355X runs the MoE gate as a standalone bf16 GEMM (`aiter_dsv3_router_gemm` → `tgemm.mm`, shape M×6144→256, ≈4.4 ms/forward summed over 75 MoE layers); B200 fuses the gate matmul into another GEMM (only the split-K reduction `cublasLt::splitKreduce` ~50 µs/fwd is itemized). A standalone bench of the exact router shape via the same `tgemm.mm` path hits **547 TF at M=8k up to 1196 TF (~peak) at M=65k**, on par with `torch.matmul`/hipblaslt — the `SPK1` pick is near-optimal, not wasteful. The 4.1 ms/forward is legitimate work (~4.7 TFLOP/fwd of gate projection at ~1150 TFLOPS), the same work B200 does fused. No lever here — an earlier "SPK1 is wrong / 5% roofline" note was incorrect on both counts (B200 fuses it, and the MI355X kernel is already near-peak).
- **Where MI355X actually trails B200 at i1k/conc64:** attention (0.67x — dense FA 27 vs 9.9, B200's SM100 fmha is far faster) and MoE (0.78x). Those are the real gaps; all-reduce/dense-GEMM/RMSNorm are at/above parity.
