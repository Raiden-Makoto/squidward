# GLM-5.2 prefill — i1k/o1k conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

GPU-busy per prefill forward: MI355X 611 ms (overlap 1.00x) vs B200 170 ms (1.02x) → 0.28x. All kernel times are **ms per prefill forward pass** (MI355X captured 4 forwards, B200 1; normalized so both are 1-forward). DSA sparse-MLA path on both (B200 forced sparse via `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0`). MI355X MoE = tuned tile CSV.

| Section | MI355X kernel | MI355X ms | B200 kernel | B200 ms | B200/MI355X |
| --- | --- | ---: | --- | ---: | ---: |
| Attention | `_sparse_mla_fwd_split_dim` (triton sparse MLA, #30575) | 128.8 | `fmhaSm100f` fp8 paged sparse MLA | 28.3 | |
| Attention | `_batched_gemm_a8w8` absorb_prepare (#30519) | 38.8 | (absorbed bmm, folded) | 1.9 | |
| Attention | `_batched_gemm_a8w8` absorb_core (#30519) | 23.5 | — | — | |
| **Attention subtotal** | | **191.2** | | **30.1** | **0.16x** |
| MoE | `mfma_moe2_t64x256` (tuned tile) | 41.7 | `bmm_E2m1` gemm1 (swiGlu) | 13.8 | |
| MoE | `ck_moe_mxgemm` stage1 | 31.0 | `bmm_Bfloat16_E2m1` gemm2 | 13.5 | |
| MoE | `moe_reduction_kernel` | 23.2 | `bmm_E2m1` gemm1 (small-M) | 5.7 | |
| MoE | — | — | `finalizeKernelVecLoad` | 5.7 | |
| MoE | — | — | `bmm_Bfloat16_E2m1` (small-M) | 3.2 | |
| **MoE subtotal** | | **95.8** | | **43.8** | **0.46x** |
| Dense GEMM | Tensile `MT256x320` (`unquant apply`) | 29.7 | `nvjet_sm100_128x256` | 10.1 | |
| Dense GEMM | Tensile `MT224x256` | 15.7 | `nvjet_sm100_128x272` | 5.8 | |
| Dense GEMM | Tensile `MT224x384` | 12.3 | `nvjet_sm100_256x240` | 3.6 | |
| Dense GEMM | `aiter::bf16gemm_256x256` | 8.5 | `nvjet_sm100_128x240` | 2.5 | |
| Dense GEMM | Tensile `MT256x256` | 7.7 | (more nvjet <1%) | ~10.7 | |
| **Dense GEMM subtotal** | | **73.9** | | **32.7** | **0.44x** |
| All-reduce | `ncclDevKernel_Generic` (serial) | 236.0 | `nccl_RING_LL` + `mnnvl` twoshot/lamport fusion (overlapped) | 54.3 | |
| **All-reduce subtotal** | | **236.0** | | **54.3** | **0.23x** |
| RMSNorm / hadamard-quant fusion (#30715) | `aiter::add_rmsnorm_quant` | 14.1 | `fused_add_rmsnorm` | 6.0 | |
| **RMSNorm subtotal** | | **14.1** | | **6.0** | **0.43x** |
| **TOTAL prefill** | | **611** | | **170** | **0.28x** |

## Levers

- **Dense-attention fallback at short prefill context (biggest untapped win).** MI355X runs triton `_sparse_mla_fwd_split_dim` **unconditionally** — 128.8 ms/forward, ~21% of prefill and the #1 prefill kernel. B200 gates the sparse path behind `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` (default = `index_topk` = 2048): at i1k it uses **dense** attention, which is **2.9x faster than its own sparse MLA** there (B200 dense 9.9 vs sparse 28.3 ms/forward). At short context the sparse indexer topk + gather + mask overhead exceeds the KV it skips. Adding the same threshold-gated dense path to MI355X should cut its biggest i1k prefill cost toward a dense-attention cost — config/backend change, no new kernel.
- **Tuned MoE tile CSV (applied above).** −17% MoE prefill GPU time via the tuned `mfma_moe2 t64x256` tile (vs default `t64x128`); already reflected in the MI355X MoE rows.
