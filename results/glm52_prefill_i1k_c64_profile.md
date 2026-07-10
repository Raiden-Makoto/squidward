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
| Dense GEMM | Tensile `MT256x256` | 7.7 | (more nvjet <1%) | ≈10.7 | |
| **Dense GEMM subtotal** | | **73.9** | | **32.7** | **0.44x** |
| All-reduce | `ncclDevKernel_Generic` (serial) | 236.0 | `nccl_RING_LL` + `mnnvl` twoshot/lamport fusion (overlapped) | 54.3 | |
| **All-reduce subtotal** | | **236.0** | | **54.3** | **0.23x** |
| RMSNorm / hadamard-quant fusion (#30715) | `aiter::add_rmsnorm_quant` | 14.1 | `fused_add_rmsnorm` | 6.0 | |
| **RMSNorm subtotal** | | **14.1** | | **6.0** | **0.43x** |
| **TOTAL prefill** | | **611** | | **170** | **0.28x** |

## Levers

- **Dense-attention fallback at short prefill context (IMPLEMENTED + validated, commit `3235a7d271`).** MI355X previously ran triton `_sparse_mla_fwd_split_dim` **unconditionally** (128.8 ms/forward, the #1 prefill kernel) because the dense-MHA fallback was hard-gated to NVIDIA SM90/SM100 in `dsa_backend.py`. We extended the `use_mha` device gate to gfx950 and routed `_forward_standard_mha` through aiter `flash_attn_varlen_func` (ck_tile dense FA). It is threshold-gated by `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`, whose effective value for GLM-5.2 is the model `index_topk` = **2048** (env raw default is also 2048; GLM-5.2 config: `index_topk=2048`, `index_topk_freq=4`); off switch = set it to 0. Dense triggers when `max_kv_len ≤ 2048`, i.e. short prefill; long context stays on sparse-MLA (correct crossover — sparse only pays off once there's enough KV to prune).
  - **Result at i1k (graph-on, GLM-5.2-MXFP4, MI355X TP4):** the ≈764 ms attention stack (sparse-MLA 515 + absorbed bmm 249) collapses to a single ck_tile dense-FA kernel at **≈104 ms**. GSM8K parity (**0.955**, 200 ex). e2e median **TTFT drops 14-32%** vs the sparse path:

| conc | sparse TTFT (ms) | dense TTFT (ms) | Δ |
| --- | ---: | ---: | ---: |
| 4  | 277.3  | 196.9  | −29% |
| 8  | 467.6  | 366.8  | −22% |
| 16 | 872.9  | 593.9  | −32% |
| 32 | 1371.1 | 1183.6 | −14% |
| 64 | 1920.5 | 1640.6 | −15% |
- **Tuned MoE tile CSV (applied above).** −17% MoE prefill GPU time via the tuned `mfma_moe2 t64x256` tile (vs default `t64x128`); already reflected in the MI355X MoE rows.
