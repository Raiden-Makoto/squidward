# GLM-5.2 prefill — i1k/o1k conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

GPU-busy per prefill forward: MI355X 363 ms (overlap 1.00x) vs B200 150 ms (1.02x) → 0.41x. All kernel times are **ms per prefill forward pass** (MI355X captured 4 forwards, B200 1; normalized so both are 1-forward). MI355X = single consistent dense+tuned capture: `run_glm52.sh --profile` (dense-MHA default at i1k kv_len≤2048, INT4 QuickReduce + aiter fusion, `--disable-cuda-graph`) with `AITER_CONFIG_FMOE=utilities/glm_fp8fp4_tuned_fmoe.csv` (tuned MoE tile). B200 dense reference (`fmhaSm100f ...H256...Causal`); B200 per-kernel for indexer/KV/rope not in the reference capture (see †).

| Section | MI355X kernel | MI355X ms | B200 kernel | B200 ms | B200/MI355X |
| --- | --- | ---: | --- | ---: | ---: |
| Attention | `ck_tile FmhaFwd` dense MHA (gfx950, `_forward_standard_mha`) | 27.0 | `fmhaSm100f ...H256...Causal` dense MHA | 9.9 | |
| Attention | `set_mla_kv_buffer_fp8_quant` + concat/cast (KV-cache write) | 6.2 | — † | — | |
| Attention | `fused_qk_rmsnorm` (q/k norm) | 4.3 | — † | — | |
| Attention | `RotaryEmbedding` (rope) | 1.3 | — † | — | |
| Attention | DSA indexer (`fast_hadamard` + `indexer_k_quant` + LN) | 0.4 | — † | — | |
| **Attention subtotal** | | **39.2** | | **9.9** † | **—** |
| MoE | `mfma_moe2_afp4_wfp4 t64x256` (tuned tile) | 41.4 | `bmm_E2m1` gemm1 (swiGlu) | 13.8 | |
| MoE | `ck_moe_mxgemm` stage1 | 31.1 | `bmm_Bfloat16_E2m1` gemm2 | 13.5 | |
| MoE | `moe_reduction_kernel` (topk9, md6144) | 22.3 | `bmm_E2m1` gemm1 (small-M) | 5.7 | |
| MoE | `dynamic_per_group_scaled_quant` (act mxfp4) | 3.9 | `finalizeKernelVecLoad` | 5.7 | |
| MoE | — | — | `bmm_Bfloat16_E2m1` (small-M) | 3.2 | |
| **MoE subtotal** | | **98.7** | | **43.8** | **0.44x** |
| Dense GEMM | Tensile `MT256x256` (`unquant apply`) | 41.3 | `nvjet_sm100_128x256` | 10.1 | |
| Dense GEMM | Tensile `MT224x256` | 23.2 | `nvjet_sm100_128x272` | 5.8 | |
| Dense GEMM | `aiter::bf16gemm_256x256` (q_b_proj ‡) | 11.6 | `nvjet_sm100_256x240` | 3.6 | |
| Dense GEMM | `hgemm_bf16_128x128` (router gemm) | 4.1 | `nvjet_sm100_128x240` | 2.5 | |
| Dense GEMM | — | — | (more nvjet <1%) | ≈10.7 | |
| **Dense GEMM subtotal** | | **80.2** | | **32.7** | **0.41x** |
| All-reduce | `quickreduce::allreduce_prototype_twoshot` (INT4 CodecQ4) | 109.9 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` (plain NCCL ring) | 54.3 | |
| All-reduce | `aiter::cross_device_reduce_2stage` | 20.8 | — | — | |
| **All-reduce subtotal** | | **130.7** | | **54.3** | **0.42x** |
| RMSNorm / add-rmsnorm-quant fusion | `aiter::add_rmsnorm_quant` | 13.9 | `fused_add_rmsnorm` | 6.0 | |
| **RMSNorm subtotal** | | **13.9** | | **6.0** | **0.43x** |
| **TOTAL prefill** | | **363** | | **150** † | **0.41x** |

† MI355X indexer/KV-cache/rope/q-k-norm rows are from the dense EXTEND trace. B200 i1k prefill per-kernel numbers for those are not in the reference capture, so B200 attention subtotal and TOTAL count FA only for those; the B200/MI355X attention ratio is omitted until the B200 rows are sourced.

‡ `q_b_proj` (ColumnParallelLinear, M×2048→4096/rank ≈19.9 ms/fwd) and `o_proj` (RowParallelLinear, M×4096→6144/rank ≈32.3 ms/fwd) are the two 128-aligned dense projections that `SGLANG_DSA_FP8_PROJ_GEMM=1` (gfx950, default off) converts from bf16 to the aiter FP8 CK GEMM (≈2x on those). Per-module ms/fwd from the dense EXTEND trace (attributed to the attention block by the module parser); the Dense-GEMM tile rows above are the kernel-category view of the same projections + dense-MLP. `fused_qkv_a` (M×6144→2624, not 128-aligned) and dense-MLP (layers 0-2) stay bf16.

## Levers

- **Dense-attention fallback at short prefill context (IMPLEMENTED + validated, commit `3235a7d271`).** MI355X previously ran triton `_sparse_mla_fwd_split_dim` **unconditionally** (128.8 ms/forward, the #1 prefill kernel) because the dense-MHA fallback was hard-gated to NVIDIA SM90/SM100 in `dsa_backend.py`. We extended the `use_mha` device gate to gfx950 and routed `_forward_standard_mha` through aiter `flash_attn_varlen_func` (ck_tile dense FA). It is threshold-gated by `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`, whose effective value for GLM-5.2 is the model `index_topk` = **2048** (env raw default is also 2048; GLM-5.2 config: `index_topk=2048`, `index_topk_freq=4`); off switch = set it to 0. Dense triggers when `max_kv_len ≤ 2048`, i.e. short prefill; long context stays on sparse-MLA (correct crossover — sparse only pays off once there's enough KV to prune).
  - **Result at i1k (graph-on, GLM-5.2-MXFP4, MI355X TP4):** the ≈764 ms attention stack (sparse-MLA 515 + absorbed bmm 249) collapses to a single ck_tile dense-FA kernel at **≈104 ms** (MI355X dense FA ≈27 ms/forward vs B200 dense ≈9.9 ms/forward). GSM8K parity (**0.955**, 200 ex). e2e median **TTFT drops 14-32%** vs the sparse path:

| conc | sparse TTFT (ms) | dense TTFT (ms) | Δ |
| --- | ---: | ---: | ---: |
| 4  | 277.3  | 196.9  | −29% |
| 8  | 467.6  | 366.8  | −22% |
| 16 | 872.9  | 593.9  | −32% |
| 32 | 1371.1 | 1183.6 | −14% |
| 64 | 1920.5 | 1640.6 | −15% |
- **Tuned MoE tile CSV (applied above via `AITER_CONFIG_FMOE=utilities/glm_fp8fp4_tuned_fmoe.csv`).** Tuned `mfma_moe2 t64x256` tile (vs default `t64x128`); reflected in the MI355X MoE rows.
- **Dense-GEMM FP8 lever (`SGLANG_DSA_FP8_PROJ_GEMM=1`, gfx950, default off).** Converts `q_b_proj` + `o_proj` (‡) from bf16 to aiter FP8 CK GEMM; not enabled in this capture. Not yet A/B'd.
- **All-reduce is shot — hardware-bound, no software lever.** B200 runs a plain NCCL ring (`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`) and still beats MI355X's INT4 QuickReduce, which already moves 4x less data over the wire — so MI355X's reduce is already the optimal path. The residual gap is the Infinity Fabric vs NVLink5/NVSwitch bandwidth floor. Neither platform fuses RMSNorm into the reduce (both keep add+rmsnorm as a separate kernel). Can't fix hardware with software.
