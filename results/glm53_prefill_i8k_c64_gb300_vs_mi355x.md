# GLM-5.3-Flash prefill — i8k/o16 conc64, TP4 — GB300 vs MI355X

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm53_flash.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, script snapshot `6aef5d5fc2` on both platforms. GB300 values come from the graphs-off tables in [`glm53_prefill_i8k_c64_gb300_profile.md`](glm53_prefill_i8k_c64_gb300_profile.md). MI355X values come from a re-profile with tuned MXFP4 MoE configs; raw artifacts are under `/data2/hf_home/glm53_mxfp4_prefill_profile_1021_tunedmoe_1019` on `smci355-ccs-aus-m12-17`.

| | GB300 | MI355X |
| --- | --- | --- |
| Hardware | 4× GB300 | MI355X (`gfx950`), TP4 on GPUs 4–7 |
| Source | `RM/glm53` `84bd7ab188` | `RM/glm53` `f320188a6a` (code identical to `e593c7ebdd`) |
| Model | `nvidia/GLM-5.3-Flash-NVFP4@da920bb0b9f4a06727223a349e55468e38352348` | `amd/GLM-5.3-Flash-Quark-MXFP4@b5688f25491202978c19c4d036eef579f61bbe07` |
| Image | `lmsysorg/sglang:latest@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae` | `rocm/sgl-dev:v0.5.22-rocm724-mi35x-20261019` |
| AITER | — | `2bb549102` with [#6435](https://github.com/ROCm/aiter/pull/6435) MXFP4 MoE configs and the four 16384-token rows posted there |
| DSA backend | TRT-LLM prefill/decode | `aiter_sparse_mla` prefill/decode (Triton sparse MLA at this batch) |
| KV cache | `fp8_e4m3` | `bfloat16` |
| MoE runner | `flashinfer_trtllm` | `aiter` |
| Shared-expert fusion | disabled by the server | enabled |
| All-reduce | NCCL | QuickReduce INT4 |
| GSM8K-20 | 95%, zero errors | 95%, zero errors |

The ratio column is MI355X ms divided by GB300 ms; values above 1.00× mean MI355X spends more time. MI355X values exclude eight trace spans that overlap other GPU work for their whole duration; they are listed under [Overlapping trace spans](#overlapping-trace-spans) and are not counted in any row or total.

## Attention

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| K-pool plan transfer/wait | 0.0 | 0.0 | — |
| MLA FP8 activation quant | no separate row | 0.9 | — |
| Sparse MLA | 28.5 | 38.8 | 1.36× |
| DSA top-k | 3.6 | 2.5 | 0.68× |
| DSA MQA logits | 0.8 | 0.9 | 1.04× |
| DSA prep/store + misc | 17.6 | 14.2 | 0.81× |
| **Attention subtotal** | **50.6** | **57.3** | **1.13×** |

## MoE

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| Routed experts gate-up | 11.9 | 13.0 | 1.09× |
| Routed experts down + combine | 17.8 | 25.7 | 1.44× |
| Shared-expert MLP + add | 7.3 | fused into routed experts | — |
| Routing / sort / quant | 3.7 | 8.2 | 2.24× |
| **MoE subtotal** | **40.7** | **46.9** | **1.15×** |

GB300 gate-up is TRT-LLM NVFP4 `bmm_E2m1`; down + combine is `bmm_Bfloat16_E2m1` plus `finalizeKernelVecLoad`. MI355X gate-up is the AITER FlyDSL A4W4 `gemm1_a4w4_port` kernel; down + combine is `gemm2_a4w4_port` plus `scatter_reduce` (16384-token bucket) or `moe_reduction` (8192-token bucket). MI355X fuses the shared expert into these kernels as a ninth expert per token; GB300 runs it as separate `nvjet` GEMMs, SiLU, and a residual add. With shared-expert work included on both sides, GB300 spends 37.0 ms and MI355X 38.7 ms (1.04×).

## Dense GEMM

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| KDA tensor materialization | 7.3 | 2.4 | 0.33× |
| KDA projections | 12.5 | 11.7 | 0.94× |
| KDA state/output | 7.9 | 10.9 | 1.39× |
| Causal convolution | 3.9 | 5.0 | 1.28× |
| DenseMLP L0–2 | 0.8 | 2.0 | 2.68× |
| Other KDA / dense kernels | 9.7 | 26.3 | 2.70× |
| **Dense/KDA subtotal** | **42.1** | **58.5** | **1.39×** |

GB300 KDA projections are a BF16 `nvjet` GEMM; MI355X KDA projections are the AITER per-token/per-channel FP8 GEMM.

Other KDA / dense kernels:

- GB300 (9.7 ms):

| GB300 kernels | ms |
| --- | ---: |
| `o_proj` BF16 GEMM (`nvjet`) | 4.0 |
| KDA chunked-prefill FlashInfer `kernel_cutlass_kernel_prep`, elementwise, DtoH copy | 2.7 |
| `o_proj` input gated RMSNorm (`layer_norm_gated_fwd_kernel`) | 1.0 |
| `fused_fg_b_proj` BF16 GEMM (`nvjet`) | 1.0 |
| Logits, embedding, final norm, sampler | 0.5 |
| Decoder-layer mean reduce | 0.2 |
| Unattributed | 0.3 |

- MI355X (26.3 ms):

| MI355X kernels | ms |
| --- | ---: |
| KDA chunked-prefill Triton kernels (`chunk_gla_fwd_kernel_o`, `_recompute_w_u_fwd_kernel`, `chunk_kda_fwd_kernel_intra_sub_chunk`, `chunk_kda_fwd_kernel_inter_solve_fused`, `kda_gate_chunk_cumsum_vector_kernel`, `l2norm_fwd_kernel`, elementwise) | 15.6 |
| `o_proj` FP8 GEMM (FlyDSL) | 3.7 |
| `o_proj` input gated RMSNorm to FP8 | 1.8 |
| KDA per-token FP8 quant and two hipBLASLt GEMMs | 1.8 |
| `fused_fg_b_proj` BF16 GEMM (hipBLASLt) | 1.5 |
| Decoder-layer DtoD copy, elementwise, mean reduce | 0.9 |
| Logits, embedding, final norm, sampler | 0.2 |
| Unattributed | 0.8 |

## Communication and normalization

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| All-reduce | 73.6 | 35.8 | 0.49× |
| Additional reduction | 0.0 | 14.1 | — |
| mHC pre RMSNorm | 9.3 | 9.2 | 0.99× |
| mHC post | 13.7 | 14.6 | 1.07× |
| mHC pre sqrsum | 8.4 | 13.3 | 1.58× |
| **Communication/norm subtotal** | **105.0** | **87.0** | **0.83×** |
| **TOTAL** | **238.3** | **249.7** | **1.05×** |
| GPU-busy time (union of all GPU events) | 237.6 | 257.0 | 1.08× |

## Overlapping trace spans

These MI355X events are recorded with long durations while the GPU keeps executing other kernels throughout, so their recorded time is a span, not GPU time. Including them gives the 580.1 ms/forward summed-event total.

| Span | Events | Recorded ms/fwd | Overlapping kernels per span | Overlapping kernel time per span |
| --- | ---: | ---: | ---: | ---: |
| K-pool plan HtoD from `_kpool_plan_to_gpu` (64–128 KiB each) | 4 | 201.8 | 1,553–1,592 | 145–316 ms |
| KDA projection FP8 GEMM (`kernel_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle_2lds`) | 3 | 115.0 | 1,085–1,162 | 241–244 ms |
| MoE combine (`moe_reduction`) | 1 | 13.7 | 664 | 60 ms |

GB300 has no equivalent spans: its summed event time (238.3 ms/forward) and GPU-busy time (237.6 ms/forward) agree. Its graphs-off all-reduce row (73.6 ms/forward) includes cross-rank waiting; the same 364 NCCL calls total 42.2 ms/forward with CUDA graphs enabled.

GB300 uses an FP8 KV cache and MI355X a BF16 KV cache, because `aiter_sparse_mla` accepts only BF16 KV, so the sparse MLA row also compares different KV bytes per token.
