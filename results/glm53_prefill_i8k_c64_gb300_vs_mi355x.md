# GLM-5.3-Flash prefill — i8k/o16 conc64, TP4 — GB300 vs MI355X

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm53_flash.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, script snapshot `6aef5d5fc2` on both platforms. Values come from the graphs-off tables in [`glm53_prefill_i8k_c64_gb300_profile.md`](glm53_prefill_i8k_c64_gb300_profile.md) and [`glm53_prefill_i8k_c64_profile.md`](glm53_prefill_i8k_c64_profile.md); see those documents for kernel names, attribution, and raw artifacts.

| | GB300 | MI355X |
| --- | --- | --- |
| Hardware | 4× GB300 | MI355X (`gfx950`), TP4 on GPUs 4–7 |
| Source | `RM/glm53` `84bd7ab188` | `RM/glm53` `e593c7ebdd` |
| Model | `nvidia/GLM-5.3-Flash-NVFP4@da920bb0b9f4a06727223a349e55468e38352348` | `amd/GLM-5.3-Flash-Quark-MXFP4@b5688f25491202978c19c4d036eef579f61bbe07` |
| Image | `lmsysorg/sglang:latest@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae` | `rocm/sgl-dev:v0.5.21-rocm724-mi35x-20261014` |
| AITER | — | `e7d2453f25` with uncommitted local modifications |
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
| Sparse MLA | 28.5 | 38.7 | 1.36× |
| DSA top-k | 3.6 | 2.4 | 0.67× |
| DSA MQA logits | 0.8 | 0.9 | 1.12× |
| DSA prep/store + misc | 17.6 | 13.9 | 0.79× |
| **Attention subtotal** | **50.6** | **56.9** | **1.12×** |

## MoE

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| Routed experts | 29.7 | 50.5 | 1.70× |
| Routing / sort / quant | 11.0 | 9.1 | 0.83× |
| **MoE subtotal** | **40.7** | **59.6** | **1.46×** |

## Dense GEMM

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| KDA tensor materialization | 7.3 | 2.5 | 0.34× |
| KDA projections | 12.5 | 12.0 | 0.96× |
| KDA state/output | 7.9 | 11.1 | 1.41× |
| Causal convolution | 3.9 | 5.0 | 1.28× |
| DenseMLP L0–2 | 0.8 | 2.0 | 2.50× |
| Other KDA / dense kernels | 9.7 | 26.6 | 2.74× |
| **Dense/KDA subtotal** | **42.1** | **59.2** | **1.41×** |

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

- MI355X (26.6 ms):

| MI355X kernels | ms |
| --- | ---: |
| KDA chunked-prefill Triton kernels (`chunk_gla_fwd_kernel_o`, `_recompute_w_u_fwd_kernel`, `chunk_kda_fwd_kernel_intra_sub_chunk`, `chunk_kda_fwd_kernel_inter_solve_fused`, `kda_gate_chunk_cumsum_vector_kernel`, `l2norm_fwd_kernel`, elementwise) | 16.0 |
| `o_proj` FP8 GEMM (FlyDSL) | 3.7 |
| `o_proj` input gated RMSNorm to FP8 | 1.8 |
| KDA per-token FP8 quant and two hipBLASLt GEMMs | 1.8 |
| `fused_fg_b_proj` BF16 GEMM (hipBLASLt) | 1.5 |
| Decoder-layer DtoD copy, elementwise, mean reduce | 0.9 |
| Logits, embedding, final norm, sampler | 0.2 |
| Unattributed | 0.7 |

## Communication and normalization

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| All-reduce | 73.6 | 34.6 | 0.47× |
| Additional reduction | 0.0 | 14.2 | — |
| mHC pre RMSNorm | 9.3 | 9.3 | 1.00× |
| mHC post | 13.7 | 14.8 | 1.08× |
| mHC pre sqrsum | 8.4 | 13.1 | 1.56× |
| **Communication/norm subtotal** | **105.0** | **86.0** | **0.82×** |
| **TOTAL** | **238.3** | **261.7** | **1.10×** |
| GPU-busy time (union of all GPU events) | 237.6 | 268.3 | 1.13× |

## Overlapping trace spans

These MI355X events are recorded with long durations while the GPU keeps executing other kernels throughout, so their recorded time is a span, not GPU time. Including them gives the 610.0 ms/forward summed-event total.

| Span | Events | Recorded ms/fwd | Overlapping kernels per span | Overlapping kernel time per span |
| --- | ---: | ---: | ---: | ---: |
| K-pool plan HtoD from `_kpool_plan_to_gpu` (64–128 KiB each) | 4 | 202.4 | 1,540–1,579 | 156–323 ms |
| MLA FP8 activation quant (`dynamic_per_group_scaled_quant`) | 3 | 128.5 | 1,172–1,247 | 265–267 ms |
| MoE routing (`mxfp4_moe_sort_kernel`) | 1 | 17.4 | 805 | 84 ms |

GB300 has no equivalent spans: its summed event time (238.3 ms/forward) and GPU-busy time (237.6 ms/forward) agree. Its graphs-off all-reduce row (73.6 ms/forward) includes cross-rank waiting; the same 364 NCCL calls total 42.2 ms/forward with CUDA graphs enabled.

GB300 uses an FP8 KV cache and MI355X a BF16 KV cache, because `aiter_sparse_mla` accepts only BF16 KV, so the sparse MLA row also compares different KV bytes per token.
