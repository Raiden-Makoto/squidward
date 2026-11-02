# GLM-5.3-Flash prefill — i8k/o16 conc64, TP4 — GB300 vs MI355X

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm53_flash.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, script snapshot `6aef5d5fc2` on both platforms. GB300 values come from the graphs-off tables in [`glm53_prefill_i8k_c64_gb300_profile.md`](glm53_prefill_i8k_c64_gb300_profile.md). MI355X values come from the graphs-off tables in [`glm53_prefill_i8k_c64_profile.md`](glm53_prefill_i8k_c64_profile.md), re-profiled on the 20261102 image; raw artifacts are under `/data2/macui/glm53/glm53_mxfp4_prefill_profile_1102` on `smci355-ccs-aus-m12-17`.

| | GB300 | MI355X |
| --- | --- | --- |
| Hardware | 4× GB300 | MI355X (`gfx950`), TP4 on GPUs 4–7 |
| Source | `RM/glm53` `84bd7ab188` | `RM/glm53` `dd845733fe` (upstream main `b55350bb93` merged); KDA prefill on the HIP chunked-prefill kernel from [#45710](https://github.com/sgl-project/sglang/pull/45710) |
| Model | `nvidia/GLM-5.3-Flash-NVFP4@da920bb0b9f4a06727223a349e55468e38352348` | `amd/GLM-5.3-Flash-Quark-MXFP4@b5688f25491202978c19c4d036eef579f61bbe07` |
| Image | `lmsysorg/sglang:latest@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae` | `rocm/sgl-dev:v0.5.23-rocm724-mi35x-20261102` |
| AITER | — | `3015e3639` (image AITER `285579b6` plus the GLM-5.3-Flash MXFP4 MoE validation state) |
| DSA backend | TRT-LLM prefill/decode | `aiter_sparse_mla` prefill/decode (Triton sparse MLA at this batch) |
| KV cache | `fp8_e4m3` | `bfloat16` |
| MoE runner | `flashinfer_trtllm` | `aiter` |
| Shared-expert fusion | disabled by the server | enabled |
| All-reduce | NCCL | QuickReduce INT4 |
| GSM8K-20 | 95%, zero errors | 95%, zero errors |

The ratio column is MI355X ms divided by GB300 ms; values above 1.00× mean MI355X spends more time. MI355X values exclude two trace spans that overlap other GPU work for their whole duration; they are listed under [Overlapping trace spans](#overlapping-trace-spans) and are not counted in any row or total.

## Attention

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| K-pool plan transfer/wait | 0.0 | 0.0 | — |
| MLA FP8 activation quant | no separate row | 0.9 | — |
| Sparse MLA | 28.5 | 38.7 | 1.36× |
| DSA top-k | 3.6 | 2.5 | 0.68× |
| DSA MQA logits | 0.8 | 0.9 | 1.03× |
| DSA prep/store + misc | 17.6 | 14.1 | 0.80× |
| **Attention subtotal** | **50.6** | **57.0** | **1.13×** |

## MoE

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| Routed experts gate-up | 11.9 | 13.0 | 1.10× |
| Routed experts down + combine | 17.8 | 25.8 | 1.45× |
| Shared-expert MLP + add | 7.3 | fused into routed experts | — |
| Routing / sort / quant | 3.7 | 8.2 | 2.23× |
| **MoE subtotal** | **40.7** | **47.0** | **1.16×** |

GB300 gate-up is TRT-LLM NVFP4 `bmm_E2m1`; down + combine is `bmm_Bfloat16_E2m1` plus `finalizeKernelVecLoad`. MI355X gate-up is the AITER FlyDSL A4W4 `gemm1_a4w4_port` kernel; down + combine is `gemm2_a4w4_port` plus `scatter_reduce` (with the scatter `gemm2` variant) or `moe_reduction` (with the reduce `gemm2` variant). MI355X fuses the shared expert into these kernels as a ninth expert per token; GB300 runs it as separate `nvjet` GEMMs, SiLU, and a residual add. With shared-expert work included on both sides, GB300 spends 37.0 ms and MI355X 38.9 ms (1.05×).

## Dense GEMM

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| KDA tensor materialization | 7.3 | 2.4 | 0.33× |
| KDA projections | 12.5 | 12.0 | 0.96× |
| KDA state/output | 7.9 | 10.6 | 1.35× |
| Causal convolution | 3.9 | 5.0 | 1.27× |
| DenseMLP L0–2 | 0.8 | 2.0 | 2.69× |
| Other KDA / dense kernels | 9.7 | 11.2 | 1.15× |
| **Dense/KDA subtotal** | **42.1** | **43.4** | **1.03×** |

GB300 KDA projections are a BF16 `nvjet` GEMM; MI355X KDA projections are the AITER per-token/per-channel FP8 GEMM. GB300 KDA state/output is the FlashInfer `kernel_cutlass_kernel_chain_dv2` kernel; MI355X KDA state/output is the HIP chunked-prefill `kda_intra_kernel` plus `kda_recur_*` kernels, which apply the gate, beta sigmoid, and Q/K L2 norms inside the kernel.

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

- MI355X (11.2 ms):

| MI355X kernels | ms |
| --- | ---: |
| `o_proj` FP8 GEMM (FlyDSL) | 3.6 |
| `o_proj` input gated RMSNorm (FP8 path) | 1.8 |
| KDA per-token FP8 quant and two hipBLASLt GEMMs | 1.8 |
| `fused_fg_b_proj` BF16 GEMM (hipBLASLt) | 1.5 |
| Decoder-layer DtoD copy, elementwise, mean reduce | 0.9 |
| KDA cache-index compare (elementwise) | 0.1 |
| Logits, embedding, final norm, sampler | 0.2 |
| Unattributed (includes one 1.9 ms `alloc_extend_kernel` launch) | 1.2 |

## Communication and normalization

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| All-reduce | 73.6 | 34.9 | 0.47× |
| Additional reduction | 0.0 | 14.2 | — |
| mHC pre RMSNorm | 9.3 | 9.3 | 1.00× |
| mHC post | 13.7 | 14.9 | 1.09× |
| mHC pre sqrsum | 8.4 | 13.3 | 1.59× |
| **Communication/norm subtotal** | **105.0** | **86.6** | **0.82×** |
| **TOTAL** | **238.3** | **234.0** | **0.98×** |
| GPU-busy time (union of all GPU events) | 237.6 | 233.2 | 0.98× |

## Overlapping trace spans

These MI355X events are recorded with long durations while the GPU keeps executing other kernels throughout, so their recorded time is a span, not GPU time. Including them gives the 360.6 ms/forward summed-event total.

| Span | Events | Recorded ms/fwd | Overlapping kernels per span | Overlapping kernel time per span |
| --- | ---: | ---: | ---: | ---: |
| K-pool plan HtoD from `_kpool_plan_to_gpu` (128 KiB) | 1 | 63.6 | 1,729 | 257 ms |
| Logits-processor BF16-to-FP32 copy (`bfloat16tofloat32_copy_kernel`) | 1 | 63.0 | 1,808 | 254 ms |

GB300 has no equivalent spans: its summed event time (238.3 ms/forward) and GPU-busy time (237.6 ms/forward) agree. Its graphs-off all-reduce row (73.6 ms/forward) includes cross-rank waiting; the same 364 NCCL calls total 42.2 ms/forward with CUDA graphs enabled.

GB300 uses an FP8 KV cache and MI355X a BF16 KV cache, because `aiter_sparse_mla` accepts only BF16 KV, so the sparse MLA row also compares different KV bytes per token.
