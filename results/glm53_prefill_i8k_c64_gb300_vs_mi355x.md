# GLM-5.3-Flash prefill — i8k/o16 conc64, TP4 — GB300 vs MI355X

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm53_flash.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, script snapshot `6aef5d5fc2` on both platforms. Values come from the graphs-off tables in [`glm53_prefill_i8k_c64_gb300_profile.md`](glm53_prefill_i8k_c64_gb300_profile.md) and [`glm53_prefill_i8k_c64_profile.md`](glm53_prefill_i8k_c64_profile.md); see those documents for kernel names, attribution, and raw artifacts.

| | GB300 | MI355X |
| --- | --- | --- |
| Hardware | 4× GB300 | MI355X (`gfx950`), TP4 on GPUs 4–7 |
| Source | `RM/glm53` `84bd7ab188` | `RM/glm53` `e593c7ebdd` |
| Model | `nvidia/GLM-5.3-Flash-NVFP4@da920bb0b9f4a06727223a349e55468e38352348` | `amd/GLM-5.3-Flash-Quark-MXFP4@b5688f25491202978c19c4d036eef579f61bbe07` |
| Image | `lmsysorg/sglang:latest@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae` | `rocm/sgl-dev:v0.5.21-rocm724-mi35x-20261014` |
| DSA backend | TRT-LLM prefill/decode | `aiter_sparse_mla` prefill/decode (Triton sparse MLA at this batch) |
| KV cache | `fp8_e4m3` | `bfloat16` |
| MoE runner | `flashinfer_trtllm` | `aiter` |
| Shared-expert fusion | disabled by the server | enabled |
| All-reduce | NCCL | QuickReduce INT4 |
| GSM8K-20 | 95%, zero errors | 95%, zero errors |

The ratio column is MI355X ms divided by GB300 ms; values above 1.00× mean MI355X spends more time.

## Attention

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| K-pool plan transfer/wait | 0.0 | 202.4 | — |
| MLA FP8 activation quant | no separate row | 129.4 | — |
| Sparse MLA | 28.5 | 38.7 | 1.36× |
| DSA top-k | 3.6 | 2.4 | 0.67× |
| DSA MQA logits | 0.8 | 0.9 | 1.12× |
| DSA prep/store + misc | 17.6 | 13.9 | 0.79× |
| **Attention subtotal** | **50.6** | **387.8** | **7.66×** |

## MoE

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| Routed experts | 29.7 | 50.5 | 1.70× |
| Routing / sort / quant | 11.0 | 26.5 | 2.41× |
| **MoE subtotal** | **40.7** | **77.0** | **1.89×** |

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

## Communication and normalization

| Component | GB300 ms | MI355X ms | MI355X / GB300 |
| --- | ---: | ---: | ---: |
| All-reduce | 73.6 | 34.6 | 0.47× |
| Additional reduction | 0.0 | 14.2 | — |
| mHC pre RMSNorm | 9.3 | 9.3 | 1.00× |
| mHC post | 13.7 | 14.8 | 1.08× |
| mHC pre sqrsum | 8.4 | 13.1 | 1.56× |
| **Communication/norm subtotal** | **105.0** | **86.0** | **0.82×** |
| **TOTAL** | **238.3** | **610.0** | **2.56×** |

## Waiting included in the totals

Both totals contain waiting rather than kernel work, so the TOTAL ratio overstates the kernel-work gap.

- MI355X: the four K-pool HtoD spans (202.4 ms/forward) and four other stalled launches (145.9 ms/forward: 128.5 ms in the MLA FP8 activation quant row and 17.4 ms in the MoE routing row) are queued dependency waits. Excluding them, the MI355X total is 261.7 ms/forward, and the two rows are 0.9 ms and 9.1 ms of kernel work.
- GB300: the graphs-off all-reduce span (73.6 ms/forward) includes cross-rank waiting.
