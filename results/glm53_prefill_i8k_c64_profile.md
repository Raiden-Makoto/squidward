# GLM-5.3-Flash prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4 integrated base on the 20261102 image)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm53_flash.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm53` `dd845733fe` (upstream main `b55350bb93`, the SGLang source of the image, merged into the integration branch) run from `/sgl-workspace/squidward` in `macui-glm53-new` on `smci355-ccs-aus-m12-17` with `utilities/run_glm53_flash.sh --mxfp4`, script snapshot `6aef5d5fc2`, AITER `3015e3639` (image AITER `285579b6` plus the GLM-5.3-Flash MXFP4 MoE validation state); `SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj`, `SGLANG_OPT_GLM53_KDA_W8A16_DECODE=1`, `SGLANG_DSA_FUSE_HADAMARD_QUANT=1`, `aiter_sparse_mla` DSA prefill/decode backends, shared-expert fusion enabled, INT4 QuickReduce. KDA prefill runs the default Triton backend, which dispatches to the gfx950 HIP chunked-prefill kernel from [#45710](https://github.com/sgl-project/sglang/pull/45710). MI355X image `rocm/sgl-dev:v0.5.23-rocm724-mi35x-20261102`; model snapshot `amd/GLM-5.3-Flash-Quark-MXFP4@b5688f25491202978c19c4d036eef579f61bbe07`. GSM8K-20 scored 95% with zero errors.

The table values are summed profiler-event durations divided by four EXTEND forwards. They exclude two trace spans that overlap other GPU work for their whole duration; those spans are listed under [Overlapping trace spans](#overlapping-trace-spans) and are not counted in any row or total.

## Attention

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| K-pool plan transfer/wait | pinned HtoD from `_kpool_plan_to_gpu` | 0.0 | 0.0% |
| MLA FP8 activation quant | AITER `dynamic_per_group_scaled_quant` from `aiter_w8a8_block_fp8_linear` | 0.9 | 0.4% |
| Sparse MLA | Triton `_sparse_mla_fwd_split_dim_kernel` | 38.7 | 16.5% |
| DSA top-k | `kpool_topk_transform_kernel<512>` | 2.5 | 1.0% |
| DSA MQA logits | Gluon FP8 MQA logits | 0.9 | 0.4% |
| DSA prep/store + misc | MLA and indexer projection GEMMs, fused Hadamard quant, norm/cache assembly, remaining attention kernels | 14.1 | 6.0% |
| **Attention subtotal** | | **57.0** | **24.4%** |

## MoE

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| Routed experts | AITER FlyDSL A4W4 `gemm1_a4w4_port` and `gemm2_a4w4_port`, plus `scatter_reduce` / `moe_reduction` | 38.9 | 16.6% |
| Routing / sort / quant | OPUS sorting, MXFP4 MoE sort, gate GEMM, grouped top-k, shared-expert append, FP4 activation quant | 8.2 | 3.5% |
| **MoE subtotal** | | **47.0** | **20.1%** |

## Dense GEMM

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| KDA tensor materialization | BF16 direct-copy kernels in `RadixLinearAttention` | 2.4 | 1.0% |
| KDA PTPC projections | AITER per-token/per-channel FP8 GEMM | 12.0 | 5.1% |
| KDA state/output | HIP chunked prefill `kda_intra_kernel` and `kda_recur_*` | 10.6 | 4.5% |
| Causal convolution | `_causal_conv1d_fwd_kernel` | 5.0 | 2.1% |
| DenseMLP L0–2 | dense block-FP8 MLP | 2.0 | 0.9% |
| Other KDA / dense kernels | KDA `o_proj` FP8 GEMMs, `f_b`/`g_b` batched GEMM, gated RMSNorm FP8 quant, activation quant, beta GEMM, logits/embedding/sampler, misc | 11.2 | 4.8% |
| **Dense/KDA subtotal** | | **43.4** | **18.5%** |

AITER logged no tuned configuration for the packed KDA PTPC projection shape `M=8192, N=6400, K=4096` and used its default.

## Communication and normalization

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| All-reduce | QuickReduce INT4 | 34.9 | 14.9% |
| Additional reduction | AITER `cross_device_reduce_2stage` | 14.2 | 6.1% |
| mHC pre RMSNorm | `mhc_pre_big_fuse_rmsnorm_kernel` | 9.3 | 4.0% |
| mHC post | `mhc_post_kernel` | 14.9 | 6.4% |
| mHC pre sqrsum | `mhc_pre_gemm_sqrsum_kernel` | 13.3 | 5.7% |
| **Communication/norm subtotal** | | **86.6** | **37.0%** |
| **TOTAL** | | **234.0** | **100.0%** |
| GPU-busy time (union of all GPU events) | | 233.2 | |

The mHC pre RMSNorm value matches the graphs-on formal trace (9.3 ms/forward graphs-off, 8.9 ms/forward graphs-on).

## Overlapping trace spans

These events are recorded with long durations while the GPU keeps executing other kernels throughout, so their recorded time is a span, not GPU time. Including them gives the 360.6 ms/forward summed-event total.

| Span | Events | Recorded ms/fwd | Overlapping kernels per span | Overlapping kernel time per span |
| --- | ---: | ---: | ---: | ---: |
| K-pool plan HtoD from `_kpool_plan_to_gpu` (128 KiB) | 1 | 63.6 | 1,729 | 257 ms |
| Logits-processor BF16-to-FP32 copy (`bfloat16tofloat32_copy_kernel`) | 1 | 63.0 | 1,808 | 254 ms |

The HtoD event moves 131,088 bytes but spans 254.5 ms. The BF16-to-FP32 copy launches have a 6.3 µs median outside this span.

## Lever ranking — excluding all-reduce and parity/faster rows

| Rank | Lever | MI355X ms | % of total | Work class |
| ---: | --- | ---: | ---: | --- |
| 1 | Routed MoE expert kernels | 38.9 | 16.6% | Kernel work |
| 2 | Sparse MLA | 38.7 | 16.5% | Kernel work |
| 3 | mHC post | 14.9 | 6.4% | Kernel work |
| 4 | mHC pre sqrsum | 13.3 | 5.7% | Kernel work |
| 5 | KDA PTPC projections | 12.0 | 5.1% | Kernel work |
| 6 | KDA state/output | 10.6 | 4.5% | Kernel work |
| 7 | mHC pre RMSNorm | 9.3 | 4.0% | Kernel work |
| 8 | MoE routing / sort / quant | 8.2 | 3.5% | Kernel work |
| 9 | Causal convolution | 5.0 | 2.1% | Kernel work |
| 10 | KDA tensor materialization | 2.4 | 1.0% | Memory / launch work |

### Graphs-on formal cross-check

The formal capture uses the same source, model, image, TP, workload, selectors, and warmup, with production decode graphs enabled. It scored 95% on GSM8K-20 with zero errors. Its TP0 EXTEND trace recorded three of the four forwards (33 sparse-MLA launches versus 44 graphs-off), so its values are divided by three.

| Component | Graphs-off ms/fwd | Graphs-on ms/fwd | Interpretation |
| --- | ---: | ---: | --- |
| K-pool HtoD spans | 63.6 | 0.0 | Span present only in the graphs-off trace |
| Logits BF16-to-FP32 copy span | 63.0 | 0.0 | Span present only in the graphs-off trace |
| mHC pre RMSNorm | 9.3 | 8.9 | Close in both traces |
| Routed MoE | 38.9 | 37.3 | Close in both traces |
| Sparse MLA | 38.7 | 36.9 | Stable on the Triton path |
| KDA HIP chunked prefill | 10.6 | 10.1 | Close in both traces |
| **Summed event duration** | **360.6** | **226.1** | Diagnostic only; overlapping events are not GPU-busy time |
| **Summed event duration excluding spans** | **234.0** | **226.1** | Same work in both traces |

A stalled launch is one longer than 1 ms and more than 20 times its kernel's median duration; the graphs-on trace has none. Every KDA prefill call in both traces runs the HIP chunked-prefill kernels: 136 `kda_intra_kernel` launches graphs-off (34 KDA layers × 4 forwards) and 102 graphs-on (× 3 forwards). No Triton `chunk_kda` kernels appear.

Raw artifacts are under `/data2/macui/glm53/glm53_mxfp4_prefill_profile_1102` on `smci355-ccs-aus-m12-17`. The graphs-off TP0 EXTEND trace is `mapping/traces/1793649767.4666905/1793649767.468719-TP-0-EXTEND.trace.json.gz`; the graphs-on cross-check is `formal/traces/1793650757.128998/1793650757.1308966-TP-0-EXTEND.trace.json.gz`.

## Decode — i1k/o1k conc64, TP4

ms per decode step, TP0 DECODE trace, batch 64, `utilities/e2e_glm53_flash.sh 1024 1024 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, same source, image, AITER, model, and selectors as the prefill tables above. Timings come from the graphs-on formal capture; the ROCm tracer recorded kernels for one of its four decode-graph replays (34 KDA recurrence launches, one per KDA layer), so the table is that single step. Kernels launched by more than one module are split by their graphs-off per-module time ratio. Median ITL for the run is 12.17 ms (TPOT 12.47 ms in warmup); GSM8K-20 scored 100% (formal) and 95% (mapping).

| Component | MI355X kernel | MI355X ms/step | % of step | Launches |
| --- | --- | ---: | ---: | ---: |
| MoE gate-up | AITER FlyDSL A4W4 `gemm1_a4w4_port` | 2.92 | 21.6% | 42 |
| MoE down + combine | AITER FlyDSL A4W4 `gemm2_a4w4_port` | 1.54 | 11.4% | 42 |
| MoE routing / sort / quant / gate | OPUS sorting, grouped top-k, shared-expert append, gate GEMM, top-k casts | 1.29 | 9.5% | 252 |
| **MoE subtotal** | | **5.75** | **42.5%** | |
| Dense GEMMs | W8A16, block-FP8 and BF16 projection GEMMs | 2.13 | 15.8% | 219 |
| KDA recurrent state update | `kda_packed_decode_rowstream_kernel` | 1.01 | 7.4% | 34 |
| KDA causal conv update | `_causal_conv1d_update_kernel` | 0.18 | 1.3% | 34 |
| KDA gated RMSNorm | `layer_norm_gated_fwd_kernel` | 0.15 | 1.1% | 34 |
| Sparse MLA decode | Triton `_sparse_mla` decode and reduce | 0.36 | 2.6% | 22 |
| DSA indexer | top-k, MQA logits, Hadamard, cache | 0.21 | 1.5% | 44 |
| All-reduce | AITER custom all-reduce | 1.04 | 7.7% | 91 |
| mHC pre RMSNorm | `mhc_pre_big_fuse_rmsnorm_kernel` | 0.52 | 3.8% | 90 |
| mHC post | `mhc_post_kernel` | 0.43 | 3.1% | 90 |
| mHC pre sqrsum | `mhc_pre_gemm_sqrsum_kernel` | 0.42 | 3.1% | 90 |
| FP8 activation quant | AITER `dynamic_per_group_scaled_quant` | 0.15 | 1.1% | 36 |
| Other | allgather, elementwise, indexer index arithmetic, RMSNorm/RoPE, sampler, misc | 1.20 | 8.9% | 201 |
| **TOTAL** | | **13.54** | **100.0%** | **1321** |

Within Dense GEMMs, the KDA layers contribute the W8A16 first-stage projection (`_a16w8_preshuffle_kernel`, 0.53 ms/step), the BF16 `o_proj` (0.30 ms/step), and the `f_b`/`g_b` batched GEMM (0.17 ms/step).

The graphs-off mapping capture (`--profile --cuda-graph-backend-decode disabled`, so decode runs without graphs) recorded all four steps with module attribution. Excluding all-reduce, its kernel work is 13.4 ms/step versus 12.5 ms/step for the graphs-on step above; its all-reduce row (61.3 ms/step) is cross-rank waiting caused by eager launch overhead and is not a timing claim.

Raw artifacts are under `/data2/macui/glm53/glm53_mxfp4_decode_profile_1102_i1k_o1k` on `smci355-ccs-aus-m12-17`. The graphs-on trace is `formal/traces/1793653278.8364007/1793653278.8384068-TP-0-DECODE.trace.json.gz`; the graphs-off trace is `mapping/traces/1793652011.0785859/1793652011.080412-TP-0-DECODE.trace.json.gz`.
