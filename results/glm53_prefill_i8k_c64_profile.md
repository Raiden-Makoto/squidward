# GLM-5.3-Flash prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4 integrated base, tuned MoE, fused KDA prefill)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm53_flash.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm53` `bb1a9c3718` (includes [#45913](https://github.com/sgl-project/sglang/pull/45913)) run from `/sgl-workspace/squidward` in `macui-glm53-new` on `smci355-ccs-aus-m15-21` with `utilities/run_glm53_flash.sh --mxfp4`, script snapshot `6aef5d5fc2`, AITER `313f86a7d` (head of the ROCm/aiter#6672 branch, which includes #5866) with the [#6435](https://github.com/ROCm/aiter/pull/6435) MXFP4 MoE configs and the four 16384-token rows posted there; `SGLANG_OPT_KDA_FUSED_PREFILL=1`, `SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj`, `SGLANG_OPT_GLM53_KDA_W8A16_DECODE=1`, `SGLANG_DSA_FUSE_HADAMARD_QUANT=1`, `aiter_sparse_mla` DSA prefill/decode backends, shared-expert fusion enabled, INT4 QuickReduce. MI355X image `rocm/sgl-dev:v0.5.22-rocm724-mi35x-20261019`; model snapshot `amd/GLM-5.3-Flash-Quark-MXFP4@b5688f25491202978c19c4d036eef579f61bbe07`. GSM8K-20 scored 95% with zero errors.

The table values are summed profiler-event durations divided by four EXTEND forwards. They exclude eight trace spans that overlap other GPU work for their whole duration; those spans are listed under [Overlapping trace spans](#overlapping-trace-spans) and are not counted in any row or total.

## Attention

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| K-pool plan transfer/wait | pinned HtoD from `_kpool_plan_to_gpu` | 0.0 | 0.0% |
| MLA FP8 activation quant | AITER `dynamic_per_group_scaled_quant` from `aiter_w8a8_block_fp8_linear` | 0.9 | 0.4% |
| Sparse MLA | Triton `_sparse_mla_fwd_split_dim_kernel` | 37.6 | 15.9% |
| DSA top-k | `kpool_topk_transform_kernel<512>` | 2.4 | 1.0% |
| DSA MQA logits | Gluon FP8 MQA logits | 0.8 | 0.4% |
| DSA prep/store + misc | MLA and indexer projection GEMMs, fused Hadamard quant, norm/cache assembly, remaining attention kernels | 13.6 | 5.8% |
| **Attention subtotal** | | **55.3** | **23.5%** |

## MoE

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| Routed experts | AITER FlyDSL A4W4 `gemm1_a4w4_port` and `gemm2_a4w4_port`, plus `scatter_reduce` / `moe_reduction` | 37.1 | 15.7% |
| Routing / sort / quant | OPUS sorting, MXFP4 MoE sort, gate GEMM, grouped top-k, shared-expert append, FP4 activation quant | 8.0 | 3.4% |
| **MoE subtotal** | | **45.1** | **19.1%** |

Every MoE token bucket dispatches a tuned row; the 16384-token bucket uses `flydsl_mxmoe_g1_a4w4_128x256x256_xcd4` with `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_scatter_sbm128`.

## Dense GEMM

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| KDA tensor materialization | BF16 direct-copy kernels in `chunk_kda` | 0.4 | 0.2% |
| KDA PTPC projections | AITER per-token/per-channel FP8 GEMM | 11.4 | 4.8% |
| KDA state/output | AITER fused `chunk_kda_walk` and `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 10.7 | 4.5% |
| Causal convolution | `_causal_conv1d_fwd_kernel` | 5.0 | 2.1% |
| DenseMLP L0–2 | dense block-FP8 MLP | 2.0 | 0.8% |
| Other KDA / dense kernels | AITER `chunk_kda_prepare`, remaining Triton chunk kernels, output GEMMs, gated RMSNorm, logits/embedding/sampler, misc | 18.2 | 7.7% |
| **Dense/KDA subtotal** | | **47.7** | **20.2%** |

## Communication and normalization

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| All-reduce | QuickReduce INT4 | 38.0 | 16.1% |
| Additional reduction | AITER `cross_device_reduce_2stage` | 13.9 | 5.9% |
| mHC pre RMSNorm | `mhc_pre_big_fuse_rmsnorm_kernel` | 8.9 | 3.8% |
| mHC post | `mhc_post_kernel` | 14.2 | 6.0% |
| mHC pre sqrsum | `mhc_pre_gemm_sqrsum_kernel` | 12.8 | 5.4% |
| **Communication/norm subtotal** | | **87.8** | **37.2%** |
| **TOTAL** | | **235.9** | **100.0%** |
| GPU-busy time (union of all GPU events) | | 242.7 | |

The mHC pre RMSNorm value matches the graphs-on formal trace (8.9 ms/forward in both).

## Overlapping trace spans

These events are recorded with long durations while the GPU keeps executing other kernels throughout, so their recorded time is a span, not GPU time. Including them gives the 549.4 ms/forward summed-event total.

| Span | Events | Recorded ms/fwd | Overlapping kernels per span | Overlapping kernel time per span |
| --- | ---: | ---: | ---: | ---: |
| K-pool plan HtoD from `_kpool_plan_to_gpu` (64–128 KiB each) | 4 | 192.6 | 1,421–1,460 | 133–304 ms |
| MoE sorting (`opus_moe_sorting_entry` P23) | 3 | 104.4 | 926–998 | 224–226 ms |
| MoE sorting (`mxfp4_moe_sort_kernel`) | 1 | 16.6 | 797 | 73 ms |

The four HtoD events move only 64–128 KiB each but span 127–215 ms, an apparent bandwidth of 0.0005–0.0006 GB/s. The P23 sorting launches have a 55.9 µs median and the MXFP4 sort launches a 25.0 µs median outside these spans.

## Lever ranking — excluding all-reduce and parity/faster rows

| Rank | Lever | MI355X ms | % of total | Work class |
| ---: | --- | ---: | ---: | --- |
| 1 | Sparse MLA | 37.6 | 15.9% | Kernel work |
| 2 | Routed MoE expert kernels | 37.1 | 15.7% | Kernel work |
| 3 | mHC post | 14.2 | 6.0% | Kernel work |
| 4 | mHC pre sqrsum | 12.8 | 5.4% | Kernel work |
| 5 | KDA PTPC projections | 11.4 | 4.8% | Kernel work |
| 6 | KDA state/output | 10.7 | 4.5% | Kernel work |
| 7 | mHC pre RMSNorm | 8.9 | 3.8% | Kernel work |
| 8 | MoE routing / sort / quant | 8.0 | 3.4% | Kernel work |
| 9 | Causal convolution | 5.0 | 2.1% | Kernel work |
| 10 | KDA tensor materialization | 0.4 | 0.2% | Memory / launch work |

### Graphs-on formal cross-check

The formal capture uses the same source, model, image, TP, workload, selectors, and warmup, with production decode graphs enabled. It scored 100% on GSM8K-20 with zero errors.

| Component | Graphs-off ms/fwd | Graphs-on ms/fwd | Interpretation |
| --- | ---: | ---: | --- |
| K-pool HtoD spans | 192.6 | 199.0 | Spans that overlap other kernels, not copy bandwidth |
| Other stalled launches | 120.9 | 116.8 | Spans land on different kernels: MoE sorting graphs-off; MLA `add_rmsnorm_quant`, `moe_reduction`, and `alloc_extend_kernel` graphs-on |
| mHC pre RMSNorm | 8.9 | 8.9 | Same in both traces |
| Routed MoE | 37.1 | 37.3 | Same in both traces (graphs-on excludes its `moe_reduction` span) |
| Sparse MLA | 37.6 | 38.1 | Stable on the Triton path |
| KDA fused walk | 8.8 | 8.8 | Same in both traces |
| **Summed event duration** | **549.4** | **551.8** | Diagnostic only; overlapping events are not GPU-busy time |
| **Summed event duration excluding stalls** | **235.9** | **236.1** | Same work in both traces |

A stalled launch is one longer than 1 ms and more than 20 times its kernel's median duration. Both traces contain 102 fused `chunk_kda_prepare` and 102 `chunk_kda_walk` launches, covering three of the four forwards. The remaining forward falls outside #45913's dispatch cells and runs the separate path, including 34 `_recompute_w_u_fwd_kernel` launches (1.6 ms across four forwards).

Raw artifacts are under `/data2/hf_home/glm53_mxfp4_prefill_profile_1023_tunedmoe_45913` on `smci355-ccs-aus-m15-21`. The graphs-off TP0 EXTEND trace is `mapping/traces/1792779655.9427779/1792779655.9446695-TP-0-EXTEND.trace.json.gz`; the graphs-on cross-check is `formal/traces/1792780709.9467244/1792780709.9485676-TP-0-EXTEND.trace.json.gz`.

## Decode — i1k/o1k conc64, TP4

ms per decode step, TP0 DECODE trace, batch 64, `utilities/e2e_glm53_flash.sh 1024 1024 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, same source, image, AITER, model, and selectors as the prefill tables above. Timings come from the graphs-on formal capture; the ROCm tracer recorded kernels for one of its four decode-graph replays (the last), so the table is that single step. Median ITL for the run is 12.1 ms (TPOT 12.5 ms); GSM8K-20 scored 95% (formal) and 100% (mapping).

| Component | MI355X kernel | MI355X ms/step | % of step | Launches |
| --- | --- | ---: | ---: | ---: |
| MoE gate-up | AITER FlyDSL A4W4 `gemm1_a4w4_port` | 2.97 | 22.3% | 42 |
| MoE down + combine | AITER FlyDSL A4W4 `gemm2_a4w4_port` | 1.55 | 11.6% | 42 |
| MoE routing / sort / quant / gate | OPUS sorting, grouped top-k, shared-expert append, FP4 activation quant | 1.02 | 7.7% | 222 |
| **MoE subtotal** | | **5.54** | **41.6%** | |
| Dense GEMMs | W8A16, PTPC FP8, block-FP8 and BF16 projection GEMMs | 2.39 | 17.9% | 239 |
| KDA recurrent state update | `kda_packed_decode_rowstream_kernel` | 0.98 | 7.4% | 34 |
| KDA causal conv update | `_causal_conv1d_update_kernel` | 0.17 | 1.3% | 34 |
| KDA gated RMSNorm | `layer_norm_gated_fwd_kernel` | 0.15 | 1.1% | 34 |
| Sparse MLA decode | Triton `_sparse_mla` decode | 0.36 | 2.7% | 22 |
| DSA indexer | top-k, MQA logits, Hadamard, cache | 0.38 | 2.9% | 75 |
| All-reduce | AITER custom all-reduce | 1.10 | 8.2% | 91 |
| mHC pre RMSNorm | `mhc_pre_big_fuse_rmsnorm_kernel` | 0.51 | 3.8% | 90 |
| mHC post | `mhc_post_kernel` | 0.42 | 3.2% | 90 |
| mHC pre sqrsum | `mhc_pre_gemm_sqrsum_kernel` | 0.42 | 3.1% | 90 |
| FP8 activation quant | AITER `dynamic_per_group_scaled_quant` | 0.15 | 1.1% | 36 |
| Other | allgather, elementwise, RMSNorm/RoPE, sampler, misc | 0.75 | 5.6% | 141 |
| **TOTAL** | | **13.34** | **100.0%** | **1282** |

The graphs-off mapping capture (`--profile --cuda-graph-backend-decode disabled`, so decode runs without graphs) recorded all four steps with module attribution. Excluding all-reduce, its kernel work is 13.1 ms/step versus 12.2 ms/step for the graphs-on step above; its all-reduce row (34.5 ms/step) is cross-rank waiting caused by eager launch overhead and is not a timing claim.

Raw artifacts are under `/data2/hf_home/glm53_mxfp4_decode_profile_1023_i1k_o1k_45913` on `smci355-ccs-aus-m15-21`. The graphs-on trace is `formal/traces/1792789147.3445837/1792789147.3463824-TP-0-DECODE.trace.json.gz`; the graphs-off trace is `mapping/traces/1792787765.026541/1792787765.0284014-TP-0-DECODE.trace.json.gz`.
