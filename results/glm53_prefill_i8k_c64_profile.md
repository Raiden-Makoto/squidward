# GLM-5.3-Flash prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4 integrated base)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm53_flash.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm53` `e593c7ebdd` run from `/sgl-workspace/squidward` in `macui-glm53` with `utilities/run_glm53_flash.sh --mxfp4`, sanctioned script snapshot `6aef5d5fc2`, AITER `e7d2453f25` (image checkout with the container's local modifications); `SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj`, `SGLANG_OPT_GLM53_KDA_W8A16_DECODE=1`, `SGLANG_DSA_FUSE_HADAMARD_QUANT=1`, `aiter_sparse_mla` DSA prefill/decode backends, shared-expert fusion enabled, INT4 QuickReduce. MI355X image `rocm/sgl-dev:v0.5.21-rocm724-mi35x-20261014`; model snapshot `amd/GLM-5.3-Flash-Quark-MXFP4@b5688f25491202978c19c4d036eef579f61bbe07`. GSM8K-20 scored 95% with zero errors and zero truncation.

The table values are summed profiler-event durations divided by four EXTEND forwards. They are not GPU-busy time: overlapping events can double-count, and the K-pool HtoD spans and the stalled launches noted below include dependency waiting.

## Attention

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| K-pool plan transfer/wait | pinned HtoD from `_kpool_plan_to_gpu` | 202.4 | 33.2% |
| MLA FP8 activation quant | AITER `dynamic_per_group_scaled_quant` from `aiter_w8a8_block_fp8_linear` | 129.4 | 21.2% |
| Sparse MLA | Triton `_sparse_mla_fwd_split_dim_kernel` | 38.7 | 6.3% |
| DSA top-k | `kpool_topk_transform_kernel<512>` | 2.4 | 0.4% |
| DSA MQA logits | Gluon FP8 MQA logits | 0.9 | 0.1% |
| DSA prep/store + misc | MLA and indexer projection GEMMs, fused Hadamard quant, norm/cache assembly, remaining attention kernels | 13.9 | 2.3% |
| **Attention subtotal** | | **387.8** | **63.6%** |

The four large HtoD events move only 64–128 KiB each but span 141–223 ms in the eager trace. Their computed bandwidth is approximately 0.0005–0.0006 GB/s, showing that the recorded duration is dominated by queued dependency wait rather than transfer work.

The MLA FP8 activation quant row is likewise wait-dominated: 3 of its 132 launches span 514.1 ms in total (128.5 ms/forward). Across all 144 launches of this quant kernel, including the 12 in DenseMLP, the other 141 total 4.0 ms with a 37.6 µs median.

## MoE

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| Routed experts | AITER MXFP4 `mfma_moe1_silu_mul_afp4_wfp4` and `mfma_moe2_afp4_wfp4` | 50.5 | 8.3% |
| Routing / sort / quant | MXFP4 MoE sort, OPUS sorting, gate GEMM, grouped top-k, shared-expert append, FP4 activation quant | 26.5 | 4.3% |
| **MoE subtotal** | | **77.0** | **12.6%** |

One `mxfp4_moe_sort_kernel` launch spans 69.5 ms (17.4 ms/forward) against a 6.2 µs median, so 9.1 ms/forward of the routing row is kernel work.

## Dense GEMM

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| KDA tensor materialization | BF16 direct-copy kernels in `chunk_kda` | 2.5 | 0.4% |
| KDA PTPC projections | AITER per-token/per-channel FP8 GEMM | 12.0 | 2.0% |
| KDA state/output | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 11.1 | 1.8% |
| Causal convolution | `_causal_conv1d_fwd_kernel` | 5.0 | 0.8% |
| DenseMLP L0–2 | dense block-FP8 MLP | 2.0 | 0.3% |
| Other KDA / dense kernels | intra/inter solve, recompute, output GEMMs, gated RMSNorm, logits/embedding/sampler, misc | 26.6 | 4.4% |
| **Dense/KDA subtotal** | | **59.2** | **9.7%** |

## Communication and normalization

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| All-reduce | QuickReduce INT4 | 34.6 | 5.7% |
| Additional reduction | AITER `cross_device_reduce_2stage` | 14.2 | 2.3% |
| mHC pre RMSNorm | `mhc_pre_big_fuse_rmsnorm_kernel` | 9.3 | 1.5% |
| mHC post | `mhc_post_kernel` | 14.8 | 2.4% |
| mHC pre sqrsum | `mhc_pre_gemm_sqrsum_kernel` | 13.1 | 2.1% |
| **Communication/norm subtotal** | | **86.0** | **14.1%** |
| **TOTAL** | | **610.0** | **100.0%** |

The mHC pre RMSNorm value matches the graphs-on formal trace (9.3 ms/forward in both).

## Lever ranking — excluding all-reduce and parity/faster rows

The queued HtoD spans, the MLA FP8 activation quant row, and the MoE routing row are also excluded because their totals are dominated by stalled launches rather than standalone kernel work.

| Rank | Lever | MI355X ms | % of total | Work class |
| ---: | --- | ---: | ---: | --- |
| 1 | Routed MoE expert kernels | 50.5 | 8.3% | Kernel work |
| 2 | Sparse MLA | 38.7 | 6.3% | Kernel work |
| 3 | mHC post | 14.8 | 2.4% | Kernel work |
| 4 | mHC pre sqrsum | 13.1 | 2.1% | Kernel work |
| 5 | KDA PTPC projections | 12.0 | 2.0% | Kernel work |
| 6 | KDA state/output | 11.1 | 1.8% | Kernel work |
| 7 | mHC pre RMSNorm | 9.3 | 1.5% | Kernel work |
| 8 | Causal convolution | 5.0 | 0.8% | Kernel work |
| 9 | KDA tensor materialization | 2.5 | 0.4% | Memory / launch work |

### Graphs-on formal cross-check

The formal capture uses the same source, model, image, TP, workload, selectors, and warmup, with production decode graphs enabled. It also scored 95% on GSM8K-20 with zero errors.

| Component | Graphs-off ms/fwd | Graphs-on ms/fwd | Interpretation |
| --- | ---: | ---: | --- |
| K-pool HtoD spans | 202.4 | 199.9 | Queued dependency spans, not copy bandwidth |
| Other stalled launches | 145.9 | 149.1 | Waits land on different kernels: FP8 activation quant and MoE sort graphs-off; TopK BF16-to-FP32 copy and KDA state graphs-on |
| mHC pre RMSNorm | 9.3 | 9.3 | Same in both traces |
| Routed MoE | 50.5 | 50.5 | Same in both traces |
| Sparse MLA | 38.7 | 38.6 | Stable on the Triton path |
| KDA recompute W/U | 3.1 | 3.1 | Same in both traces |
| **Summed event duration** | **610.0** | **611.0** | Diagnostic only; overlapping events are not GPU-busy time |
| **Summed event duration excluding stalls** | **261.7** | **261.9** | Same work in both traces |

A stalled launch is one longer than 1 ms and more than 20 times its kernel's median duration. Both traces contain 136 `_recompute_w_u_fwd_kernel` launches (12.5 ms and 12.6 ms across four forwards). The current gfx950 fused-intra dispatch requires `q.shape[0] == 1` and `len(cu_seqlens) == 2`, so multi-sequence batches retain the separate recompute path.

Raw artifacts are under `/data2/hf_home/glm53_mxfp4_prefill_profile_1020_e593c7eb`. The graphs-off TP0 EXTEND trace is `mapping/traces/1792519009.092709/1792519009.0953023-TP-0-EXTEND.trace.json.gz`; the graphs-on cross-check is `formal/traces/1792520087.8017945/1792520087.8042243-TP-0-EXTEND.trace.json.gz`.
