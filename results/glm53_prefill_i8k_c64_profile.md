# GLM-5.3-Flash prefill — i8k/o16 conc64, TP4 — MI355X (FP8 integrated base)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm53_flash.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm53` `73e4ac62e1`, sanctioned script snapshot `6aef5d5fc2`, AITER `e7d2453f25`; `SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj`, shared-expert fusion enabled, INT4 QuickReduce. MI355X image `rocm/sgl-dev:v0.5.21-rocm724-mi35x-20261001`; model snapshot `zai-org/GLM-5.3-Flash@03eb5366286afd40d2221b1d9c63a6dd1ba4832e`. GSM8K-20 scored 95% with zero errors and zero truncation.

The table values are summed profiler-event durations divided by four EXTEND forwards. They are not GPU-busy time: overlapping events can double-count, and the K-pool HtoD spans below include dependency waiting.

## Attention

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| K-pool plan transfer/wait | pinned HtoD from `_kpool_plan_to_gpu` | 286.0 | 35.1% |
| Sparse MLA | Triton `_sparse_mla_fwd_split_dim_kernel` | 38.3 | 4.7% |
| DSA top-k | `kpool_topk_transform_kernel<512>` | 2.4 | 0.3% |
| DSA MQA logits | Gluon FP8 MQA logits | 0.9 | 0.1% |
| DSA prep/store + misc | norm/Hadamard/cache assembly and remaining attention kernels | 16.7 | 2.0% |
| **Attention subtotal** | | **344.3** | **42.2%** |

The four large HtoD events move only 64–128 KiB each but span 177–323 ms in the eager trace. Their computed bandwidth is approximately 0.00036 GB/s, showing that the recorded duration is dominated by queued dependency wait rather than transfer work.

## MoE

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| Routed experts | AITER `fmoe_bf16_blockscaleFp8_g1u1_vs_ps_silu_64x256` | 90.3 | 11.1% |
| Routing / sort / quant | OPUS P0/P1/P23, grouped top-k, shared-expert append, activation quant | 7.5 | 0.9% |
| **MoE subtotal** | | **97.8** | **12.0%** |

## Dense GEMM

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| KDA tensor materialization | BF16 direct-copy kernels in `chunk_kda` | 24.7 | 3.0% |
| KDA PTPC projections | AITER per-token/per-channel FP8 GEMM | 12.1 | 1.5% |
| KDA state/output | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 11.1 | 1.4% |
| Causal convolution | `_causal_conv1d_fwd_kernel` | 5.0 | 0.6% |
| DenseMLP L0–2 | dense BF16 MLP | 2.1 | 0.3% |
| Other KDA / dense kernels | fused intra solve, recompute, output GEMMs, gated RMSNorm, misc | 27.5 | 3.4% |
| **Dense/KDA subtotal** | | **82.5** | **10.1%** |

## Communication and normalization

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| All-reduce | QuickReduce INT4 | 58.6 | 7.2% |
| Additional reduction | AITER `cross_device_reduce_2stage` | 13.9 | 1.7% |
| mHC pre RMSNorm | `mhc_pre_big_fuse_rmsnorm_kernel` | 191.1 | 23.4% |
| mHC post | `mhc_post_kernel` | 14.7 | 1.8% |
| mHC pre sqrsum | `mhc_pre_gemm_sqrsum_kernel` | 13.0 | 1.6% |
| **Communication/norm subtotal** | | **291.3** | **35.7%** |
| **TOTAL** | | **815.8** | **100.0%** |

The mHC pre value is not a production timing claim. The identical kernel totals 9.3 ms/forward in the warmed graphs-on formal trace, so the 191.1 ms eager value is excluded from optimization ranking pending a matched eager replication.

## Lever ranking — excluding all-reduce and parity/faster rows

The queued HtoD spans and the mHC eager artifact are also excluded because the graphs-on cross-check shows that neither is standalone kernel work.

| Rank | Lever | MI355X ms | % of total | Work class |
| ---: | --- | ---: | ---: | --- |
| 1 | Routed MoE expert kernel | 90.3 | 11.1% | Kernel work |
| 2 | Sparse MLA | 38.3 | 4.7% | Kernel work |
| 3 | KDA tensor materialization | 24.7 | 3.0% | Memory / launch work |
| 4 | mHC post | 14.7 | 1.8% | Kernel work |
| 5 | mHC pre sqrsum | 13.0 | 1.6% | Kernel work |
| 6 | KDA PTPC projections | 12.1 | 1.5% | Kernel work |
| 7 | KDA state/output | 11.1 | 1.4% | Kernel work |
| 8 | Causal convolution | 5.0 | 0.6% | Kernel work |

### Graphs-on formal cross-check

The formal capture uses the same source, model, image, TP, workload, selector, and warmup, with production decode graphs enabled. It also scored 95% on GSM8K-20 with zero errors.

| Component | Graphs-off ms/fwd | Graphs-on ms/fwd | Interpretation |
| --- | ---: | ---: | --- |
| K-pool HtoD spans | 286.0 | 320.0 | Queued dependency spans, not copy bandwidth |
| mHC pre RMSNorm | 191.1 | 9.3 | Eager-trace artifact; not a production lever |
| Routed MoE | 90.3 | 109.8 | Batch/scheduling-sensitive kernel work |
| Sparse MLA | 38.3 | 38.0 | Stable and already on the integrated Triton path |
| KDA recompute W/U | 3.4 | 177.0 | Formal c64 batching misses the single-sequence fused-intra envelope |
| **Summed event duration** | **815.8** | **839.2** | Diagnostic only; overlapping events are not GPU-busy time |

The formal trace contains 136 `_recompute_w_u_fwd_kernel` launches totaling 708.1 ms across four forwards. The current gfx950 fused-intra dispatch requires `q.shape[0] == 1` and `len(cu_seqlens) == 2`, so multi-sequence production batching retains the separate recompute path. Extending that path requires separate correctness and per-cell performance validation; the trace does not justify enabling it broadly.

Raw artifacts are under `/data2/hf_home/glm53_base_profile_20261002_73e4ac62`. The graphs-off TP0 EXTEND trace is `mapping/traces/1790966421.4288876/1790966421.4307415-TP-0-EXTEND.trace.json.gz`; the graphs-on cross-check is `formal/traces/1790966864.3698053/1790966864.3719068-TP-0-EXTEND.trace.json.gz`.
