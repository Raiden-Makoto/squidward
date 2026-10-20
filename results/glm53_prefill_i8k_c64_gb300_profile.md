# GLM-5.3-Flash prefill — i8k/o16 conc64, TP4 — GB300 (NVFP4 cookbook)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm53_flash.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm53` `84bd7ab188`, script snapshot `6aef5d5fc2`; cookbook GB300 High Throughput NVFP4 command (`--quantization modelopt_fp4 --tp-size 4 --dsa-prefill-backend trtllm --dsa-decode-backend trtllm --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_trtllm`) plus `--disable-radix-cache --disable-cuda-graph`, Triton linear-attention backend with FlashInfer KDA prefill, shared-expert fusion disabled by the server, NCCL all-reduce. 4× GB300 (driver 595.58.03, aarch64), image `lmsysorg/sglang:latest@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae` (`sglang-kernel 0.4.9`, `flashinfer 0.7.1`, `torch 2.14.1+cu130`); model snapshot `nvidia/GLM-5.3-Flash-NVFP4@da920bb0b9f4a06727223a349e55468e38352348`. GSM8K-20 scored 95% with zero errors and zero truncation.

The table values are summed profiler-event durations divided by four EXTEND forwards. They are not GPU-busy time: overlapping events can double-count, and the graphs-off all-reduce spans below include cross-rank waiting. Kernels are attributed with the MI355X run's `glm5_prof_csv.py`, extended to correlate `cuda_driver` launches and without its 5 s lookback cutoff; the 0.3 ms/forward that remains unattributed is counted under other KDA / dense kernels, as in the MI355X table.

## Attention

| Component | GB300 kernel | GB300 ms | % of total |
| --- | --- | ---: | ---: |
| K-pool plan transfer/wait | pinned and pageable HtoD | 0.0 | 0.0% |
| Sparse MLA | TRT-LLM `fmhaSm100f` sparse FMHA | 28.5 | 12.0% |
| DSA top-k | `kpool_topk_transform_kernel<512>` | 3.6 | 1.5% |
| DSA MQA logits | `deep_gemm::sm100_mqa_logits` | 0.8 | 0.4% |
| DSA prep/store + misc | norm/Hadamard/cache assembly and remaining attention kernels | 17.6 | 7.4% |
| **Attention subtotal** | | **50.6** | **21.2%** |

The 68 HtoD events total 0.09 ms across the four forwards, so the K-pool plan transfer does not appear as a queued dependency span in this trace.

## MoE

| Component | GB300 kernel | GB300 ms | % of total |
| --- | --- | ---: | ---: |
| Routed experts | TRT-LLM NVFP4 `bmm_E2m1` gate-up + `bmm_Bfloat16_E2m1` down + `finalizeKernelVecLoad` | 29.7 | 12.4% |
| Routing / sort / quant | routing, router GEMM, NVFP4 activation quant, shared-expert MLP + add | 11.0 | 4.6% |
| **MoE subtotal** | | **40.7** | **17.1%** |

## Dense GEMM

| Component | GB300 kernel | GB300 ms | % of total |
| --- | --- | ---: | ---: |
| KDA tensor materialization | BF16 direct-copy kernels in the KDA prefill path | 7.3 | 3.1% |
| KDA projections | BF16 `nvjet` GEMM for `fused_qkvbfg_a_proj` | 12.5 | 5.2% |
| KDA state/output | FlashInfer KDA prefill `kernel_cutlass_kernel_chain_dv2` | 7.9 | 3.3% |
| Causal convolution | `_causal_conv1d_fwd_kernel` | 3.9 | 1.6% |
| DenseMLP L0–2 | dense NVFP4 MLP | 0.8 | 0.3% |
| Other KDA / dense kernels | FlashInfer KDA prep, `o_proj` and `fused_fg_b_proj` GEMMs, gated RMSNorm, misc | 9.7 | 4.1% |
| **Dense/KDA subtotal** | | **42.1** | **17.7%** |

## Communication and normalization

| Component | GB300 kernel | GB300 ms | % of total |
| --- | --- | ---: | ---: |
| All-reduce | NCCL `AllReduce_Sum_bf16_RING_LL` | 73.6 | 30.9% |
| Additional reduction | none in trace | 0.0 | 0.0% |
| mHC pre RMSNorm | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 9.3 | 3.9% |
| mHC post | `mhc_post_tilelang_kernel` | 13.7 | 5.7% |
| mHC pre sqrsum | `deep_gemm::sm100_tf32_hc_prenorm_gemm_impl` | 8.4 | 3.5% |
| **Communication/norm subtotal** | | **105.0** | **44.1%** |
| **TOTAL** | | **238.3** | **100.0%** |

The all-reduce value is not a production timing claim. The same 364 NCCL calls total 42.2 ms/forward in the warmed graphs-on formal trace, so the 73.6 ms graphs-off value includes cross-rank waiting.

## Lever ranking — excluding all-reduce and parity/faster rows

The ranking uses the same rows as the MI355X ranking. mHC pre RMSNorm is added because the graphs-on cross-check reproduces it as standalone kernel work.

| Rank | Lever | GB300 ms | % of total | Work class |
| ---: | --- | ---: | ---: | --- |
| 1 | Routed MoE expert kernel | 29.7 | 12.4% | Kernel work |
| 2 | Sparse MLA | 28.5 | 12.0% | Kernel work |
| 3 | mHC post | 13.7 | 5.7% | Kernel work |
| 4 | KDA projections | 12.5 | 5.2% | Kernel work |
| 5 | mHC pre RMSNorm | 9.3 | 3.9% | Kernel work |
| 6 | mHC pre sqrsum | 8.4 | 3.5% | Kernel work |
| 7 | KDA state/output | 7.9 | 3.3% | Kernel work |
| 8 | KDA tensor materialization | 7.3 | 3.1% | Memory / launch work |
| 9 | Causal convolution | 3.9 | 1.6% | Kernel work |

### Graphs-on formal cross-check

The formal capture uses the same source, model, image, TP, workload, and warmup, with the cookbook command's CUDA graphs enabled (no `--disable-cuda-graph`). It also scored 100% on GSM8K-20 with zero errors. Both captures ran the same four EXTEND batches (8,192, 16,309, and 2× 16,384 tokens), which exceed the captured prefill graph shapes, so the EXTEND trace records no graph launches.

| Component | Graphs-off ms/fwd | Graphs-on ms/fwd | Interpretation |
| --- | ---: | ---: | --- |
| K-pool HtoD spans | 0.0 | 0.0 | No material transfer or queued span |
| All-reduce | 73.6 | 42.2 | Graphs-off span includes cross-rank waiting |
| Routed MoE | 29.7 | 29.6 | Stable kernel work |
| Sparse MLA | 28.5 | 28.5 | Stable kernel work |
| mHC pre RMSNorm | 9.3 | 9.3 | Stable; not an eager-trace artifact |
| KDA state/output | 7.9 | 7.9 | Stable kernel work |
| **Summed event duration** | **238.3** | **206.4** | Diagnostic only; overlapping events are not GPU-busy time |

KDA extend dispatches to `FlashInferKDAPrefillKernel` (decode and verify use `TritonKDAKernel`), so the MI355X `_recompute_w_u_fwd_kernel` path does not run on GB300. The FlashInfer chain and prep kernels together total 10.4 ms/forward.

Raw artifacts are under `/home/macui/hf_home/traces/glm53_flash_gb300_i8k_o16_20261020` on `pod4-gb300-2-tray02-f3`. The graphs-off TP0 EXTEND trace is `attr/traces/1792516814.6633525/1792516814.6653461-TP-0-EXTEND.trace.json.gz`; the graphs-on cross-check is `formal/traces/1792517444.5163414/1792517444.5180416-TP-0-EXTEND.trace.json.gz`.
