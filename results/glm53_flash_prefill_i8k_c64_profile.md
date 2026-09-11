# GLM-5.3-Flash prefill — i8k/o16 conc64, TP4 — MI355X FP8

ms/prefill forward, TP0 EXTEND trace (`n_fwd=4`), graphs-off eager, random 8192/16, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm53-day0-main-integration` tested tree `9d0d62b744` / current attribution-only rewrite `fb57dfc032`, AITER `d9e5ef7ce0`. Image `lmsysorg/sglang@sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72`; BF16 KV cache, TileLang sparse DSA, AITER MoE.

The stage profiler emitted separate EXTEND and DECODE traces. The EXTEND trace contains four complete prefill forwards: the KDA state-update kernel appears 136 times, exactly 34 KDA layers × four forwards. All aggregate GPU times are divided by four. Three KDA events report impossible 197.7–204.5 ms durations while later kernels begin on the same stream only 3–63 µs after their start. The tables replace those three correlation errors with the median normal launch duration. Raw trace time is 840.2 ms/forward; corrected attributable GPU time is 689.4 ms/forward. The profiling run's serving latency and throughput are intentionally excluded because profiler collection and trace flushing perturb wall-clock measurements.

## Attention and hybrid linear attention

| Component | MI355X kernel/path | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| K-pool plan CPU→GPU | 28 `Memcpy HtoD` calls from `_kpool_plan_to_gpu` | 270.8 | 39.3% |
| KDA state update | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` plus KDA support kernels | 36.2 | 5.2% |
| Sparse DSA attention | TileLang `main_kernel` plus KV gather/concatenation | 70.2 | 10.2% |
| DSA k-pool indexer | top-k transform, Hadamard, FP8 quantization and MQA logits | 8.3 | 1.2% |
| Attention projections | replicated, row-parallel and column-parallel attention linears | 9.0 | 1.3% |
| Attention norm / misc | q/k norm, RoPE-adjacent and layernorm work | 2.9 | 0.4% |
| **Attention + KDA subtotal** |  | **397.4** | **57.6%** |

### KDA detail

| Component | MI355X kernel | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| State update | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 12.0 | 1.7% |
| Causal convolution | `_causal_conv1d_fwd_kernel` | 5.7 | 0.8% |
| Output accumulation | `chunk_gla_fwd_kernel_o` | 3.5 | 0.5% |
| Recompute W/U | `_recompute_w_u_fwd_kernel` | 3.1 | 0.4% |
| Intra-chunk solve | `chunk_kda_fwd_kernel_intra_sub_chunk` | 3.0 | 0.4% |
| Inter-chunk solve | `chunk_kda_fwd_kernel_inter_solve_fused` | 2.1 | 0.3% |
| Gate cumsum | `kda_gate_chunk_cumsum_vector_kernel` | 1.9 | 0.3% |
| L2 norm | `l2norm_fwd_kernel` | 1.5 | 0.2% |
| Copies, fills and elementwise support | mixed | 3.3 | 0.5% |
| **KDA subtotal** |  | **36.2** | **5.2%** |

### Sparse DSA detail

| Component | MI355X kernel/path | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| Sparse attention | TileLang `main_kernel` | 67.0 | 9.7% |
| KV concatenation / gather support | HIP cat/copy kernels | 3.2 | 0.5% |
| Fused k-pool top-k transform | `kpool_topk_transform_kernel<512>` | 2.5 | 0.4% |
| Indexer Hadamard | `fast_hadamard_transform_kernel` | 1.7 | 0.2% |
| Indexer query projection | `triton_poi_fused__to_copy_gemm_a16w16_0` | 1.1 | 0.1% |
| FP8 MQA logits | `_gluon_fp8_mqa_logits_kernel` | 1.1 | 0.1% |
| Indexer quantization and prep | `act_quant`, fill, assemble and gather kernels | 2.0 | 0.3% |
| Attention projections / normalization | CK/AITER GEMMs and norm kernels | 12.0 | 1.7% |
| **Sparse DSA subtotal** |  | **90.5** | **13.1%** |

## MoE

| Component | MI355X kernel/path | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| Routed experts | `aiter::fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256` | 97.2 | 14.1% |
| Shared expert MLP | block-FP8 gate-up/down GEMMs, quantization and activation | 8.4 | 1.2% |
| Gate / router GEMMs | CK blockscale GEMMs | 2.1 | 0.3% |
| Sorting and grouped top-k | AITER OPUS sort phases plus grouped top-k | 2.7 | 0.4% |
| Routed activation quantization | `dynamic_per_group_scaled_quant` | 1.6 | 0.2% |
| Output add / copies | BF16 add and DtoD copy kernels | 2.4 | 0.3% |
| **MoE subtotal** |  | **114.3** | **16.6%** |

## Dense projections and dense MLP

| Component | MI355X kernel/path | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| Row-parallel projections | CK/AITER BF16 and blockscale GEMMs | 39.0 | 5.7% |
| QKV projections | CK blockscale GEMM | 16.2 | 2.3% |
| Column / replicated projections | CK blockscale GEMMs | 4.4 | 0.6% |
| Dense MLP layers 0–2 | block-FP8 gate-up/down, quantization and activation | 2.1 | 0.3% |
| **Dense subtotal** |  | **61.7** | **9.0%** |

## Communication, mHC and remainder

| Component | MI355X kernel/path | MI355X ms | % of total |
| --- | --- | ---: | ---: |
| All-reduce | QuickReduce INT4 plus AITER cross-device reduction | 74.7 | 10.8% |
| mHC pre/post | AITER mHC pre GEMM/square-sum, fused RMSNorm and post | 37.6 | 5.5% |
| Output head, embedding, sampling and other | mixed | 3.7 | 0.5% |
| **CORRECTED TOTAL** |  | **689.4** | **100.0%** |

## Lever ranking

| Rank | Lever | MI355X ms | % of total | Work class | Evidence |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | K-pool plan CPU→GPU handoff | 270.8 | 39.3% | Metadata / transfer | 28 HtoD copies map to `kpool_plan.py:_kpool_plan_to_gpu` |
| 2 | Routed-expert FP8 MoE | 97.2 | 14.1% | Kernel work | AITER fused MoE main kernel |
| 3 | All-reduce | 74.7 | 10.8% | Communication | QuickReduce plus AITER cross-device reduction |
| 4 | TileLang sparse attention | 67.0 | 9.7% | Kernel work | 88 TileLang sparse-attention launches |
| 5 | Dense / unquantized GEMM family | 54.1 | 7.8% | Kernel work | CK GEMMs attributed mainly to unquantized linears |
| 6 | mHC pre/post/norm | 36.9 | 5.4% | Kernel work | 360 launches each across pre/post paths |
| 7 | Complete KDA path | 36.2 | 5.2% | Kernel work | State update is 12.0 ms after removing correlation errors |
| 8 | AITER cross-device reduction | 13.9 | 2.0% | Communication | 91 two-stage reductions |
| 9 | QK/RoPE/KV write family | 11.1 | 1.6% | Fusion candidate | Profiler catalog finds an existing matching fused path |

The first hypothesis to validate is the K-pool plan handoff. The trace maps 39.3% of corrected prefill GPU time to tiny plan tensors copied from pinned CPU memory in `_kpool_plan_to_gpu`; a graphs-on formal run must confirm the production wall-clock impact before changing this path.

The KDA aggregate contains a repeatable profiler correlation error: 133 of 136 state-update launches complete in 0.31–0.38 ms, while layer-slot 19 in each of the three multi-request profile waves is reported as 197.7–204.5 ms. Later kernels on the same stream begin only 3–63 µs after those events start, so the reported 200 ms durations cannot be real kernel execution. Replacing them with the normal median gives 12.0 ms/forward for state update and 36.2 ms/forward for the complete KDA path.

The compiled production specialization is `K=128`, `V=128`, `BT=64`, `BV=32`, 4 warps, 2 stages, 84 VGPRs and 76,160 bytes shared memory. A production-grid rocprofv3 harness (`grid=(4,48,1)`, 256-thread workgroups) measures 0.33–0.35 ms kernel duration, 6.20% occupancy, 99.99% VALU utilization, 2.51% LDS bank conflicts, 332,680 KB fetched and 198,144 KB written. The low occupancy is real, but this kernel is only 1.7% of corrected prefill time; K-pool plan transfer, MoE, communication and sparse DSA are higher-value targets.

## PTPC projection candidates

The Flash checkpoint uses mixed projection precision rather than one uniform blockscale format:

| Path | Checkpoint/runtime precision | Current ROCm dispatch | Prefill cost | PTPC status |
| --- | --- | --- | ---: | --- |
| KDA `qkv_proj` | BF16, TP4 packed weight `[6144,4096]` | `UnquantizedLinearMethod` → AITER `tgemm.mm` | 16.2 ms | primary candidate |
| KDA `o_proj` | BF16, TP4 weight `[4096,2048]` | `UnquantizedLinearMethod` → AITER `tgemm.mm` | 33.8 ms | primary candidate |
| KDA `b/f/g` projections | BF16; replicated or column-sharded | `UnquantizedLinearMethod` → AITER `tgemm.mm` | 4.3 ms | secondary candidate |
| Dense MLP | FP8 `[128,128]` blockscale | AITER blockscale FP8 | 2.1 ms | already quantized |
| Routed MoE | FP8 `[128,128]` blockscale | AITER fused MoE | 97.2 ms | already quantized |
| DSA q/kv/o projections | FP8 `[128,128]` blockscale | AITER blockscale FP8 | included above | already quantized |

PR #33602 contains the reusable PTPC pieces: one-time BF16 weight repacking to per-channel FP8, `apply_fp8_ptpc_linear`, fused RMSNorm plus per-token quantization for q projections, and standalone per-token quantization for o projection. Its enablement gate is restricted to GLM-5.2 (`model_type == "glm_moe_dsa"`), so it never marks `Glm5NextLinearAttention` modules. Flash needs a separate KDA marker and forward path; the blockscale MLP, MoE and DSA modules must remain unchanged.

The exact module, checkpoint and TP4 runtime shapes are recorded in `results/glm53_flash_ptpc_projection_candidates.csv`.

## Artifacts

- Trace root: `/data2/hf_home/profiles/glm53_main_tp4_i8k_o16_c64_20260908/traces/1788897752.6961296`
- TP0 EXTEND: `1788897752.6985245-TP-0-EXTEND.trace.json.gz`
- TP0 DECODE: `1788897752.6985245-TP-0-DECODE.trace.json.gz`
- Functional attribution: `/data2/hf_home/profiles/glm53_main_tp4_i8k_o16_c64_20260908/glm53_tp0_extend.csv`
- Unified triage: `/data2/hf_home/profiles/glm53_main_tp4_i8k_o16_c64_20260908/triage.txt`
- KDA rocprof harness: `/data2/hf_home/profiles/glm53_main_tp4_i8k_o16_c64_20260908/kda_rocprof_harness.py`
- KDA rocprof counters: `/data2/hf_home/profiles/glm53_main_tp4_i8k_o16_c64_20260908/rocprof_smoke`
- PTPC candidate map: `results/glm53_flash_ptpc_projection_candidates.csv`
