# GLM-5.2-NVFP4 — B200 TP4 prefill profile

- Host: hungry-hippo-fin-03-8 (VM reservation, ephemeral), 8x NVIDIA B200 183GB
- Image: lmsysorg/sglang:dev-cu13-glm52-nvfp4 (sglang dev1+g430418e21, torch 2.11+cu130)
- Model: nvidia/GLM-5.2-NVFP4 (modelopt NVFP4; 64 heads, kv_lora 512, qk_rope 64, 78 layers)
- Launch: tp4, --quantization modelopt_fp4, --mem-fraction-static 0.80, --kv-cache-dtype fp8_e4m3, --chunked-prefill-size 16384, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- DSA backends: prefill=trtllm, decode=trtllm; MoE runner=flashinfer_trtllm
- Workload: random 8192/1024, conc4, 16 prompts, --profile-num-steps 4 --profile-by-stage
- Trace: b200_traces/1783469631.2453542-TP-0-EXTEND.trace.json.gz

## Serving

| metric | value |
|---|---|
| input token throughput (tok/s) | 2555.30 |
| output token throughput (tok/s) | 319.41 |
| median TTFT (ms) | 1128.16 |
| mean TTFT (ms) | 1550.64 |
| P99 TTFT (ms) | 6927.98 |
| median TPOT (ms) | 9.65 |
| mean TPOT (ms) | 11.02 |

## Prefill (EXTEND) top GPU kernels, TP0 (total kernel 1178.4 ms)

| kernel | ms | calls | pct | what it is |
|---|---|---|---|---|
| fmhaSm100fKernel_Qkv E4m3 HQk576 HV512 PagedKvDenseStatic | 410.3 | 234 | 34.8% | TRT-LLM fused fp8 sparse-MLA prefill attn |
| ncclDevKernel_AllReduce_Sum_bf16_RING_LL | 193.0 | 471 | 16.4% | TP all-reduce |
| nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT | 154.4 | 1014 | 13.1% | dense projection GEMMs (nvjet SM100) |
| bmm_E2m1_E2m1E2m1 t128x128x512 | 85.6 | 225 | 7.3% | MoE expert GEMM (NVFP4) |
| bmm_Bfloat16_E2m1E2m1 t128x128x256 | 70.4 | 225 | 6.0% | MoE expert GEMM (NVFP4) |
| moe::dev::finalize::finalizeKernelVecLoad | 41.6 | 225 | 3.5% | MoE combine/finalize |
| flashinfer fused_add_rmsnorm | 38.4 | 468 | 3.3% | RMSNorm+residual |
| nvjet_sm100_tst_176x128_64x8_1x2_2cta_h_bz_TNN | 24.2 | 78 | 2.1% | dense GEMM |
| topk_transform_prefill_kernel | 22.0 | 63 | 1.9% | DSA indexer topk |
| flashinfer RopeQuantizeKernel | 19.7 | 234 | 1.7% | rope + fp8 quant |
| deep_gemm::sm100_mqa_logits | 10.8 | 63 | 0.9% | DSA indexer logits |
| flashinfer nvfp4_quant | 5.3 | 150 | 0.4% | activation NVFP4 quant |

## Analogous kernels B200 vs MI355X (both TP4, 8192/1024 conc4)

MI355X column = perf config (triton #28975 + INT4 quick-reduce + fused indexer on).
B200 total kernel 1178 ms; MI355X total kernel 1583 ms.

| role | B200 (ms) | MI355X (ms) | B200 kernels | MI355X kernels |
|---|---|---|---|---|
| sparse-MLA prefill attn | 410 | 494 | fmhaSm100f | _sparse_mla_fwd_kernel (triton #28975) |
| MoE expert GEMMs (gate/up + down + sort/finalize) | ~197 | ~351 | bmm_E2m1 85.6 + bmm_bf16_E2m1 70.4 + finalize 41.6 | mfma_moe2 180.6+64.9 + moe1_silu 66.8+30.0 + sort 9.0 |
| dense projection GEMMs | ~179 | ~237 | nvjet 154.4 + 24.2 | Cijk 144.8 + 55.8 + 18.4 + 18.3 |
| TP all-reduce | 193 | 273 | nccl RING_LL | quickreduce INT4 twoshot |
| RMSNorm(+quant) | 38 | 43 | flashinfer fused_add_rmsnorm | aiter add_rmsnorm_quant |
| DSA indexer topk | 22 | 31 | topk_transform_prefill | topk_transform_prefill |
| DSA indexer logits | 11 | 29 | deep_gemm sm100_mqa_logits | _gluon_fp8_mqa_logits |
| q/k rope + quant + cache prepare | ~25 | ~48 | RopeQuantize 19.7 + nvfp4_quant 5.3 | idxqk fused 16.0 (#30422) + MLA rope-cache 15.6 + act_quant 16.1 |

## Optimization status

| target | MI355X vs B200 | status |
|---|---|---|
| DSA indexer logits (`_gluon_fp8_mqa_logits`) | 29 vs 11 ms | Closed — hardware gap. Both materialize full [q x k] logits; deep_gemm wins via Blackwell tcgen05 UMMA + TMEM accumulators + TMA vs gfx950 MFMA + register accum. gfx950 gluon already beats non-gluon (0.349 vs 0.836 ms) and is near the CDNA4 ceiling; no software lever. |
| Fused indexer q/k (#30422) | fused 16.0 ms | Weak at conc4. Same-config A/B (triton+QR, conc4): total prefill kernel +21.7 ms and TTFT +22 ms with it ON; fused kernel (16 ms) is heavier than the fragmented pieces it replaces (~4-5 ms). Only upside is -0.5 ms TPOT. Original PR claim is a scheduling gain at conc>=16; revisit there. |
| MoE expert GEMMs (mfma afp4/wfp4) | ~351 vs ~197 ms | Closed for tuning. The two t64/t128 tile variants are the two M-buckets (M=24501 / M=8192), ~equal per-token (98 vs 106 ns/tok) -> Tensile already at CDNA4 ceiling, no tile-selection win. Only lever is a better kernel (CK/stream-K, vendor/upstream) -- not a config tune. |
| Dense projection GEMMs (`gemm_a16w16` bf16) | ~237 vs ~179 ms | Ceiling-limited. Workhorse MT256x256x64 ~37% of bf16 roofline; Tensile per-(shape,M) solutions already near ceiling (B200 generalizes via TMA-predication + TMEM + 2-CTA, CDNA4 can't). Only lever = fp8/mxfp4 dense quant, which is adjacent to Jacob's fp8-MLA theme (#30519) -> coordinate/defer. |
| TP all-reduce (quick-reduce INT4) | 273 vs 193 ms | Jacob + hardware. Closable overhead is aiter allreduce fusion (Jacob); residual is NVLink5/NVSwitch vs Infinity Fabric bandwidth floor. Not our lane. |
| MLA absorbed bmm -> fp8 | Jacob | Jacob #30519 (w_kc/w_vc fp8 + batched_gemm_a8w8). Not our lane. |
| **Sparse-MLA prefill attn (`_sparse_mla_fwd`, #28975)** | **494 vs 410 ms** | **OPEN, ours, has room.** seq=8192 kernel = 1.732 ms/call, effective DRAM ~4 TB/s (5.58 TB/s logical minus ~28% L2 assist). Achievable HBM here is only ~3.5-4.5 TB/s vs ~8 spec, and compute floor is 0.225 ms (7.7x below kernel) -> almost pure memory with large compute slack that plain Triton does NOT hide on CDNA4. Levers: gluon async double-buffered `buffer_load_to_shared` gather prefetch (overlap gather latency behind MFMA), single coalesced 576-wide page load (vs current 512+64), swizzled-LDS staging to lift effective BW toward ~6 TB/s. Est: recover a real chunk of the 84 ms (plausibly ~=/< B200's 410); magnitude TBD by prototype. |
