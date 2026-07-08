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

| role | B200 (ms) | MI355X (ms) |
|---|---|---|
| sparse-MLA prefill attn | 410 (fmhaSm100f) | 492 (_sparse_mla_fwd_kernel, triton #28975) |
| TP all-reduce | 193 (nccl RING_LL) | 250 (quickreduce INT4) |
| dense proj GEMMs | 154 (nvjet) | 143 (Cijk Tensile) |
| DSA indexer topk | 22 | 31 |
| DSA indexer logits | 11 (deep_gemm) | 29 (_gluon_fp8_mqa_logits) |
| RMSNorm(+quant) | 38 | 43 |

## Optimization status

| target | MI355X vs B200 | status |
|---|---|---|
| DSA indexer logits (`_gluon_fp8_mqa_logits`) | 29 vs 11 ms | Closed — hardware gap. Both materialize full [q x k] logits; deep_gemm wins via Blackwell tcgen05 UMMA + TMEM accumulators + TMA vs gfx950 MFMA + register accum. gfx950 gluon already beats non-gluon (0.349 vs 0.836 ms) and is near the CDNA4 ceiling; no software lever. |
