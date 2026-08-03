# GLM-5.2 prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4). MI355X 870.5 measured graphs-off eager with `utilities/e2e_glm5.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`; `RM/glm51` `d4171d84e1`, image `raidenmakoto916/raidenmakoto:glm52-fork-aiter0724-ckaf7118-ckgemm1`. B200 values are supplied source-row projections; `≈` marks incomplete attribution.

| Section             | MI355X kernel                                      | MI355X ms | B200 kernel                                  | B200 ms | B200/MI355X |
| ------------------- | -------------------------------------------------- | --------- | -------------------------------------------- | ------- | ----------- |
| Attention           | TileLang `main_kernel` sparse MLA                  | 343.1     | `fmhaSm100f` sparse FMHA                     | 189.2   | 0.55x       |
| Attention           | `_fused_qk_rope_cat_and_cache_mla` + concat        | 26.3      | `RopeQuantize` + `set_mla_kv_buffer`         | 9.7     | 0.37x       |
| Attention           | q/k norm + rope                                    | 3.2       | `RMSNorm` + `fused_rope`                     | 2.7     | 0.84x       |
| Attention           | DSA indexer                                        | 33.5      | DSA indexer                                  | ≈3.4    | ≈0.10x      |
| **Attention**       |                                                    | **406.0** |                                              | **≈205.0** | **≈0.50x** |
| MoE gate-up (gemm1) | `mfma_moe1`                                        | 44.9      | `bmm_E2m1`                                   | 37.0    | 0.82x       |
| MoE down (gemm2)    | `mfma_moe2`                                        | 57.3      | `bmm_Bfloat16_E2m1`                          | 28.5    | 0.50x       |
| MoE combine         | `moe_reduction_kernel`                             | 29.6      | `finalizeKernelVecLoad`                      | 20.2    | 0.68x       |
| MoE act quant       | `dynamic_per_group_scaled_quant`                   | 5.4       | `NVFP4Quantize`                              | 3.2     | 0.60x       |
| MoE routing / sort  | `mxfp4_moe_sort` + `p0/p1/p23` + grouped top-k    | 7.3       | routing + shared/add overhead                | ≈19.2   | ≈2.64x      |
| **MoE**             |                                                    | **144.5** |                                              | **≈108.1** | **≈0.75x** |
| Dense GEMM          | o_proj                                             | 39.6      | —                                            | —       |             |
| Dense GEMM          | q_b absorbed BMM                                   | 73.3      | —                                            | —       |             |
| Dense GEMM          | q_a + kv_a                                         | 37.5      | —                                            | —       |             |
| Dense GEMM          | kv_b                                               | 15.1      | —                                            | —       |             |
| Dense GEMM          | router GEMMs                                       | 6.2       | —                                            | —       |             |
| Dense GEMM          | DenseMLP L0–2                                      | 3.5       | —                                            | —       |             |
| **Dense GEMM**      |                                                    | **175.2** | `nvjet_sm100` source rows                    | **≈107.0** | **≈0.61x** |
| All-reduce          | QuickReduce INT4                                   | 126.2     | named NCCL rows                              | 95.0    | 0.75x       |
| RMSNorm             | `aiter::add_rmsnorm_quant`                         | 18.1      | `fused_add_rmsnorm`                          | 18.9    | 1.04x       |
| Other               | activation + output head + embedding + sampling   | 0.4       | —                                            | —       |             |
| **TOTAL**           |                                                    | **870.5** |                                              | **—**   |             |
