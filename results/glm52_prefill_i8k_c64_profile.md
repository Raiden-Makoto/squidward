# GLM-5.2 prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm5.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm51` `9e0563b9c3`. MI355X image `raidenmakoto916/raidenmakoto:glm52-fork-aiter0724-ckaf7118-ckgemm1`; B200 image `lmsysorg/sglang:dev-cu13-glm52-nvfp4` with `sglang-kernel 0.4.5` and `flashinfer 0.6.15.post1`.

| Section             | MI355X kernel                                   | MI355X ms | B200 kernel                          | B200 ms   | B200/MI355X |
| ------------------- | ----------------------------------------------- | --------- | ------------------------------------ | --------- | ----------- |
| Attention           | TileLang `main_kernel` sparse MLA               | 343.1     | `fmhaSm100f` sparse FMHA             | 228.6     | 0.67x       |
| Attention           | `_fused_qk_rope_cat_and_cache_mla` + concat     | 26.3      | `RopeQuantize` + `set_mla_kv_buffer` | 10.8      | 0.41x       |
| Attention           | q/k norm + rope                                 | 3.2       | `RMSNorm`                            | 3.0       | 0.95x       |
| Attention           | DSA indexer                                     | 33.5      | fused indexer + DeepGEMM logits      | 23.2      | 0.69x       |
| **Attention**       |                                                 | **406.0** |                                      | **265.7** | **0.65x**   |
| MoE gate-up (gemm1) | `mfma_moe1`                                     | 44.9      | `bmm_E2m1`                           | 70.2      | 1.56x       |
| MoE down (gemm2)    | `mfma_moe2`                                     | 57.3      | `bmm_Bfloat16_E2m1`                  | 33.4      | 0.58x       |
| MoE combine         | `moe_reduction_kernel`                          | 29.6      | `finalizeKernelVecLoad`              | 22.7      | 0.77x       |
| MoE act quant       | `dynamic_per_group_scaled_quant`                | 5.4       | `NVFP4Quantize`                      | 4.1       | 0.76x       |
| MoE routing / sort  | `mxfp4_moe_sort` + `p0/p1/p23` + grouped top-k  | 7.3       | routing + shared-expert MLP + add    | 34.4      | 4.71x       |
| **MoE**             |                                                 | **144.5** |                                      | **164.8** | **1.14x**   |
| Dense GEMM          | o_proj                                          | 39.6      | `nvjet_sm100`                        | 42.4      | 1.07x       |
| Dense GEMM          | q_b_proj                                        | 15.1      | `nvjet_sm100`                        | 14.3      | 0.95x       |
| Dense GEMM          | q_a + kv_a                                      | 37.5      | `nvjet_sm100`                        | 35.9      | 0.96x       |
| Dense GEMM          | kv_b absorbed BMM                               | 73.3      | `nvjet_sm100`                        | 12.5      | 0.17x       |
| Dense GEMM          | router GEMMs                                    | 6.2       | `nvjet_sm100`                        | 4.0       | 0.64x       |
| Dense GEMM          | DenseMLP L0–2                                   | 3.5       | `nvjet_sm100`                        | 3.8       | 1.09x       |
| **Dense GEMM**      |                                                 | **175.2** |                                      | **113.0** | **0.64x**   |
| All-reduce          | QuickReduce INT4                                | 126.2     | NCCL RING_LL                         | 94.0      | 0.74x       |
| RMSNorm             | `aiter::add_rmsnorm_quant`                      | 18.1      | `fused_add_rmsnorm`                  | 21.1      | 1.16x       |
| Other               | activation + output head + embedding + sampling | 0.4       | same                                 | 0.4       | 1.12x       |
| **TOTAL**           |                                                 | **870.5** |                                      | **659.0** | **0.76x**   |

## Lever ranking — excluding dense GEMM, all-reduce, and parity/faster rows

| Rank | Lever                | MI355X ms | B200 ms | Excess ms | % of MI total | Constraint                       |
| ---- | -------------------- | --------- | ------- | --------- | ------------- | -------------------------------- |
| 1    | Sparse MLA core      | 343.1     | 228.6   | 114.5     | 13.2%         | —                                |
| 2    | MoE gemm2            | 57.3      | 33.4    | 23.9      | 2.7%          | Existing config levers exhausted |
| 3    | MLA cache / concat   | 26.3      | 10.8    | 15.5      | 1.8%          | —                                |
| 4    | DSA indexer          | 33.5      | 23.2    | 10.3      | 1.2%          | —                                |
| 5    | MoE combine          | 29.6      | 22.7    | 6.9       | 0.8%          | HBM-bound; prior fusion lost     |
| 6    | MoE activation quant | 5.4       | 4.1     | 1.3       | 0.1%          | HBM-bound; prior fusion lost     |
