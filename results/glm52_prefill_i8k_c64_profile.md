# GLM-5.2 prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm5.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm51` `6e2eb1c598`. MI355X image `raidenmakoto916/raidenmakoto:glm52-fork-aiter0724-ckaf7118-ckgemm1`; B200 image `lmsysorg/sglang:dev-cu13-glm52-nvfp4` with `sglang-kernel 0.4.5` and `flashinfer 0.6.15.post1`.

| Profile                          | Total ms | Δ vs previous MI |
| -------------------------------- | -------: | ---------------: |
| Previous MI355X                  | 870.4624 |                — |
| MI355X FP8 projections + MXFP4 BMM | 727.9136 |           −16.38% |
| B200                             | 659.0219 |           −24.29% |

| Section             | MI355X kernel                                      | MI355X ms | B200 kernel                          | B200 ms | B200/MI355X |
| ------------------- | -------------------------------------------------- | --------: | ------------------------------------ | -------: | -----------: |
| Attention           | TileLang `main_kernel` sparse MLA                  | 296.8     | `fmhaSm100f` sparse FMHA             | 228.6    | 0.77x        |
| Attention           | `_fused_qk_rope_cat_and_cache_mla` + concat        | 23.4      | `RopeQuantize` + `set_mla_kv_buffer` | 10.8     | 0.46x        |
| Attention           | q/k norm + rope                                    | 5.6       | `RMSNorm`                            | 3.0      | 0.53x        |
| Attention           | DSA indexer                                        | 28.6      | fused indexer + DeepGEMM logits      | 23.2     | 0.81x        |
| **Attention**       |                                                    | **354.4** |                                      | **265.7** | **0.75x**    |
| MoE gate-up (gemm1) | `mfma_moe1`                                        | 40.4      | `bmm_E2m1`                           | 70.2     | 1.74x        |
| MoE down (gemm2)    | `mfma_moe2`                                        | 51.2      | `bmm_Bfloat16_E2m1`                  | 33.4     | 0.65x        |
| MoE combine         | `moe_reduction_kernel`                             | 26.0      | `finalizeKernelVecLoad`              | 22.7     | 0.87x        |
| MoE act quant       | `dynamic_per_group_scaled_quant`                   | 4.6       | `NVFP4Quantize`                      | 4.1      | 0.88x        |
| MoE routing / sort  | `mxfp4_moe_sort` + `p0/p1/p23` + grouped top-k     | 6.6       | routing + shared-expert MLP + add    | 34.4     | 5.24x        |
| **MoE**             |                                                    | **128.8** |                                      | **164.8** | **1.28x**    |
| Dense GEMM          | o_proj PTPC FP8                                    | 26.8      | `nvjet_sm100`                        | 42.4     | 1.58x        |
| Dense GEMM          | q_b_proj PTPC FP8                                  | 11.4      | `nvjet_sm100`                        | 14.3     | 1.25x        |
| Dense GEMM          | q_a + kv_a                                         | 31.3      | `nvjet_sm100`                        | 35.9     | 1.15x        |
| Dense GEMM          | A16WFP4 K/V absorbed BMM                           | 40.1      | `nvjet_sm100`                        | 12.5     | 0.31x        |
| Dense GEMM          | router GEMMs                                       | 5.5       | `nvjet_sm100`                        | 4.0      | 0.73x        |
| Dense GEMM          | DenseMLP L0–2                                      | 3.0       | `nvjet_sm100`                        | 3.8      | 1.27x        |
| **Dense GEMM**      |                                                    | **118.2** |                                      | **113.0** | **0.96x**    |
| All-reduce          | QuickReduce INT4                                   | 110.2     | NCCL RING_LL                         | 94.0     | 0.85x        |
| RMSNorm             | `aiter::add_rmsnorm_quant`                         | 16.0      | `fused_add_rmsnorm`                  | 21.1     | 1.32x        |
| Other               | activation + output head + embedding + sampling    | 0.4       | same                                 | 0.4      | 1.12x        |
| **TOTAL**           |                                                    | **727.9** |                                      | **659.0** | **0.91x**    |

## Lever ranking — excluding dense GEMM, all-reduce, and parity/faster rows

| Rank | Lever              | MI355X ms | B200 ms | Excess ms | % of MI total | Work class                              |
| ---: | ------------------ | --------: | ------: | --------: | ------------: | --------------------------------------- |
| 1    | Sparse MLA core    | 296.8     | 228.6   | 68.2      | 9.4%          | Attention kernel work in progress       |
| 2    | MoE gemm2          | 51.2      | 33.4    | 17.8      | 2.4%          | Optional; config exhausted, kernel work |
| 3    | MLA cache / concat | 23.4      | 10.8    | 12.6      | 1.7%          | Fusion / memory-path work               |
| 4    | DSA indexer        | 28.6      | 23.2    | 5.4       | 0.7%          | Low ceiling                             |
| 5    | MoE combine        | 26.0      | 22.7    | 3.3       | 0.5%          | HBM-bound; prior fusion lost            |
