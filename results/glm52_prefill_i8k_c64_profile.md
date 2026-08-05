# GLM-5.2 prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm5.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm51` `b50a95b128`. MI355X image `raidenmakoto916/raidenmakoto:glm52-fork-aiter0724-ckaf7118-ckgemm1`; B200 image `lmsysorg/sglang:dev-cu13-glm52-nvfp4` with `sglang-kernel 0.4.5` and `flashinfer 0.6.15.post1`.


| Profile                                      | Total ms | Δ vs combined MI |
| -------------------------------------------- | -------- | ---------------- |
| Previous MI355X                              | 870.4624 | +19.58%          |
| Combined MI355X FP8 projections + MXFP4 BMM  | 727.9136 | —                |
| Combined MI355X + zero-copy TileLang `q_cat` | 719.7370 | −1.12%           |
| B200                                         | 659.0219 | −9.46%           |

| Section    | Combined MI ms | Zero-copy MI ms | Δ ms     |
| ---------- | -------------- | --------------- | -------- |
| Attention  | 352.1399       | 336.6381        | −15.5019 |
| MoE        | 128.8465       | 128.8110        | −0.0355  |
| Dense GEMM | 120.5302       | 123.8974        | +3.3672  |
| All-reduce | 110.1660       | 114.1837        | +4.0177  |
| RMSNorm    | 16.0047        | 15.9793         | −0.0254  |
| Other      | 0.2263         | 0.2276          | +0.0013  |
| **TOTAL**  | **727.9136**   | **719.7370**    | **−8.1766** |

| Section             | MI355X kernel                                   | MI355X ms | B200 kernel                          | B200 ms   | B200/MI355X |
| ------------------- | ----------------------------------------------- | --------- | ------------------------------------ | --------- | ----------- |
| Attention           | TileLang `main_kernel` sparse MLA               | 298.0     | `fmhaSm100f` sparse FMHA             | 228.6     | 0.77x       |
| Attention           | `_fused_qk_rope_cat_and_cache_mla`              | 6.4       | `RopeQuantize` + `set_mla_kv_buffer` | 10.8      | 1.69x       |
| Attention           | q/k norm + rope                                 | 3.4       | `RMSNorm`                            | 3.0       | 0.88x       |
| Attention           | DSA indexer                                     | 28.8      | fused indexer + DeepGEMM logits      | 23.2      | 0.81x       |
| **Attention**       |                                                 | **336.6** |                                      | **265.7** | **0.79x**   |
| MoE gate-up (gemm1) | `mfma_moe1`                                     | 40.7      | `bmm_E2m1`                           | 70.2      | 1.73x       |
| MoE down (gemm2)    | `mfma_moe2`                                     | 51.1      | `bmm_Bfloat16_E2m1`                  | 33.4      | 0.65x       |
| MoE combine         | `moe_reduction_kernel`                          | 25.9      | `finalizeKernelVecLoad`              | 22.7      | 0.88x       |
| MoE act quant       | `dynamic_per_group_scaled_quant`                | 4.6       | `NVFP4Quantize`                      | 4.1       | 0.88x       |
| MoE routing / sort  | `mxfp4_moe_sort` + `p0/p1/p23` + grouped top-k  | 6.5       | routing + shared-expert MLP + add    | 34.4      | 5.29x       |
| **MoE**             |                                                 | **128.8** |                                      | **164.8** | **1.28x**   |
| Dense GEMM          | o_proj PTPC FP8                                 | 28.3      | `nvjet_sm100`                        | 42.4      | 1.50x       |
| Dense GEMM          | q_b_proj PTPC FP8                               | 11.8      | `nvjet_sm100`                        | 14.3      | 1.21x       |
| Dense GEMM          | q_a + kv_a                                      | 32.4      | `nvjet_sm100`                        | 35.9      | 1.11x       |
| Dense GEMM          | A16WFP4 K/V absorbed BMM                        | 40.2      | `nvjet_sm100`                        | 12.5      | 0.31x       |
| Dense GEMM          | PTPC o_proj input `flatten_fp8_per_token_quant` | 2.3       | fused / not standalone               | —         |             |
| Dense GEMM          | router GEMMs                                    | 5.7       | `nvjet_sm100`                        | 4.0       | 0.70x       |
| Dense GEMM          | DenseMLP L0–2                                   | 3.2       | `nvjet_sm100`                        | 3.8       | 1.19x       |
| **Dense GEMM**      |                                                 | **123.9** |                                      | **113.0** | **0.91x**   |
| All-reduce          | QuickReduce INT4                                | 114.2     | NCCL RING_LL                         | 94.0      | 0.82x       |
| RMSNorm             | `aiter::add_rmsnorm_quant`                      | 16.0      | `fused_add_rmsnorm`                  | 21.1      | 1.32x       |
| Other               | activation + output head + embedding + sampling | 0.2       | same                                 | 0.4       | 1.76x       |
| **TOTAL**           |                                                 | **719.7** |                                      | **659.0** | **0.92x**   |

## Lever ranking — excluding dense GEMM, all-reduce, and parity/faster rows

| Rank | Lever           | MI355X ms | B200 ms | Excess ms | % of MI total | Work class                              |
| ---- | --------------- | --------- | ------- | --------- | ------------- | --------------------------------------- |
| 1    | Sparse MLA core | 298.0     | 228.6   | 69.4      | 9.6%          | Attention kernel work in progress       |
| 2    | MoE gemm2       | 51.1      | 33.4    | 17.7      | 2.5%          | Optional; config exhausted, kernel work |
| 3    | DSA indexer     | 28.8      | 23.2    | 5.6       | 0.8%          | Low ceiling                             |
| 4    | MoE combine     | 25.9      | 22.7    | 3.2       | 0.4%          | HBM-bound; prior fusion lost            |
