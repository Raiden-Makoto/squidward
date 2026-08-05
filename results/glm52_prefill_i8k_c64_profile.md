# GLM-5.2 prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm5.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm51` `b79e5e4cc9`. MI355X image `raidenmakoto916/raidenmakoto:glm52-fork-aiter0724-ckaf7118-ckgemm1`; B200 image `lmsysorg/sglang:dev-cu13-glm52-nvfp4` with `sglang-kernel 0.4.5` and `flashinfer 0.6.15.post1`.

| Profile                                        | Total ms | Δ vs prior |
| ---------------------------------------------- | -------- | ---------- |
| Previous MI355X                                | 870.4624 | —          |
| FP8 projections + MXFP4 BMM                    | 727.9136 | −16.38%    |
| + zero-copy TileLang `q_cat`                   | 719.7370 | −1.12%     |
| + fused indexer query Hadamard / FP8 quant     | 718.3000 | −0.20%     |
| B200                                           | 659.0219 | −8.25%     |

| DSA indexer                | Before ms | Fused ms | Δ ms    | Δ       |
| -------------------------- | --------- | -------- | ------- | ------- |
| Query Hadamard + FP8 quant | 4.2250    | 1.7455   | −2.4795 | −58.69% |
| Full indexer               | 28.8151   | 26.6938  | −2.1213 | −7.36%  |

| Section             | MI355X kernel                                   | MI355X ms | B200 kernel                          | B200 ms   | B200/MI355X |
| ------------------- | ----------------------------------------------- | --------- | ------------------------------------ | --------- | ----------- |
| Attention           | TileLang `main_kernel` sparse MLA               | 303.0     | `fmhaSm100f` sparse FMHA             | 228.6     | 0.75x       |
| Attention           | `_fused_qk_rope_cat_and_cache_mla`              | 6.3       | `RopeQuantize` + `set_mla_kv_buffer` | 10.8      | 1.70x       |
| Attention           | q/k norm + rope                                 | 3.4       | `RMSNorm`                            | 3.0       | 0.88x       |
| Attention           | DSA indexer                                     | 26.7      | fused indexer + DeepGEMM logits      | 23.2      | 0.87x       |
| **Attention**       |                                                 | **339.4** |                                      | **265.7** | **0.78x**   |
| MoE gate-up (gemm1) | `mfma_moe1`                                     | 40.9      | `bmm_E2m1`                           | 70.2      | 1.72x       |
| MoE down (gemm2)    | `mfma_moe2`                                     | 51.3      | `bmm_Bfloat16_E2m1`                  | 33.4      | 0.65x       |
| MoE combine         | `moe_reduction_kernel`                          | 26.0      | `finalizeKernelVecLoad`              | 22.7      | 0.87x       |
| MoE act quant       | `dynamic_per_group_scaled_quant`                | 4.6       | `NVFP4Quantize`                      | 4.1       | 0.88x       |
| MoE routing / sort  | `mxfp4_moe_sort` + `p0/p1/p23` + grouped top-k  | 6.6       | routing + shared-expert MLP + add    | 34.4      | 5.21x       |
| **MoE**             |                                                 | **129.4** |                                      | **164.8** | **1.27x**   |
| Dense GEMM          | o_proj PTPC FP8                                 | 27.3      | `nvjet_sm100`                        | 42.4      | 1.55x       |
| Dense GEMM          | q_b_proj PTPC FP8                               | 11.5      | `nvjet_sm100`                        | 14.3      | 1.24x       |
| Dense GEMM          | q_a + kv_a                                      | 31.7      | `nvjet_sm100`                        | 35.9      | 1.13x       |
| Dense GEMM          | A16WFP4 K/V absorbed BMM                        | 40.5      | `nvjet_sm100`                        | 12.5      | 0.31x       |
| Dense GEMM          | PTPC o_proj input `flatten_fp8_per_token_quant` | 2.3       | fused / not standalone               | —         |             |
| Dense GEMM          | router GEMMs                                    | 5.6       | `nvjet_sm100`                        | 4.0       | 0.71x       |
| Dense GEMM          | DenseMLP L0–2                                   | 3.0       | `nvjet_sm100`                        | 3.8       | 1.26x       |
| **Dense GEMM**      |                                                 | **121.9** |                                      | **113.0** | **0.93x**   |
| All-reduce          | QuickReduce INT4                                | 111.2     | NCCL RING_LL                         | 94.0      | 0.85x       |
| RMSNorm             | `aiter::add_rmsnorm_quant`                      | 16.0      | `fused_add_rmsnorm`                  | 21.1      | 1.32x       |
| Other               | activation + output head + embedding + sampling | 0.4       | same                                 | 0.4       | 1.12x       |
| **TOTAL**           |                                                 | **718.3** |                                      | **659.0** | **0.92x**   |

## Lever ranking — excluding dense GEMM, all-reduce, and parity/faster rows

| Rank | Lever           | MI355X ms | B200 ms | Excess ms | % of MI total | Work class                              |
| ---- | --------------- | --------- | ------- | --------- | ------------- | --------------------------------------- |
| 1    | Sparse MLA core | 303.0     | 228.6   | 74.4      | 10.4%         | Attention kernel work in progress       |
| 2    | MoE gemm2       | 51.3      | 33.4    | 17.9      | 2.5%          | Optional; config exhausted, kernel work |
| 3    | DSA MQA logits  | 10.1      | 6.1     | 4.0       | 0.6%          | Kernel / hardware                       |
| 4    | MoE combine     | 26.0      | 22.7    | 3.3       | 0.5%          | HBM-bound; prior fusion lost            |
