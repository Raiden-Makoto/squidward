# GLM-5.2 prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4) vs GB300 (NVFP4)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm5.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm51` `355234e16b`. MI355X image `raidenmakoto916/raidenmakoto:glm52-fork-aiter0724-ckaf7118-ckgemm1`; GB300 ARM64 image `lmsysorg/sglang:dev-cu13-glm52-nvfp4` with `sglang-kernel 0.4.5` and `flashinfer 0.6.15.post1`.

| Section             | MI355X kernel                                   | MI355X ms | GB300 kernel                         | GB300 ms  | GB300/MI355X |
| ------------------- | ----------------------------------------------- | --------- | ------------------------------------ | --------- | ------------ |
| Attention           | Triton sparse MLA                               | 188.3     | `fmhaSm100f` sparse FMHA             | 204.0     | 1.08x        |
| Attention           | `_fused_qk_rope_cat_and_cache_mla`              | 8.3       | `RopeQuantize` + `set_mla_kv_buffer` | 9.8       | 1.19x        |
| Attention           | q/k norm + rope                                 | 4.2       | `RMSNorm`                            | 2.6       | 0.61x        |
| DSA indexer         | top-k transform                                 | 15.6      | `topk_transform_prefill`             | 11.0      | 0.71x        |
| DSA indexer         | FP8 ragged MQA logits                           | 13.1      | `deep_gemm::sm100_mqa_logits`        | 5.2       | 0.40x        |
| DSA indexer         | norm/rope/quant/cache + misc                    | 5.4       | fused indexer prep/store + misc      | 4.8       | 0.88x        |
| **Attention**       |                                                 | **235.0** |                                      | **237.5** | **1.01x**    |
| MoE gate-up (gemm1) | `mfma_moe1`                                     | 50.2      | `bmm_E2m1`                           | 64.2      | 1.28x        |
| MoE down (gemm2)    | `mfma_moe2`                                     | 64.7      | `bmm_Bfloat16_E2m1`                  | 30.9      | 0.48x        |
| MoE combine         | `moe_reduction_kernel`                          | 33.5      | `finalizeKernelVecLoad`              | 22.3      | 0.67x        |
| MoE act quant       | `dynamic_per_group_scaled_quant`                | 5.9       | `NVFP4Quantize`                      | 3.9       | 0.67x        |
| MoE routing / sort  | `mxfp4_moe_sort` + `p0/p1/p23` + grouped top-k  | 8.1       | routing + shared-expert MLP + add    | 31.3      | 3.87x        |
| **MoE**             |                                                 | **162.5** |                                      | **152.7** | **0.94x**    |
| Dense GEMM          | o_proj PTPC FP8                                 | 35.0      | `nvjet_sm103`                        | 35.5      | 1.01x        |
| Dense GEMM          | q_b_proj PTPC FP8                               | 14.6      | `nvjet_sm103`                        | 12.1      | 0.83x        |
| Dense GEMM          | q_a + kv_a                                      | 41.4      | `nvjet_sm103`                        | 30.6      | 0.74x        |
| Dense GEMM          | A16WFP4 K/V absorbed BMM                        | 51.8      | `nvjet_sm103`                        | 11.0      | 0.21x        |
| Dense GEMM          | PTPC o_proj input `flatten_fp8_per_token_quant` | 2.9       | fused / not standalone               | —         |              |
| Dense GEMM          | router GEMMs                                    | 7.1       | `nvjet_sm103`                        | 3.8       | 0.54x        |
| Dense GEMM          | DenseMLP L0–2                                   | 3.9       | `nvjet_sm103`                        | 3.1       | 0.80x        |
| **Dense GEMM**      |                                                 | **156.7** |                                      | **96.0**  | **0.61x**    |
| All-reduce          | QuickReduce INT4                                | 136.2     | NCCL RING_LL                         | 126.9     | 0.93x        |
| RMSNorm             | `aiter::add_rmsnorm_quant`                      | 20.4      | `fused_add_rmsnorm`                  | 20.3      | 0.99x        |
| Other               | activation + output head + embedding + sampling | 0.4       | same                                 | 0.5       | 1.14x        |
| **TOTAL**           |                                                 | **711.2** |                                      | **633.8** | **0.89x**    |

## Lever ranking — excluding all-reduce and parity/faster rows

| Rank | Lever                | MI355X ms | GB300 ms | Excess ms | % of MI total | Work class                              |
| ---- | -------------------- | --------- | -------- | --------- | ------------- | --------------------------------------- |
| 1    | A16WFP4 absorbed BMM | 51.8      | 11.0     | 40.8      | 5.7%          | Kernel work                             |
| 2    | MoE gemm2            | 64.7      | 30.9     | 33.8      | 4.8%          | Optional; config exhausted, kernel work |
| 3    | MoE combine          | 33.5      | 22.3     | 11.2      | 1.6%          | HBM-bound; prior fusion lost            |
| 4    | q_a + kv_a           | 41.4      | 30.6     | 10.8      | 1.5%          | Vendor GEMM / kernel work               |
| 5    | DSA MQA logits       | 13.1      | 5.2      | 7.9       | 1.1%          | AITER #4180: −25.21% kernel             |
| 6    | DSA top-k            | 15.6      | 11.0     | 4.6       | 0.6%          | Near parity                             |
| 7    | Router GEMMs         | 7.1       | 3.8      | 3.3       | 0.5%          | Kernel work                             |
| 8    | q_b_proj PTPC FP8    | 14.6      | 12.1     | 2.5       | 0.4%          | Near parity                             |
| 9    | MoE act quant        | 5.9       | 3.9      | 2.0       | 0.3%          | HBM-bound; prior fusion lost            |
| 10   | q/k norm + rope      | 4.2       | 2.6      | 1.6       | 0.2%          | Near parity                             |
| 11   | DenseMLP L0–2        | 3.9       | 3.1      | 0.8       | 0.1%          | Near parity                             |
| 12   | DSA prep / store     | 5.4       | 4.8      | 0.6       | 0.1%          | Near parity                             |
