# GLM-5.2 prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4) vs GB300 (NVFP4)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm5.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm51` `48e0ba7a80`; AITER `RM/glm52-mqa-blockq` `59904c2b94`. MI355X image `raidenmakoto916/raidenmakoto:glm52-fork-aiter0724-ckaf7118-ckgemm1`; GB300 ARM64 image `lmsysorg/sglang:dev-cu13-glm52-nvfp4` with `sglang-kernel 0.4.5` and `flashinfer 0.6.15.post1`.

## Attention

| Component              | MI355X kernel                              | MI355X ms | GB300 kernel                         | GB300 ms  | GB300/MI355X |
| ---------------------- | ------------------------------------------ | --------- | ------------------------------------ | --------- | ------------ |
| Sparse MLA             | Triton sparse MLA                          | 166.0     | `fmhaSm100f` sparse FMHA             | 204.0     | 1.23x        |
| MLA cache              | `_fused_qk_rope_cat_and_cache_mla`         | 7.4       | `RopeQuantize` + `set_mla_kv_buffer` | 9.8       | 1.33x        |
| q/k norm + rope        | q/k norm + rope                            | 3.8       | `RMSNorm`                            | 2.6       | 0.69x        |
| DSA top-k              | top-k transform                            | 14.4      | `topk_transform_prefill`             | 11.0      | 0.76x        |
| DSA MQA logits         | BLOCK_Q FP8 ragged MQA logits              | 7.6       | `deep_gemm::sm100_mqa_logits`        | 5.2       | 0.69x        |
| DSA prep/store + misc  | norm/rope/quant/cache + misc               | 4.9       | fused indexer prep/store + misc      | 4.8       | 0.97x        |
| **Attention subtotal** |                                            | **204.1** |                                      | **237.5** | **1.16x**    |

## MoE

| Component        | MI355X kernel                                      | MI355X ms | GB300 kernel                      | GB300 ms  | GB300/MI355X |
| ---------------- | -------------------------------------------------- | --------- | --------------------------------- | --------- | ------------ |
| Gate-up (gemm1)  | `mfma_moe1`                                        | 45.1      | `bmm_E2m1`                        | 64.2      | 1.42x        |
| Down (gemm2)     | `mfma_moe2`                                        | 56.9      | `bmm_Bfloat16_E2m1`               | 30.9      | 0.54x        |
| Combine          | `moe_reduction_kernel`                             | 29.8      | `finalizeKernelVecLoad`           | 22.3      | 0.75x        |
| Activation quant | `dynamic_per_group_scaled_quant`                   | 5.4       | `NVFP4Quantize`                   | 3.9       | 0.72x        |
| Routing / sort   | `mxfp4_moe_sort` + `p0/p1/p23` + grouped top-k    | 7.2       | routing + shared-expert MLP + add | 31.3      | 4.34x        |
| **MoE subtotal** |                                                    | **144.4** |                                   | **152.7** | **1.06x**    |

## Dense GEMM

| Component          | MI355X kernel                                   | MI355X ms | GB300 kernel           | GB300 ms  | GB300/MI355X |
| ------------------ | ----------------------------------------------- | --------- | ---------------------- | --------- | ------------ |
| o_proj             | PTPC FP8                                        | 31.5      | `nvjet_sm103`          | 35.5      | 1.13x        |
| q_b_proj           | PTPC FP8                                        | 13.2      | `nvjet_sm103`          | 12.1      | 0.91x        |
| q_a + kv_a         | q_a + kv_a                                      | 37.1      | `nvjet_sm103`          | 30.6      | 0.82x        |
| Absorbed K/V BMM   | A16WFP4                                         | 45.9      | `nvjet_sm103`          | 11.0      | 0.24x        |
| o_proj input quant | `flatten_fp8_per_token_quant`                    | 2.6       | fused / not standalone | —         |              |
| Router GEMMs       | router GEMMs                                    | 6.5       | `nvjet_sm103`          | 3.8       | 0.59x        |
| DenseMLP L0–2      | DenseMLP L0–2                                   | 3.5       | `nvjet_sm103`          | 3.1       | 0.89x        |
| **Dense subtotal** |                                                 | **140.2** |                        | **96.0**  | **0.68x**    |

## Communication and normalization

| Component           | MI355X kernel                                   | MI355X ms | GB300 kernel | GB300 ms  | GB300/MI355X |
| ------------------- | ----------------------------------------------- | --------- | ------------ | --------- | ------------ |
| All-reduce          | QuickReduce INT4                                | 237.6     | NCCL RING_LL | 126.9     | 0.53x        |
| RMSNorm             | `aiter::add_rmsnorm_quant`                      | 18.2      | `fused_add_rmsnorm` | 20.3 | 1.11x        |
| Other               | activation + output head + embedding + sampling | 0.4       | same         | 0.5       | 1.31x        |
| **TOTAL**           |                                                 | **745.0** |              | **633.8** | **0.85x**    |

## Lever ranking — excluding all-reduce and parity/faster rows

| Rank | Lever                | MI355X ms | GB300 ms | Excess ms | % of MI total | Work class                              |
| ---- | -------------------- | --------- | -------- | --------- | ------------- | --------------------------------------- |
| 1    | A16WFP4 absorbed BMM | 45.9      | 11.0     | 34.9      | 4.7%          | Kernel work                             |
| 2    | MoE gemm2            | 56.9      | 30.9     | 26.0      | 3.5%          | Optional; config exhausted, kernel work |
| 3    | MoE combine          | 29.8      | 22.3     | 7.5       | 1.0%          | HBM-bound; prior fusion lost            |
| 4    | q_a + kv_a           | 37.1      | 30.6     | 6.5       | 0.9%          | Vendor GEMM / kernel work               |
| 5    | DSA top-k            | 14.4      | 11.0     | 3.4       | 0.5%          | Near parity                             |
| 6    | Router GEMMs         | 6.5       | 3.8      | 2.7       | 0.4%          | Kernel work                             |
| 7    | DSA MQA logits       | 7.6       | 5.2      | 2.4       | 0.3%          | BLOCK_Q active; residual kernel gap     |
| 8    | MoE act quant        | 5.4       | 3.9      | 1.5       | 0.2%          | HBM-bound; prior fusion lost            |
| 9    | q/k norm + rope      | 3.8       | 2.6      | 1.2       | 0.2%          | Near parity                             |
| 10   | q_b_proj PTPC FP8    | 13.2      | 12.1     | 1.1       | 0.2%          | Near parity                             |
| 11   | DenseMLP L0–2        | 3.5       | 3.1      | 0.4       | 0.1%          | Near parity                             |
| 12   | DSA prep / store     | 4.9       | 4.8      | 0.1       | 0.0%          | Near parity                             |
