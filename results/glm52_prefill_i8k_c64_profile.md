# GLM-5.2 prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4) vs GB300 (NVFP4)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm5.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm51` `1d023bcaba`; stock AITER `6c48c5fa0e`, `SGLANG_DSA_FP8_PROJ_GEMM=1`. MI355X image `raidenmakoto916/raidenmakoto:glm52-fork-aiter0724-ckaf7118-ckgemm1`; GB300 ARM64 image `lmsysorg/sglang:dev-cu13-glm52-nvfp4` with `sglang-kernel 0.4.5` and `flashinfer 0.6.15.post1`.

## Attention

| Component              | MI355X kernel                              | MI355X ms | GB300 kernel                         | GB300 ms  | GB300/MI355X |
| ---------------------- | ------------------------------------------ | --------- | ------------------------------------ | --------- | ------------ |
| Sparse MLA             | Triton sparse MLA                          | 183.7     | `fmhaSm100f` sparse FMHA             | 204.0     | 1.11x        |
| MLA cache              | `_fused_qk_rope_cat_and_cache_mla`         | 8.3       | `RopeQuantize` + `set_mla_kv_buffer` | 9.8       | 1.18x        |
| q/k norm + rope        | q/k norm + rope                            | 0.9       | `RMSNorm`                            | 2.6       | 2.89x        |
| DSA top-k              | top-k transform                            | 15.8      | `topk_transform_prefill`             | 11.0      | 0.70x        |
| DSA MQA logits         | Gluon FP8 ragged MQA logits                | 13.1      | `deep_gemm::sm100_mqa_logits`        | 5.2       | 0.40x        |
| DSA prep/store + misc  | norm/rope/quant/cache + misc               | 5.4       | fused indexer prep/store + misc      | 4.8       | 0.89x        |
| **Attention subtotal** |                                            | **227.3** |                                      | **237.5** | **1.04x**    |

## MoE

| Component        | MI355X kernel                                      | MI355X ms | GB300 kernel                      | GB300 ms  | GB300/MI355X |
| ---------------- | -------------------------------------------------- | --------- | --------------------------------- | --------- | ------------ |
| Gate-up (gemm1)  | `mfma_moe1`                                        | 48.8      | `bmm_E2m1`                        | 64.2      | 1.32x        |
| Down (gemm2)     | `mfma_moe2`                                        | 62.2      | `bmm_Bfloat16_E2m1`               | 30.9      | 0.50x        |
| Combine          | `moe_reduction_kernel`                             | 33.3      | `finalizeKernelVecLoad`           | 22.3      | 0.67x        |
| Activation quant | `dynamic_per_group_scaled_quant`                   | 6.1       | `NVFP4Quantize`                   | 3.9       | 0.64x        |
| Routing / sort   | `mxfp4_moe_sort` + `p0/p1/p23` + grouped top-k    | 7.9       | routing + shared-expert MLP + add | 31.3      | 3.97x        |
| **MoE subtotal** |                                                    | **158.2** |                                   | **152.7** | **0.97x**    |

## Dense GEMM

| Component          | MI355X kernel                                   | MI355X ms | GB300 kernel           | GB300 ms  | GB300/MI355X |
| ------------------ | ----------------------------------------------- | --------- | ---------------------- | --------- | ------------ |
| o_proj GEMM                  | PTPC FP8                                | 36.2      | `nvjet_sm103`          | 35.5      | 0.98x        |
| o_proj input quant           | `dynamic_per_token_scaled_quant`       | 2.9       | fused / not standalone | —         |              |
| q_b_proj GEMM                | PTPC FP8                                | 15.0      | `nvjet_sm103`          | 12.1      | 0.81x        |
| q_b input norm + quant       | fused RMSNorm + per-token quant        | 3.1       | fused / not standalone | —         |              |
| q_a + kv_a GEMM              | PTPC FP8                                | 33.4      | `nvjet_sm103`          | 30.6      | 0.92x        |
| q_a + kv_a input quant       | `dynamic_per_token_scaled_quant`       | 5.3       | fused / not standalone | —         |              |
| Absorbed K/V BMM             | A16WFP4                                 | 51.2      | `nvjet_sm103`          | 11.0      | 0.21x        |
| Router GEMMs                 | AITER BF16                              | 7.3       | `nvjet_sm103`          | 3.8       | 0.52x        |
| DenseMLP L0–2                | DenseMLP L0–2                           | 4.0       | `nvjet_sm103`          | 3.1       | 0.77x        |
| **Dense subtotal**           |                                         | **158.5** |                        | **96.0**  | **0.61x**    |

## Communication and normalization

| Component           | MI355X kernel                                   | MI355X ms | GB300 kernel | GB300 ms  | GB300/MI355X |
| ------------------- | ----------------------------------------------- | --------- | ------------ | --------- | ------------ |
| All-reduce          | QuickReduce INT4                                | 138.3     | NCCL RING_LL | 126.9     | 0.92x        |
| RMSNorm             | `aiter::add_rmsnorm_quant`                      | 20.4      | `fused_add_rmsnorm` | 20.3 | 0.99x        |
| Other               | activation + output head + embedding + sampling | 0.4       | same         | 0.5       | 1.19x        |
| **TOTAL**           |                                                 | **703.1** |              | **633.8** | **0.90x**    |

## Lever ranking — excluding all-reduce and parity/faster rows

| Rank | Lever                | MI355X ms | GB300 ms | Excess ms | % of MI total | Work class                              |
| ---- | -------------------- | --------- | -------- | --------- | ------------- | --------------------------------------- |
| 1    | A16WFP4 absorbed BMM | 51.2      | 11.0     | 40.2      | 5.7%          | Kernel work                             |
| 2    | MoE gemm2            | 62.2      | 30.9     | 31.3      | 4.4%          | Optional; config exhausted, kernel work |
| 3    | MoE combine          | 33.3      | 22.3     | 11.0      | 1.6%          | HBM-bound; prior fusion lost            |
| 4    | q_a + kv_a PTPC path | 38.7      | 30.6     | 8.1       | 1.2%          | GEMM 33.4 ms + quant 5.3 ms             |
| 5    | DSA MQA logits       | 13.1      | 5.2      | 7.9       | 1.1%          | Gluon fallback; BLOCK_Q disabled        |
| 6    | q_b_proj PTPC path   | 18.1      | 12.1     | 6.0       | 0.9%          | GEMM 15.0 ms + fused norm/quant 3.1 ms  |
| 7    | DSA top-k            | 15.8      | 11.0     | 4.8       | 0.7%          | Near parity                             |
| 8    | o_proj PTPC path     | 39.1      | 35.5     | 3.6       | 0.5%          | GEMM 36.2 ms + quant 2.9 ms             |
| 9    | Router GEMMs         | 7.3       | 3.8      | 3.5       | 0.5%          | Tune padded M=8192/32768 buckets        |
| 10   | MoE act quant        | 6.1       | 3.9      | 2.2       | 0.3%          | HBM-bound; prior fusion lost            |
| 11   | DenseMLP L0–2        | 4.0       | 3.1      | 0.9       | 0.1%          | Near parity                             |
| 12   | DSA prep / store     | 5.4       | 4.8      | 0.6       | 0.1%          | Near parity                             |
