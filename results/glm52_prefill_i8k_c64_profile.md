# GLM-5.2 prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4) vs GB300 (NVFP4)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm5.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm51` `89cec3bb97`; stock AITER `d9e5ef7ce0`, `SGLANG_DSA_FP8_PROJ_GEMM=1`, `SGLANG_USE_MXFP4_MLA_BMM=1`, `SGLANG_DSA_FUSE_HADAMARD_QUANT=1`. MI355X image `raidenmakoto916/raidenmakoto:glm52-fork-20260810-ckgemm1`; GB300 ARM64 image `lmsysorg/sglang:dev-cu13-glm52-nvfp4` with `sglang-kernel 0.4.5` and `flashinfer 0.6.15.post1`.

## Attention

| Component              | MI355X kernel                      | MI355X ms | GB300 kernel                         | GB300 ms  | GB300/MI355X |
| ---------------------- | ---------------------------------- | --------- | ------------------------------------ | --------- | ------------ |
| Sparse MLA             | Triton sparse MLA                  | 235.4     | `fmhaSm100f` sparse FMHA             | 204.0     | 0.87x        |
| MLA cache              | `_fused_qk_rope_cat_and_cache_mla` | 8.5       | `RopeQuantize` + `set_mla_kv_buffer` | 9.8       | 1.15x        |
| q/k norm + rope        | q/k norm + rope                    | 1.6       | `RMSNorm`                            | 2.6       | 1.63x        |
| DSA top-k              | top-k transform                    | 15.3      | `topk_transform_prefill`             | 11.0      | 0.72x        |
| DSA MQA logits         | Gluon FP8 ragged MQA logits        | 9.2       | `deep_gemm::sm100_mqa_logits`        | 5.2       | 0.57x        |
| DSA prep/store + misc  | norm/rope/quant/cache + misc       | 4.4       | fused indexer prep/store + misc      | 4.8       | 1.09x        |
| **Attention subtotal** |                                    | **274.3** |                                      | **237.5** | **0.87x**    |

## MoE

| Component        | MI355X kernel                                  | MI355X ms | GB300 kernel                      | GB300 ms  | GB300/MI355X |
| ---------------- | ---------------------------------------------- | --------- | --------------------------------- | --------- | ------------ |
| Gate-up (gemm1)  | `mfma_moe1`                                    | 45.8      | `bmm_E2m1`                        | 64.2      | 1.40x        |
| Down (gemm2)     | `mfma_moe2`                                    | 51.1      | `bmm_Bfloat16_E2m1`               | 30.9      | 0.60x        |
| Combine          | `moe_reduction_kernel`                         | 30.4      | `finalizeKernelVecLoad`           | 22.3      | 0.73x        |
| Activation quant | `dynamic_per_group_scaled_quant`               | 4.0       | `NVFP4Quantize`                   | 3.9       | 0.98x        |
| Routing / sort   | `mxfp4_moe_sort` + `p0/p1/p23` + grouped top-k | 8.1       | routing + shared-expert MLP + add | 31.3      | 3.86x        |
| **MoE subtotal** |                                                | **139.4** |                                   | **152.7** | **1.10x**    |

## Dense GEMM

| Component              | MI355X kernel                    | MI355X ms | GB300 kernel           | GB300 ms | GB300/MI355X |
| ---------------------- | -------------------------------- | --------- | ---------------------- | -------- | ------------ |
| o_proj GEMM            | PTPC FP8                         | 38.9      | `nvjet_sm103`          | 35.5     | 0.91x        |
| o_proj input quant     | `dynamic_per_token_scaled_quant` | 2.9       | fused / not standalone | —        |              |
| q_b_proj GEMM          | PTPC FP8                         | 15.0      | `nvjet_sm103`          | 12.1     | 0.81x        |
| q_b input norm + quant | fused RMSNorm + per-token quant  | 3.1       | fused / not standalone | —        |              |
| q_a + kv_a GEMM        | PTPC FP8                         | 38.2      | `nvjet_sm103`          | 30.6     | 0.80x        |
| q_a + kv_a input quant | `dynamic_per_token_scaled_quant` | 5.4       | fused / not standalone | —        |              |
| Absorbed K/V BMM       | A16WFP4                          | 43.9      | `nvjet_sm103`          | 11.0     | 0.25x        |
| Router GEMMs           | AITER BF16                       | 7.3       | `nvjet_sm103`          | 3.8      | 0.52x        |
| DenseMLP L0–2          | DenseMLP L0–2                    | 4.4       | `nvjet_sm103`          | 3.1      | 0.70x        |
| **Dense subtotal**     |                                  | **159.1** |                        | **96.0** | **0.60x**    |

## Communication and normalization

| Component  | MI355X kernel                                   | MI355X ms | GB300 kernel        | GB300 ms  | GB300/MI355X |
| ---------- | ----------------------------------------------- | --------- | ------------------- | --------- | ------------ |
| All-reduce | QuickReduce INT4                                | 146.4     | NCCL RING_LL        | 126.9     | 0.87x        |
| RMSNorm    | `aiter::add_rmsnorm_quant`                      | 20.4      | `fused_add_rmsnorm` | 20.3      | 0.99x        |
| Other      | activation + output head + embedding + sampling | 0.2       | same                | 0.5       | 2.00x        |
| **TOTAL**  |                                                 | **739.9** |                     | **633.8** | **0.86x**    |

## Lever ranking — excluding all-reduce and parity/faster rows

| Rank | Lever                | MI355X ms | GB300 ms | Excess ms | % of MI total | Work class                              |
| ---- | -------------------- | --------- | -------- | --------- | ------------- | --------------------------------------- |
| 1    | A16WFP4 absorbed BMM | 43.9      | 11.0     | 32.9      | 4.4%          | Kernel work                             |
| 2    | MoE gemm2            | 51.1      | 30.9     | 20.2      | 2.7%          | Optional; config exhausted, kernel work |
| 3    | q_a + kv_a PTPC path | 43.6      | 30.6     | 13.0      | 1.8%          | GEMM 38.2 ms + quant 5.4 ms             |
| 4    | MoE combine          | 30.4      | 22.3     | 8.1       | 1.1%          | HBM-bound; prior fusion lost            |
| 5    | o_proj PTPC path     | 41.8      | 35.5     | 6.3       | 0.9%          | GEMM 38.9 ms + quant 2.9 ms             |
| 6    | q_b_proj PTPC path   | 18.1      | 12.1     | 6.0       | 0.8%          | GEMM 15.0 ms + fused norm/quant 3.1 ms  |
| 7    | DSA top-k            | 15.3      | 11.0     | 4.3       | 0.6%          | Near parity                             |
| 8    | DSA MQA logits       | 9.2       | 5.2      | 4.0       | 0.5%          | Gluon fallback; BLOCK_Q disabled        |
| 9    | Router GEMMs         | 7.3       | 3.8      | 3.5       | 0.5%          | Tune padded M=8192/32768 buckets        |
| 10   | DenseMLP L0–2        | 4.4       | 3.1      | 1.3       | 0.2%          | Near parity                             |
| 11   | MoE act quant        | 4.0       | 3.9      | 0.1       | 0.0%          | HBM-bound; prior fusion lost            |
