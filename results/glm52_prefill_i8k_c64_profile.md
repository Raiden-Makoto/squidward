# GLM-5.2 prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4) vs GB300 (NVFP4)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm5.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm51` `a2b889bd20`; absorbed BMM reprofile `RM/glm52-mxfp4-bmm-tuning` `ebc1b48339`, AITER `7b3a801f`; `SGLANG_DSA_FP8_PROJ_GEMM=1`, `SGLANG_USE_MXFP4_MLA_BMM=1`, `SGLANG_DSA_FUSE_HADAMARD_QUANT=1`. MI355X image `raidenmakoto916/raidenmakoto:glm52-fork-20260810-ckgemm1`; GB300 ARM64 image `lmsysorg/sglang:dev-cu13-glm52-nvfp4` with `sglang-kernel 0.4.5` and `flashinfer 0.6.15.post1`.

## Attention


| Component              | MI355X kernel                      | MI355X ms | GB300 kernel                         | GB300 ms  | GB300/MI355X |
| ---------------------- | ---------------------------------- | --------- | ------------------------------------ | --------- | ------------ |
| Sparse MLA             | Triton sparse MLA                  | 210.2     | `fmhaSm100f` sparse FMHA             | 204.0     | 0.97x        |
| MLA cache              | `_fused_qk_rope_cat_and_cache_mla` | 7.2       | `RopeQuantize` + `set_mla_kv_buffer` | 9.8       | 1.36x        |
| q/k norm + rope        | q/k norm + rope                    | 1.5       | `RMSNorm`                            | 2.6       | 1.73x        |
| DSA top-k              | top-k transform                    | 14.1      | `topk_transform_prefill`             | 11.0      | 0.78x        |
| DSA MQA logits         | Gluon FP8 ragged MQA logits        | 8.1       | `deep_gemm::sm100_mqa_logits`        | 5.2       | 0.64x        |
| DSA prep/store + misc  | norm/rope/quant/cache + misc       | 4.1       | fused indexer prep/store + misc      | 4.8       | 1.17x        |
| **Attention subtotal** |                                    | **245.2** |                                      | **237.5** | **0.97x**    |




## MoE


| Component        | MI355X kernel                                  | MI355X ms | GB300 kernel                      | GB300 ms  | GB300/MI355X |
| ---------------- | ---------------------------------------------- | --------- | --------------------------------- | --------- | ------------ |
| Gate-up (gemm1)  | `mfma_moe1`                                    | 41.4      | `bmm_E2m1`                        | 64.2      | 1.55x        |
| Down (gemm2)     | `mfma_moe2`                                    | 45.6      | `bmm_Bfloat16_E2m1`               | 30.9      | 0.68x        |
| Combine          | `moe_reduction_kernel`                         | 27.0      | `finalizeKernelVecLoad`           | 22.3      | 0.83x        |
| Activation quant | `dynamic_per_group_scaled_quant`               | 3.4       | `NVFP4Quantize`                   | 3.9       | 1.15x        |
| Routing / sort   | `mxfp4_moe_sort` + `p0/p1/p23` + grouped top-k | 7.4       | routing + shared-expert MLP + add | 31.3      | 4.23x        |
| **MoE subtotal** |                                                | **124.8** |                                   | **152.7** | **1.22x**    |




## Dense GEMM


| Component              | MI355X kernel                    | MI355X ms | GB300 kernel           | GB300 ms | GB300/MI355X |
| ---------------------- | -------------------------------- | --------- | ---------------------- | -------- | ------------ |
| o_proj GEMM            | PTPC FP8                         | 33.7      | `nvjet_sm103`          | 35.5     | 1.05x        |
| o_proj input quant     | `dynamic_per_token_scaled_quant` | 2.4       | fused / not standalone | —        |              |
| q_b_proj GEMM          | PTPC FP8                         | 13.2      | `nvjet_sm103`          | 12.1     | 0.92x        |
| q_b input norm + quant | fused RMSNorm + per-token quant  | 2.8       | fused / not standalone | —        |              |
| q_a + kv_a GEMM        | PTPC FP8                         | 33.7      | `nvjet_sm103`          | 30.6     | 0.91x        |
| q_a + kv_a input quant | `dynamic_per_token_scaled_quant` | 4.4       | fused / not standalone | —        |              |
| Absorbed K/V BMM       | A16WFP4 - tuned                  | 32.3      | `nvjet_sm103`          | 11.0     | 0.34x        |
| Router GEMMs           | AITER BF16                       | 6.4       | `nvjet_sm103`          | 3.8      | 0.59x        |
| DenseMLP L0–2          | DenseMLP L0–2                    | 3.9       | `nvjet_sm103`          | 3.1      | 0.79x        |
| **Dense subtotal**     |                                  | **133.8** |                        | **96.0** | **0.72x**    |




## Communication and normalization


| Component  | MI355X kernel                                   | MI355X ms | GB300 kernel        | GB300 ms  | GB300/MI355X |
| ---------- | ----------------------------------------------- | --------- | ------------------- | --------- | ------------ |
| All-reduce | QuickReduce INT4                                | 124.5     | NCCL RING_LL        | 126.9     | 1.02x        |
| RMSNorm    | `aiter::add_rmsnorm_quant`                      | 18.2      | `fused_add_rmsnorm` | 20.3      | 1.12x        |
| Other      | activation + output head + embedding + sampling | 0.2       | same                | 0.5       | 2.50x        |
| **TOTAL**  |                                                 | **638.6** |                     | **633.8** | **0.99x**    |




## Lever ranking — excluding all-reduce and parity/faster rows


| Rank | Lever                | MI355X ms | GB300 ms | Excess ms | % of MI total | Work class                              |
| ---- | -------------------- | --------- | -------- | --------- | ------------- | --------------------------------------- |
| 1    | MoE gemms            | 45.6      | 30.9     | 14.7      | 2.3%          | Optional; config exhausted, kernel work |
| 2    | Absorbed K/V BMM     | 25.3      | 11.0     | 14.3      | 2.2%          | NEEDS NEW KERNEL                        |
| 3    | q_a + kv_a PTPC path | 38.1      | 30.6     | 7.5       | 1.2%          | COMPLETED                               |
| 4    | MoE combine          | 27.0      | 22.3     | 4.7       | 0.7%          | Near-parity                             |
| 5    | q_b_proj PTPC path   | 16.0      | 12.1     | 3.9       | 0.6%          | COMPLETED                               |
| 6    | DSA top-k            | 14.1      | 11.0     | 3.1       | 0.5%          | Near parity                             |
| 7    | DSA MQA logits       | 8.1       | 5.2      | 2.9       | 0.4%          | Depends on aiter #4180                  |
| 8    | Router GEMMs         | 6.4       | 3.8      | 2.6       | 0.4%          | Tune padded M=8192/32768 buckets        |
| 9    | DenseMLP L0–2        | 3.9       | 3.1      | 0.8       | 0.1%          | Near parity                             |
| 10   | o_proj PTPC path     | 36.1      | 35.5     | 0.6       | 0.1%          | COMPLETED                               |


