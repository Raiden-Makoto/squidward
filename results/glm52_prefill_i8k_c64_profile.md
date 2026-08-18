# GLM-5.2 prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4) vs GB300 (NVFP4)

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm5.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`, `RM/glm51` `ef79d71d17`, AITER `0c76039b4`; `SGLANG_DSA_FP8_PROJ_GEMM=1`, `SGLANG_USE_MXFP4_MLA_BMM=1`, `SGLANG_DSA_FUSE_HADAMARD_QUANT=1`. MI355X image `raidenmakoto916/raidenmakoto:glm52-fork-20260810-ckgemm1`; GB300 ARM64 image `lmsysorg/sglang:dev-cu13-glm52-nvfp4` with `sglang-kernel 0.4.5` and `flashinfer 0.6.15.post1`.

## Attention

| Component              | MI355X kernel                      | MI355X ms | GB300 kernel                         | GB300 ms  | GB300/MI355X |
| ---------------------- | ---------------------------------- | --------- | ------------------------------------ | --------- | ------------ |
| Sparse MLA             | Triton sparse MLA                  | 232.3     | `fmhaSm100f` sparse FMHA             | 204.0     | 0.88x        |
| MLA cache              | `_fused_qk_rope_cat_and_cache_mla` | 8.0       | `RopeQuantize` + `set_mla_kv_buffer` | 9.8       | 1.23x        |
| q/k norm + rope        | q/k norm + rope                    | 1.0       | `RMSNorm`                            | 2.6       | 2.60x        |
| DSA top-k              | top-k transform                    | 17.1      | `topk_transform_prefill`             | 11.0      | 0.64x        |
| DSA MQA logits         | Gluon FP8 ragged MQA logits        | 10.4      | `deep_gemm::sm100_mqa_logits`        | 5.2       | 0.50x        |
| DSA prep/store + misc  | norm/rope/quant/cache + misc       | 12.4      | fused indexer prep/store + misc      | 4.8       | 0.39x        |
| **Attention subtotal** |                                    | **281.3** |                                      | **237.5** | **0.84x**    |

## MoE

| Component        | MI355X kernel                                  | MI355X ms | GB300 kernel                      | GB300 ms  | GB300/MI355X |
| ---------------- | ---------------------------------------------- | --------- | --------------------------------- | --------- | ------------ |
| Gate-up (gemm1)  | FlyDSL MXFP4                                   | 45.2      | `bmm_E2m1`                        | 64.2      | 1.42x        |
| Down (gemm2)     | FlyDSL MXFP4                                   | 51.8      | `bmm_Bfloat16_E2m1`               | 30.9      | 0.60x        |
| Combine          | scatter reduce                                 | 30.4      | `finalizeKernelVecLoad`           | 22.3      | 0.73x        |
| Activation quant | `dynamic_per_group_scaled_quant`               | 4.0       | `NVFP4Quantize`                   | 3.9       | 0.98x        |
| Routing / sort   | `mxfp4_moe_sort` + `p0/p1/p23` + grouped top-k | 7.9       | routing + shared-expert MLP + add | 31.3      | 3.96x        |
| **MoE subtotal** |                                                | **139.3** |                                   | **152.7** | **1.10x**    |

## Dense GEMM

| Component              | MI355X kernel                    | MI355X ms | GB300 kernel           | GB300 ms | GB300/MI355X |
| ---------------------- | -------------------------------- | --------- | ---------------------- | -------- | ------------ |
| o_proj GEMM            | PTPC FP8                         | 36.6      | `nvjet_sm103`          | 35.5     | 0.97x        |
| o_proj input quant     | `dynamic_per_token_scaled_quant` | 2.7       | fused / not standalone | —        |              |
| q_b_proj GEMM          | PTPC FP8                         | 14.6      | `nvjet_sm103`          | 12.1     | 0.83x        |
| q_b input norm + quant | fused RMSNorm + per-token quant  | 3.7       | fused / not standalone | —        |              |
| q_a + kv_a GEMM        | PTPC FP8                         | 37.5      | `nvjet_sm103`          | 30.6     | 0.82x        |
| q_a + kv_a input quant | `dynamic_per_token_scaled_quant` | 5.4       | fused / not standalone | —        |              |
| Absorbed K/V BMM       | A16WFP4 tuned                    | 36.2      | `nvjet_sm103`          | 11.0     | 0.30x        |
| Router GEMMs           | AITER BF16                       | 7.4       | `nvjet_sm103`          | 3.8      | 0.51x        |
| DenseMLP L0–2          | DenseMLP L0–2                    | 4.4       | `nvjet_sm103`          | 3.1      | 0.70x        |
| **Dense subtotal**     |                                  | **148.6** |                        | **96.0** | **0.65x**    |

## Communication and normalization

| Component  | MI355X kernel                                   | MI355X ms | GB300 kernel        | GB300 ms  | GB300/MI355X |
| ---------- | ----------------------------------------------- | --------- | ------------------- | --------- | ------------ |
| All-reduce | QuickReduce INT4                                | 158.4     | NCCL RING_LL        | 126.9     | 0.80x        |
| RMSNorm    | `aiter::add_rmsnorm_quant`                      | 20.3      | `fused_add_rmsnorm` | 20.3      | 1.00x        |
| Other      | activation + output head + embedding + sampling | 0.2       | same                | 0.5       | 2.50x        |
| **TOTAL**  |                                                 | **748.0** |                     | **633.8** | **0.85x**    |

## Lever ranking — excluding all-reduce and parity/faster rows

| Rank | Lever                | MI355X ms | GB300 ms | Excess ms | % of MI total | Work class        |
| ---- | -------------------- | --------- | -------- | --------- | ------------- | ----------------- |
| 1    | Sparse MLA           | 232.3     | 204.0    | 28.3      | 3.8%          | Kernel work       |
| 2    | Absorbed K/V BMM     | 36.2      | 11.0     | 25.2      | 3.4%          | Kernel work       |
| 3    | MoE gemm2            | 51.8      | 30.9     | 20.9      | 2.8%          | Kernel work       |
| 4    | q_a + kv_a PTPC path | 42.9      | 30.6     | 12.3      | 1.6%          | Kernel work       |
| 5    | MoE combine          | 30.4      | 22.3     | 8.1       | 1.1%          | Near-parity       |
| 6    | DSA prep / store     | 12.4      | 4.8      | 7.6       | 1.0%          | Chunking overhead |
| 7    | q_b_proj PTPC path   | 18.3      | 12.1     | 6.2       | 0.8%          | Kernel work       |
| 8    | DSA top-k            | 17.1      | 11.0     | 6.1       | 0.8%          | Kernel work       |
| 9    | DSA MQA logits       | 10.4      | 5.2      | 5.2       | 0.7%          | Kernel work       |
| 10   | o_proj PTPC path     | 39.3      | 35.5     | 3.8       | 0.5%          | Near-parity       |
| 11   | Router GEMMs         | 7.4       | 3.8      | 3.6       | 0.5%          | Kernel work       |
| 12   | DenseMLP L0–2        | 4.4       | 3.1      | 1.3       | 0.2%          | Near-parity       |
