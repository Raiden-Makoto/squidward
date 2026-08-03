# GLM-5.2 prefill — i8k/o16 conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

| Section                         | MI355X kernel                              | MI355X ms | B200 kernel                           | B200 ms | B200/MI355X |
| ------------------------------- | ------------------------------------------ | --------- | ------------------------------------- | ------- | ----------- |
| Dense layers (3) · Attn prepare | `quickreduce` + `add_rmsnorm_quant`        | 4.5       | `nvjet` + `fused_add_rmsnorm`         | 1.3     | 0.29x       |
| Dense layers (3) · Attention    | DSA + sparse MLA + projection GEMMs        | 16.7      | DSA + sparse FMHA + `nvjet` GEMMs     | 13.5    | 0.81x       |
| Dense layers (3) · MLP prepare  | `quickreduce` + `add_rmsnorm_quant`        | 4.7       | NCCL + `nvjet` + `fused_add_rmsnorm`  | 1.9     | 0.41x       |
| Dense layers (3) · MLP          | Tensile GEMMs + `act_and_mul`              | 3.8       | `nvjet` GEMMs + `act_and_mul`         | 1.7     | 0.46x       |
| MoE layers (75) · Attn prepare  | `quickreduce` + `add_rmsnorm_quant`        | 107.3     | `nvjet` + `fused_add_rmsnorm`         | 32.6    | 0.30x       |
| MoE layers (75) · Attention     | DSA + sparse MLA + projection GEMMs        | 345.5     | DSA + sparse FMHA + `nvjet` GEMMs     | 239.4   | 0.69x       |
| MoE layers (75) · MLP prepare   | `quickreduce` + `add_rmsnorm_quant`        | 107.4     | NCCL + `nvjet` + `fused_add_rmsnorm`  | 51.8    | 0.48x       |
| MoE layers (75) · MoE           | `mfma_moe1` + `mfma_moe2` + combine       | 125.0     | `bmm_E2m1` + `bmm_Bfloat16` + combine | 108.1   | 0.86x       |
| **Σ section duration**          |                                            | **714.9**  |                                       | **450.3** | **0.63x**   |
| **GPU-busy total**              |                                            | **1209.1** |                                       | **857.1** | **0.71x**   |
