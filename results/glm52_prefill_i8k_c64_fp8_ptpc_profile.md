# GLM-5.2 prefill — i8k/o16 conc64, TP4 — baseline vs FP8 PTPC

ms/prefill forward, TP0 EXTEND trace (n_fwd=4), graphs-off eager, `utilities/e2e_glm5.sh 8192 16 1`, conc64 num256, `--profile-num-steps 4 --profile-by-stage`. Baseline `RM/glm51` `9e0563b9c3`; FP8 PTPC `RM/glm52-mixed-scale-ck` `e6911fa40b`, `SGLANG_DSA_FP8_PROJ_GEMM=1`, `SGLANG_USE_DSA_FP8_PROJ_PTPC=1`, `/sgl-workspace/aiter_fp8`.

| Section     | Baseline ms | FP8 PTPC ms | Δ      |
| ----------- | ----------: | -----------: | -----: |
| Attention   |       406.0 |        461.7 | +13.7% |
| MoE         |       144.5 |        155.8 |  +7.9% |
| Dense GEMM  |       175.2 |        198.5 | +13.3% |
| All-reduce  |       126.2 |        140.7 | +11.5% |
| RMSNorm     |        18.1 |         20.4 | +12.4% |
| Other       |         0.4 |          0.4 |  +8.4% |
| **TOTAL**   |   **870.5** |    **977.5** | **+12.3%** |

| Dense GEMM       | Baseline ms | FP8 PTPC ms | Δ      |
| ---------------- | ----------: | -----------: | -----: |
| q_b_proj         |        15.1 |         15.4 |  +1.9% |
| o_proj           |        39.6 |         36.3 |  −8.1% |
| **q_b + o_proj** |    **54.6** |     **51.7** | **−5.4%** |
| q_a + kv_a       |        37.5 |         43.9 | +17.1% |
| kv_b absorbed BMM |       73.3 |         91.8 | +25.2% |
| router GEMMs     |         6.2 |          7.1 | +13.2% |
| DenseMLP L0–2    |         3.5 |          3.9 | +12.9% |
