# GLM-5.2 all-projections PTPC A/B — c4, i8k/o1k, REPS=4

- Branch/commit: `RM/glm52-ptpc-qkv-a-proj` / `cd1eef175db4c0eb9964f5b03aef96a1b2158061`
- AITER: `/sgl-workspace/aiter` / `6c48c5fa0e7c299adcf1987c8f4817720ceab311`
- Image: `raidenmakoto916/raidenmakoto:glm52-fork-aiter0724-ckaf7118-ckgemm1`
- Result root: `/data/macui/results/glm52_ptpc_all_projections_c4_i8k_o1k_reps4_20260806_231519`
- Launcher: `bash utilities/run_glm52.sh --use-triton`
- Benchmark: `CONCURRENCY=4 REPS=4 OUT_DIR=<arm-dir> bash utilities/e2e_glm5.sh 8192 1024 0`
- Accuracy: baseline 4/4, feature 4/4; 16/16 successful requests and 16384 generated tokens/run

A. Baseline (`SGLANG_DSA_FP8_PROJ_GEMM=0`)

| concurrency | TTFT median (ms) | ITL (ms) | E2EL (ms) | output tok/s |
| --- | --- | --- | --- | --- |
| 4 | 995.17 | 11.780 | 13376.78 | 305.68 |

B. Feature (`SGLANG_DSA_FP8_PROJ_GEMM=1`, Δ vs baseline)

| concurrency | TTFT median (ms) | Δ | ITL (ms) | Δ | E2EL (ms) | Δ | output tok/s | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 1014.52 | +1.94% | 11.815 | +0.30% | 13384.29 | +0.06% | 305.40 | −0.09% |

## TTFT by repetition

| repetition | Baseline mean TTFT (ms) | Feature mean TTFT (ms) | Baseline median TTFT (ms) | Feature median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 998.55 | 983.32 | 995.39 | 974.16 |
| 2 | 976.58 | 980.19 | 995.82 | 978.49 |
| 3 | 916.25 | 937.87 | 994.97 | 982.36 |
| 4 | 952.57 | 967.62 | 994.51 | 1123.08 |
| **Average** | **960.99** | **967.25** | **995.17** | **1014.52** |
| **Δ** | | **+0.65%** | | **+1.94%** |

## Dense GEMM profile

ms/prefill forward, TP0 EXTEND trace (n_fwd=2), graphs-off eager, c4 i8k/o16, `--profile-num-steps 4 --profile-by-stage`. Traces: `/tmp/glm52_ptpc_c4_profile_20260807_1438Z`.

| Component | Baseline kernel | Baseline ms | Feature kernel | Feature ms | Δ |
| --- | --- | ---: | --- | ---: | ---: |
| o_proj | AITER BF16 | 39.7 | PTPC FP8 | 32.7 | −17.7% |
| q_b_proj | AITER BF16 | 15.0 | PTPC FP8 | 13.5 | −9.7% |
| q_a + kv_a | AITER BF16 | 39.2 | PTPC FP8 | 34.4 | −12.3% |
| Absorbed K/V BMM | A16WFP4 | 46.2 | A16WFP4 | 45.6 | −1.3% |
| o_proj input quant | — | — | fused / not standalone | — | — |
| Router GEMMs | router GEMMs | 6.6 | router GEMMs | 6.8 | +2.3% |
| DenseMLP L0–2 | DenseMLP L0–2 | 3.5 | DenseMLP L0–2 | 3.6 | +1.9% |
| **Dense subtotal** | | **150.2** | | **136.5** | **−9.1%** |
