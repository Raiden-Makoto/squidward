# MXFP4 absorbed BMM — MI355X
- Branch: `RM/glm51`
- Commit: `b685569c5e799ebe4ee656cb4b168e2efe3f8637`
- Settings: `graphs=off`, `input=8192`, `output=16`, `concurrency=64`, `prompts=256`, `profiled_forwards=4`
- Baseline: `/data/macui/results/mxfp4_absorbed_bmm_rerun_20260804_1418/baseline_profile`
- Feature: `/data/macui/results/mxfp4_absorbed_bmm_kblock_fix_20260804_1319/feature_profile`

| correctness | result |
| --- | ---: |
| focused unit tests | 9/9 |
| GSM8K baseline | 19/20 |
| GSM8K feature | 19/20 |
| K finite | 1 |
| K cosine | 0.9999985 |
| V finite | 1 |
| V cosine | 0.9999985 |

| TP0 EXTEND, per forward | baseline (µs) | feature (µs) | Δ (µs) | Δ |
| --- | ---: | ---: | ---: | ---: |
| K absorbed BMM | 45,263.150 | 20,744.975 | −24,518.175 | −54.168% |
| V absorbed BMM | 33,488.950 | 25,593.775 | −7,895.175 | −23.575% |
| K + V absorbed BMM | 78,752.100 | 46,338.750 | −32,413.350 | −41.159% |
| total attributed GPU kernels | 901,973.975 | 846,254.475 | −55,719.500 | −6.178% |

| A16WFP4 dispatch | calls | BLOCK_SIZE_K | NUM_KSPLIT |
| --- | ---: | ---: | ---: |
| K | 312 | 64 | 1 |
| V | 312 | 256 | 1 |

| profiled serving | baseline | feature | Δ |
| --- | ---: | ---: | ---: |
| duration (s) | 383.03 | 386.56 | +0.922% |
| input tok/s | 5,475.19 | 5,425.12 | −0.914% |
| output tok/s | 10.69 | 10.60 | −0.842% |
