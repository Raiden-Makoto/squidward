# ROCm tree speculative sampling validation

Branch `RM/rocm-tree-spec-sampling-triton`.

Capture `ac114dcb0b`, component benchmark `d903206a6f`, top-p candidate
`d24cec19d8`, candidate benchmark `ee081c5224`, model
`zai-org/GLM-5.2-FP8@ba978f7d347eaf65d22f1a86833408afdb953541`.

## Triton tree verifier

| Platform | Passed | Subtests | Skipped | CUDA AOT oracle |
| -------- | -----: | -------: | ------: | --------------- |
| B200     |     36 |       41 |       0 | Passed          |

## Renormalization correctness

| Case | Previous fallback | Corrected fallback | Observed difference | Reference |
| ---- | ----------------- | ------------------ | ------------------: | --------- |
| Top-k cutoff ties | Kept exactly `k`; ties broken by sort order | Keeps every probability at or above the kth-value cutoff | 429 support differences / 1536 rows | FlashInfer cutoff semantics |
| Top-p `p=1.0` | Compared float32 cumsum against literal `1.0` | Compares against each row's own total probability | FlashInfer kept 4 / 133177 nonzero tokens on a peaked row | Preserve full row support |

## Compatibility policy

| Path | Policy |
| ---- | ------ |
| Top-k | Match FlashInfer value-cutoff support semantics |
| Top-p `p<1.0` | Match nucleus cutoff semantics |
| Top-p `p=1.0` | Preserve the full normalized row; do not regress to AOT behavior |

## Renormalization implementation

| Change | Before | After | Result |
| ------ | ------ | ----- | ------ |
| Pivot selection | `torch.sort`, `O(V log V)` | Exact `torch.topk`, `O(V)` selection | Same cutoff semantics |
| Apply + normalize | Materialized masked copy | Fused HIP Triton kernel; one vocabulary read | 3.7 GB → 2.8 GB traffic |
| Top-p prefix | 1024 entries | 4096 entries | Top-k selection remains flat through 4096 |
| Nucleus 1024–4096 | 13.4 ms sort fallback | 2.88 ms prefix path | 16.95 ms → 2.88 ms |
| Rows already within prefix | Baseline | Wider prefix | +0.19 ms |

## Shipped platform paths

| Platform | Tree verification | Top-k / top-p renormalization |
| -------- | ----------------- | ----------------------------- |
| MI355X / ROCm | `tree_speculative_sampling_target_only_triton` | `top_k_renorm_probs_triton`, `top_p_renorm_probs_triton` |
| B200 / CUDA | `sgl_kernel.tree_speculative_sampling_target_only` | FlashInfer `top_k_renorm_probs`, `top_p_renorm_probs` |

## Production capture

| Hardware | Vocab | Top-k | Steps | Draft tokens | Captured rows | Batch sizes | Decode call indices |
| -------- | ----: | ----: | ----: | -----------: | ------------: | ----------- | ------------------ |
| MI355X | 154880 | 2 | 5 | 6 | 192 | 1 / 2 / 4 / 8 | 0–7485 |

| Metric | Min | Median | Max |
| ------ | --: | -----: | --: |
| Maximum token probability | 0.3132 | 0.9970 | 1.0000 |
| Top-p nucleus size | 1 | 1 | 15 |
| Acceptance length | 0 | 3 | 5 |

| Metric | Value |
| ------ | ----: |
| Rows with nucleus > 4096 | 0 / 192 |
| Top-k renormalization calls | 0 |
| Batch-8 decode calls | 7336 |
| Total decode calls | 7486–8191 |
| Batch-8 call share | 89.6–98.0% |

## MI355X component latency

Milliseconds, p50 unless marked p95. Real captured rows repeated to each batch size.

| Batch | Rows | Top-p pivot | Masked sum | Masked scale | Apply | Full top-p | Full top-p p95 | Tree verifier | Tree p95 | Index move / layer | Index move / 78 layers |
| ----: | ---: | ----------: | ---------: | -----------: | ----: | ---------: | -------------: | ------------: | --------: | -----------------: | ----------------------: |
| 1 | 6 | 0.162 | 0.016 | 0.017 | 0.037 | 0.210 | 0.244 | 0.085 | 0.088 | 0.0286 | 1.886 |
| 2 | 12 | 0.200 | 0.016 | 0.016 | 0.036 | 0.252 | 0.292 | 0.085 | 0.089 | 0.0267 | 1.924 |
| 4 | 24 | 0.227 | 0.015 | 0.015 | 0.036 | 0.265 | 0.307 | 0.085 | 0.095 | 0.0264 | 1.901 |
| 8 | 48 | 0.257 | 0.014 | 0.017 | 0.035 | 0.297 | 0.328 | 0.096 | 0.106 | 0.0262 | 1.867 |
| 32 | 192 | 0.445 | 0.025 | 0.038 | 0.060 | 0.512 | 0.545 | 0.126 | 0.131 | 0.0268 | 1.885 |
| 128 | 768 | 1.360 | 0.086 | 0.199 | 0.279 | 1.651 | 1.662 | 0.163 | 0.165 | 0.0268 | 1.891 |
| 256 | 1536 | 2.334 | 0.166 | 0.370 | 0.539 | 2.885 | 2.901 | 0.198 | 0.202 | 0.0266 | 1.887 |

## MI355X top-p fast path

Milliseconds, p50. Fast-path coverage: 192 / 192 captured rows.

| Batch | Rows | Baseline | Topk64 selection | Scale apply | Scatter apply | Scale full | Scatter full | Scale speedup | Scatter speedup |
| ----: | ---: | -------: | ---------------: | ----------: | ------------: | ---------: | -----------: | ------------: | --------------: |
| 1 | 6 | 0.211 | 0.141 | 0.016 | 0.013 | 0.199 | 0.202 | 1.059x | 1.043x |
| 2 | 12 | 0.247 | 0.160 | 0.016 | 0.011 | 0.214 | 0.219 | 1.155x | 1.128x |
| 4 | 24 | 0.262 | 0.192 | 0.015 | 0.011 | 0.246 | 0.251 | 1.067x | 1.043x |
| 8 | 48 | 0.296 | 0.212 | 0.017 | 0.012 | 0.266 | 0.270 | 1.114x | 1.095x |
| 32 | 192 | 0.513 | 0.365 | 0.038 | 0.023 | 0.438 | 0.431 | 1.170x | 1.190x |
| 128 | 768 | 1.631 | 1.151 | 0.195 | 0.085 | 1.360 | 1.292 | 1.199x | 1.262x |
| 256 | 1536 | 2.886 | 2.058 | 0.371 | 0.162 | 2.453 | 2.269 | 1.176x | 1.272x |

## Optimization ranking

| Rank | Path | Batch-8 ms | Batch-8 share | Batch-256 ms | Batch-256 share |
| ---: | ---- | ---------: | ------------: | -----------: | --------------: |
| 1 | DSA index relocation, 78 layers | 1.867 | 82.8% | 1.887 | 41.3% |
| 2 | Top-p topk64 selection | 0.212 | 9.4% | 2.058 | 45.1% |
| 3 | Target-only tree verifier | 0.096 | 4.2% | 0.198 | 4.3% |
| 4 | Top-p scale apply | 0.017 | 0.8% | 0.371 | 8.1% |
| 5 | Top-p full-sort fallback | 0 | 0.0% | 0 | 0.0% |
