# ROCm tree speculative sampling validation

Branch `RM/rocm-tree-spec-sampling-triton`.

Capture `ac114dcb0b`, component benchmark `d903206a6f`, top-p candidate
`f1849b3772`, candidate benchmark `9d069df673`, selection breakdown
`b681a216c7`, model
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

| Batch | Rows | Baseline | Topk32 selection | Scale apply | Scatter apply | Scale full | Scatter full | Scale speedup | Scatter speedup |
| ----: | ---: | -------: | ---------------: | ----------: | ------------: | ---------: | -----------: | ------------: | --------------: |
| 1 | 6 | 0.211 | 0.139 | 0.017 | 0.012 | 0.201 | 0.200 | 1.049x | 1.058x |
| 2 | 12 | 0.252 | 0.157 | 0.017 | 0.014 | 0.214 | 0.218 | 1.176x | 1.157x |
| 4 | 24 | 0.266 | 0.189 | 0.017 | 0.013 | 0.246 | 0.249 | 1.083x | 1.068x |
| 8 | 48 | 0.301 | 0.207 | 0.018 | 0.012 | 0.263 | 0.267 | 1.146x | 1.126x |
| 32 | 192 | 0.516 | 0.367 | 0.039 | 0.023 | 0.440 | 0.434 | 1.172x | 1.190x |
| 128 | 768 | 1.646 | 1.155 | 0.196 | 0.083 | 1.367 | 1.294 | 1.204x | 1.273x |
| 256 | 1536 | 2.873 | 2.055 | 0.388 | 0.156 | 2.430 | 2.262 | 1.182x | 1.270x |

## MI355X top-k sorting

Milliseconds, p50. Fast-path coverage: 192 / 192 captured rows.

| Batch | Topk16 sorted | Topk16 unsorted + sort | Topk32 sorted | Topk32 unsorted + sort | Topk64 sorted | Topk64 unsorted + sort |
| ----: | ------------: | ----------------------: | ------------: | ----------------------: | ------------: | ----------------------: |
| 1 | 0.137 | 0.156 | 0.136 | 0.156 | 0.137 | 0.156 |
| 2 | 0.155 | 0.166 | 0.157 | 0.167 | 0.160 | 0.168 |
| 4 | 0.186 | 0.193 | 0.189 | 0.197 | 0.192 | 0.201 |
| 8 | 0.202 | 0.212 | 0.206 | 0.216 | 0.212 | 0.222 |
| 32 | 0.365 | 0.374 | 0.367 | 0.376 | 0.368 | 0.376 |
| 128 | 1.154 | 1.162 | 1.156 | 1.165 | 1.159 | 1.169 |
| 256 | 2.046 | 2.052 | 2.051 | 2.059 | 2.059 | 2.068 |

## MI355X topk32 selection breakdown

Milliseconds, p50. CPU sync is wall time; remaining columns use GPU events.

| Batch | Row sum | `torch.topk` | Prefix math | GPU fallback check | CPU fallback sync | Combined selection |
| ----: | ------: | -----------: | ----------: | -----------------: | ----------------: | -----------------: |
| 1 | 0.016 | 0.078 | 0.096 | 0.007 | 0.019 | 0.142 |
| 2 | 0.015 | 0.101 | 0.087 | 0.006 | 0.018 | 0.157 |
| 4 | 0.015 | 0.117 | 0.086 | 0.006 | 0.018 | 0.186 |
| 8 | 0.016 | 0.132 | 0.100 | 0.009 | 0.019 | 0.208 |
| 32 | 0.029 | 0.286 | 0.101 | 0.009 | 0.019 | 0.366 |
| 128 | 0.083 | 1.015 | 0.102 | 0.009 | 0.019 | 1.158 |
| 256 | 0.170 | 1.814 | 0.101 | 0.009 | 0.019 | 2.053 |

## Optimization ranking

| Rank | Path | Batch-8 ms | Batch-8 share | Batch-256 ms | Batch-256 share |
| ---: | ---- | ---------: | ------------: | -----------: | --------------: |
| 1 | DSA index relocation, 78 layers | 1.867 | 82.9% | 1.887 | 41.5% |
| 2 | Top-p topk32 selection | 0.207 | 9.2% | 2.055 | 45.2% |
| 3 | Target-only tree verifier | 0.096 | 4.2% | 0.198 | 4.3% |
| 4 | Top-p scale apply | 0.018 | 0.8% | 0.388 | 8.5% |
| 5 | Top-p full-sort fallback | 0 | 0.0% | 0 | 0.0% |
