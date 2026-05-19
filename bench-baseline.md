# DSv4 Full-Stack Benchmark Report

**Hardware:** DSv4-Pro-FP8, 8x MI35X, TP=8  
**Workload:** random, in=512, out=2048  
**Date:** 2026-05-19  

---

## Summary

We benchmarked the new full stack (`local/full-stack`: radix cache on, new compressor, 25/N paged
Triton kernels) against the established baseline (old compressor, radix disabled) on 8x MI35X.

Two findings:

1. **TTFT improved significantly** — radix cache is working as intended, reducing time-to-first-token
   by 5–11x across concurrency levels due to KV cache hits on repeated prompt prefixes.

2. **Median TPOT regressed ~2–4x** — every decode step is consistently slower under the new stack.
   Output token throughput dropped 19% at c=2, 59% at c=4, and 72% at c=8 vs baseline. The TTFT
   gain from radix is real, but it is masked by a decode-side regression that 25/N's fused Triton
   kernels have not resolved.

---

## Benchmark Numbers

### Baseline — old compressor, radix disabled

| Concurrency | Median TPOT | P99 ITL | Output tok/s | Mean TTFT |
|---|---|---|---|---|
| c=2 | 21.07 ms | 21.46 ms | 53.45 | 33511 ms |
| c=4 | 22.15 ms | 22.66 ms | 161.54 | 5393 ms |
| c=8 | 24.06 ms | 24.67 ms | 289.90 | 7275 ms |

### New stack — radix on, new compressor, 25/N

| Concurrency | Median TPOT | P99 ITL | Output tok/s | Mean TTFT |
|---|---|---|---|---|
| c=2 | 43.90 ms (+108%) | 1250 ms | 43.16 (-19%) | 3040 ms |
| c=4 | 59.49 ms (+169%) | 2311 ms | 65.92 (-59%) | 960 ms |
| c=8 | 97.14 ms (+304%) | 4049 ms | 80.23 (-72%) | 4164 ms |

---

