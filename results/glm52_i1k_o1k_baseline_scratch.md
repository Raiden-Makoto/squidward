# GLM-5.2-MXFP4 i1k/o1k (scratch)

TP4 MI355X GPUs 4-7, port 8552. `bash utilities/e2e_glm5.sh 1024 1024 0`
(random, input=1024, output=1024, num-prompts=conc*4), graphs-on. Both tables are
**same-session, one clean sweep each** (baseline `SGLANG_DSA_FP8_PROJ_GEMM=0`, per-proj `=1`;
env confirmed via `/proc/<pid>/environ`). 3 reps per concurrency; each cell is the
**median of the 3 reps** (median rejects the first-run JIT/autotune warmup spike, e.g.
base c8 rep1=317 vs 257 warm, per-proj c32 rep1=784 vs ~707/728 warm, and queue-variance
highs at c64).

## Baseline — fp8-proj off (`SGLANG_DSA_FP8_PROJ_GEMM=0`, bf16)

| concurrency | TTFT (ms) | ITL (ms) | E2EL (ms) | output tok/s |
|-------------|-----------|----------|-----------|--------------|
| 4           | 199.1     | 11.82    | 12308     | 332.41       |
| 8           | 257.7     | 13.70    | 14346     | 570.97       |
| 16          | 508.6     | 16.34    | 17297     | 947.32       |
| 32          | 703.2     | 20.34    | 21869     | 1501.25      |
| 64          | 1131.1    | 27.08    | 29691     | 2204.85      |

## o_proj decode microbench — why FP8 is gated at M>16

Per-op GPU time (µs), o_proj K=4096 N=6144, cuda-graph + floor-amortized (many ops/graph)
+ cold weights (32 distinct buffers to defeat Infinity Cache) + tuned CK config. This is the
decode-representative per-layer cost. Earlier no-graph numbers were a ~9µs replay-floor artifact
(a trivial add_ also measured 9µs) — the quant is NOT expensive.

| M | bf16 gemm | quant | fp8 gemm | fp8 total (q+g) | winner |
|---|-----------|-------|----------|-----------------|--------|
| 4   | 12.02 | 2.02 | 12.63 | 14.65 | bf16 +22% |
| 8   | 12.04 | 2.07 | 12.65 | 14.72 | bf16 +22% |
| 16  | 12.50 | 2.16 | 11.58 | 13.74 | bf16 +10% |
| 32  | 14.42 | 2.38 | 11.60 | 13.99 | fp8 −3% |
| 64  | 16.35 | 2.89 | 12.69 | 15.58 | fp8 −5% |
| 128 | 20.26 | 3.94 | 14.60 | 18.54 | fp8 −9% |
| 512 | 36.95 | 10.00 | 27.08 | 37.09 | bf16 +0.4% |

At M<=16 the bf16 and fp8 GEMMs both cost ~12µs (latency-bound, not weight-BW bound), so fp8's
~2µs quant is pure overhead → fp8 loses (+22% at M=4). Crossover M≈32: bf16 scales with its 2×
weight bytes while fp8 stays flat → fp8 wins for M>=32. Fix: o_proj `_fp8_proj_m_min=16`.

## Feature — per-proj gated (`SGLANG_DSA_FP8_PROJ_GEMM=1`, o_proj fp8 M>16, q_b prefill-only)

o_proj → tuned fp8 CK GEMM for M>16 (decode M>=32 + prefill), bf16 at M<=16; q_b_proj → fp8 only
at prefill (M>512), bf16 at decode. GSM8K 400q = 0.932 / 0 invalid. Same-session clean sweep,
3-rep median. Δ = vs baseline above.

| concurrency | TTFT (ms) | Δ | ITL (ms) | Δ | E2EL (ms) | Δ | output tok/s | Δ |
|-------------|-----------|---|----------|---|-----------|---|--------------|---|
| 4           | 198.7     | −0.2% | 11.79 | −0.3% | 12266 | −0.3% | 333.71 | +0.4% |
| 8           | 238.4     | −7.5% | 13.71 | +0.1% | 14341 | −0.0% | 570.47 | −0.1% |
| 16          | 502.7     | −1.2% | 16.37 | +0.2% | 17314 | +0.1% | 946.38 | −0.1% |
| 32          | 735.2     | +4.5% | 20.31 | −0.1% | 21789 | −0.4% | 1502.82 | +0.1% |
| 64          | 1183.4    | +4.6% | 27.10 | +0.1% | 29663 | −0.1% | 2205.61 | +0.0% |

Decode ITL now neutral at all conc (was +2.1%/+1.8%/+1.2% at c4/c8/c16 under all-M). E2EL/tok-s
neutral. TTFT queue-variance-dominated at c32/c64 (rep spread c32 657-786, c64 1133-1232).
