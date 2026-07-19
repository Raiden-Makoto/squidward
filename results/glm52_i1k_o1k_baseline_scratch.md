# GLM-5.2-MXFP4 i1k/o1k (scratch)

TP4 MI355X GPUs 4-7, port 8552. `bash utilities/e2e_glm5.sh 1024 1024 0`
(random, input=1024, output=1024, num-prompts=conc*4), graphs-on. Both tables are
**same-session, one clean sweep each** (baseline `SGLANG_DSA_FP8_PROJ_GEMM=0`, per-proj `=1`;
env confirmed via `/proc/<pid>/environ`). 3 reps per concurrency; each cell is the
**median of the 3 reps** (median rejects the first-run JIT/autotune warmup spike, e.g.
base c8 rep1=317 vs 257 warm, per-proj c32 rep1=784 vs ~707/728 warm, and queue-variance
highs at c64).

## Baseline — fp8-proj off (`SGLANG_DSA_FP8_PROJ_GEMM=0`, bf16), same-session as feature below

Post q_b-quant-fusion stack (commit 2a6c41ed91). 3-rep median.

| concurrency | TTFT (ms) | ITL (ms) | E2EL (ms) | output tok/s |
|-------------|-----------|----------|-----------|--------------|
| 4           | 198.9     | 11.80    | 12280     | 333.21       |
| 8           | 258.5     | 13.71    | 14380     | 567.36       |
| 16          | 510.3     | 16.45    | 17370     | 942.75       |
| 32          | 749.6     | 20.43    | 21936     | 1492.67      |
| 64          | 1157.8    | 27.10    | 29693     | 2205.74      |

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

## Feature — fp8-proj + q_b quant fusion (`SGLANG_DSA_FP8_PROJ_GEMM=1`, o_proj fp8 M>16, q_b prefill-only)

o_proj → tuned fp8 CK GEMM for M>16 (decode M>=32 + prefill), bf16 at M<=16; q_b_proj → fp8 only
at prefill (M>512), bf16 at decode, with its activation quant folded into q_a_layernorm (commit
2a6c41ed91). GSM8K 200q = 0.945 / 0 invalid. Same-session 3-rep median. Δ = vs baseline above.

| concurrency | TTFT (ms) | Δ | ITL (ms) | Δ | E2EL (ms) | Δ | output tok/s | Δ |
|-------------|-----------|---|----------|---|-----------|---|--------------|---|
| 4           | 197.5     | −0.7% | 11.83 | +0.3% | 12337 | +0.5% | 331.70 | −0.5% |
| 8           | 256.3     | −0.9% | 13.70 | −0.1% | 14367 | −0.1% | 570.60 | +0.6% |
| 16          | 497.7     | −2.5% | 16.39 | −0.4% | 17308 | −0.4% | 945.69 | +0.3% |
| 32          | 797.6     | +6.4%* | 20.24 | −0.9% | 21764 | −0.8% | 1505.05 | +0.8% |
| 64          | 1127.5    | −2.6% | 27.04 | −0.2% | 29583 | −0.4% | 2214.17 | +0.4% |

Net E2E win: throughput +0.3–0.8% (c4 −0.5% noise), E2EL neutral-to-better (−0.4/−0.8% at
c16/c32/c64), TTFT neutral-to-better (c16 −2.5%, c64 −2.6%), ITL neutral. The earlier cross-session
+4.5% c32/c64 TTFT regression is gone: it was the q_b fp8 host-idle penalty (fixed by the fusion,
prefill idle% 48.6→45.4) plus cross-session drift.

*c32 TTFT +6.4% is noise: v2 reps 753/798/878 overlap baseline 729/750/791; the median caught a
high v2 rep. c16/c64 (also queue-variance-prone) both improved, so no real c32 regression.
