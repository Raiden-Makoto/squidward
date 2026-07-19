# GLM-5.2 prefill Dense GEMM — fp8-proj vs bf16 (scratch)

Same stack/session, TP4 MI355X GPUs 4-7, i1k/o1k conc64 num256 prefill, `--profile-num-steps 4
--profile-by-stage`, TP0 EXTEND trace, ms/prefill-forward (Total_us / n_fwd=4 / 1000).
Parsed with `utilities/glm5_prof_csv.py` (LeafModule attribution). bf16 = `SGLANG_DSA_FP8_PROJ_GEMM=0`,
fp8 = `=1` (gate16: o_proj fp8 M>16, q_b prefill-only). Only o_proj + q_b are fp8-marked; q_a+kv_a
(fused_qkv_a, not 128-aligned) and kv_b stay bf16.

Projection identified by kernel grid (grid = N-tiles × M-tiles). ms/prefill-forward (÷ n_fwd=4).

| Projection | bf16 kernel (grid → M,N) | bf16 ms | fp8 kernel (grid → M,N) | fp8 ms | Δ ms |
|------------|--------------------------|---------|--------------------------|--------|------|
| o_proj     | Cijk MT256x256/240 | 35.58 | QuantGemm grid≈2064 (M16384,N6144) | 26.32 | **−9.26** |
| q_b_proj   | bf16gemm_256x256 grid[16,64] (M16384,N4096) | 12.53 | ck blockscale_bpreshuffle grid≈4096 (M16384,N4096) | 11.81 | −0.72 |
| q_a+kv_a (bf16 both) | Cijk MT224x256/240x256 | 23.61 | (unchanged) | 23.86 | +0.25 |
| kv_b (bf16 both)     | Cijk MT256x256/256x240 | 8.22 | (unchanged) | 8.23 | +0.01 |
| router/misc (bf16 both) | hgemm + small Cijk | 2.15 | (unchanged) | 2.13 | −0.02 |
| **Dense GEMM total** |            | **82.09** |          | **72.35** | **−9.74** |
| fp8 act-quant (o_proj 2.26 + q_b 1.22) | — | 0 | dynamic_per_group_scaled_quant | 3.48 | +3.48 |
| **Net (GEMM + fp8 quant)** |          | **82.09** |          | **75.83** | **−6.26** |

Validated: q_a+kv_a and kv_b (bf16 in both configs) are identical across captures (±0.25ms).

Result — fp8 dense GEMM is **−9.7 ms/fwd** (o_proj −9.26, q_b −0.72; q_b is a small
N4096·K2048 GEMM so fp8 buys almost nothing there — the win is essentially all o_proj), net
**−6.3 ms/fwd** after the +3.5ms activation quant. So fp8 dense GEMM is faster at prefill; the
+4.5% TTFT is not a Dense-GEMM regression.

Note: q_b's fp8 GEMM is the `ck ...blockscale_b_preshuffle` kernel (grid≈4096 = M16384,N4096),
NOT the tiny `fp8gemm_blockscale_128x128` (grid[32,8]=M1024, 0.4ms tail-chunk remnant). Reading
the wrong kernel earlier produced a bogus q_b 0.71ms / −12ms. Matches reference q_b fp8 (12.0).

## Root cause of the +4.5% TTFT — host-side idle, not GEMM

GEMM time (above) is GPU-busy; TTFT is wall-clock. Measuring GPU wall-clock span vs union-busy
on the SAME EXTEND traces (whole trace, TP0):

| | bf16 | fp8 | Δ |
|---|------|-----|---|
| wall-clock span | 3213 ms | 3478 ms | **+265 ms (+8.2%)** |
| GPU-busy (union) | 1734.8 ms | 1787.7 ms | +53 ms |
| **GPU-idle** | 1478.4 ms | 1690.4 ms | **+212 ms (+14%)** |
| idle % | 46.0% | 48.6% | +2.6 pp |

fp8 wall-clock is +265ms longer, of which **+212ms is added GPU-idle**, not compute (busy is
flat — dense-GEMM savings offset by the added quant kernels). The idle is host-side: the fp8 path
adds per-layer eager work between launches — quant-kernel launches, per-call scale-tensor allocs,
and the `(fp8, scale)` tuple / `input_scale` branch in `apply()` — creating GPU bubbles. Prefill
is eager (variable M → not CUDA-graphable), so this launch overhead is paid every forward and
accumulates with more prefill forwards → grows with concurrency, matching the c32/c64-only
signature. A kernel-duration sum cannot see it (GPU-busy drops while wall-clock rises).

Levers (all host-side, reduce launches/allocs — NOT graphs, prefill can't be graphed):
- fold activation quant into the producer (o_proj → v_up bmm epilogue; q_b → `q_a_layernorm` via
  `fused_rms_fp8_group_quant`, which today only fires on `weight.dtype==fp8`, not bf16-marker).
- preallocate scale buffers instead of `torch.empty` per call.
- thin the tuple/`input_scale` dispatch in `apply()`.

## Fix applied — q_b quant folded into q_a_layernorm (commit 2a6c41ed91)

Extended the `fused_rms_fp8_group_quant` branch in `forward_mha` to fire for the bf16-marker
`_fp8_proj_ready` q_b (was gated on `weight.dtype==fp8`), folding q_b's activation quant into the
RMSNorm kernel → one fewer eager launch per layer at prefill. Re-profiled EXTEND (whole-trace TP0;
separate captures, so treat idle% + kernel-count as the robust signal, absolute span has capture
variance):

| | base bf16 | gate16 fp8 (no fusion) | gate16-v2 (+q_b fusion) |
|---|-----------|------------------------|--------------------------|
| kernels | 10136 | 10760 | 10130 |
| GPU-idle | 1478 ms | 1690 ms | 1296 ms |
| idle % | 46.0% | 48.6% | 45.4% |

Removed ~630 kernel launches (the standalone q_b `dynamic_per_group_scaled_quant`); prefill idle%
dropped 48.6% → 45.4%, back to the bf16 baseline — the fp8 host-idle penalty is erased.
GSM8K 200q = 0.945 / 0 invalid. Remaining lever: o_proj quant (still a standalone launch; needs
an attention-epilogue fusion in the backend, not a host-side change).

## Decision — revert o_proj to prefill-only; keep q_b fusion (commit pending)

Final config: q_b_proj AND o_proj FP8 **prefill-only** (M>512 → fp8, bf16 at decode), plus the
q_b activation-quant folded into q_a_layernorm at prefill. This is the original prefill-only plan
+ the host-side q_b fusion.

Why o_proj is NOT fp8 at decode (dropping the earlier M>16 gate): running o_proj fp8 at decode is
the only change that touches the decode path, and it raises **c32 TTFT (+7% median)** while every
prefill-only optimization *decreases* c32 TTFT. Cause: c32 sits at the prefill↔decode throughput
knee (tok/s scaling 1.72→1.66→1.58→1.48× per doubling from c4→c64 — decode saturating). Below the
knee (c≤16) TTFT is prefill-bound → fp8 prefill helps; above (c64) it's decode-bound → cheaper fp8
decode frees slots → helps; AT the knee (c32) neither dominates, so the extra o_proj fp8 GEMM +
quant on the decode steps purely contends with the interleaved new-request prefill → TTFT up. The
isolated decode-GEMM win is only ~−3% at M=32 (see microbench in glm52_i1k_o1k doc), not worth the
knee TTFT cost. So o_proj stays prefill-only. Decode-side note recorded per the revert decision.
