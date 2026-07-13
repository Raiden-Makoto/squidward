# GLM-5.2 MoE down-GEMM — findings (PINNED)

Reference doc for the aiter MoE stage-2 (down) GEMM investigation.
Goal was to close down GEMM 46.7 ms/fwd (MI355X) vs 26.2 ms (B200), i.e. the ~20 ms prefill gap.
Outcome: the mainloop is already at its practical floor (see Experiments); the gap is not a
mainloop rewrite target. Kept as a record so the disproven levers are not retried.

## Diagnosis (rocprofv3, real production forward via `bench_one_batch`, M≈573)

Kernel `mfma_moe2 ... t64x128x256 ... cshuffle ... persist`:

| metric | value | reading |
| --- | ---: | --- |
| MfmaUtil | 13% | matrix cores idle ~87% |
| VALUBusy | 26% | not dequant-throughput-bound |
| MemUnitStalled | 0.6% | NOT HBM-bandwidth-bound |
| OccupancyPercent | 37% | ~3 waves; VGPR 64 / LDS 32.5 KB allow ~8 |

Hypothesis (from `results/glm52_prefill_i1k_c64_profile.md`, still open): down-GEMM is
latency/dependency-bound / under-pipelined (2-stage ping-pong + per-tile `gpu.barrier()` at
~3 waves). E2E target: **gemm2 46.7 ms/fwd vs B200 26.2 ms (0.56x)** — B200 wins via better
SW pipelining (cutlass grouped `bmm`), NOT hardware. Proposed lever: deepen the pipeline
2->3/4-stage prefetch so fp4 dequant/scale overlaps MFMA **while keeping** the ping-pong
overlap.

## Experiments run so far — the proposed lever is still UNTESTED

Correct kernel/file: **`compile_mixed_moe_gemm2` in `mixed_moe_gemm_2stage.py`** (live dispatch
`moe_kernels.py:462`). Atomic vs REDUCE differ ONLY in the epilogue — **the mainloop is
shared**, so mainloop signals below apply to the production reduce kernel too. (Profile doc
line 57's `compile_moe_gemm2 / moe_gemm_2stage.py` is stale/wrong.)

microbench `-q 4 -dim 6144,2048 -e 257 -k 9 -t 512` (dispatches ATOMIC, same mainloop),
rocprofv3 kernel-trace, moe2 avg us:

| mainloop change | moe2 avg | vs baseline | what it tests |
| --- | ---: | ---: | --- |
| baseline (2-buffer ping-pong, 1 barrier/tile) | ~320 us | — | — |
| full-K resident (8 barriers -> 1, overlap REMOVED) | 422 us | +31% | barrier count is NOT the limiter; overlap is |
| `cu_num_mul=3` (occupancy 3x WGs) | 319 us | ~0% | occupancy knob is a wash |

**What is proven:** cutting barriers by killing overlap hurts; the occupancy knob does nothing.
**What is NOT proven / still open:** the profile doc's actual lever — **deepen the pipeline
2->3/4-stage prefetch that KEEPS the ping-pong overlap** (one more in-flight prefetch of
x/scale/b so fp4 dequant/scale overlaps MFMA). full-K was the OPPOSITE of this. This is the
real experiment and it has not been run. Down-GEMM (46.7 vs 26.2 ms, 0.56x) remains OPEN;
B200 wins by SW pipelining, not hardware. All experimental edits reverted (fork `01f52ea7`).

## Next (do this properly)

1. Implement 3-stage (then 4-stage) prefetch in `compile_mixed_moe_gemm2`: keep the 2 LDS
   ping-pong buffers' overlap, add a 3rd buffer + prefetch x/scale/b one tile further ahead.
   Do NOT collapse barriers to one (full-K proved that regresses).
2. Measure moe2 us vs baseline; validate correctness by logits_diff order-of-magnitude.
3. If a win, validate e2e on the reduce production path (`bench_one_batch` batch16/i1k, M≈573)
   and update the profile md with real ms/fwd deltas.

## Correct kernel target (DO NOT edit the wrong file)

- **`compile_mixed_moe_gemm2`** in `/home/macui/aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`
  (lines ~2910-4720; launch `launch_mixed_moe_gemm2` ~4720). Cache prefix `launch_mixed_moe_gemm2_*`.
- NOT `moe_gemm_2stage.py` / `compile_moe_gemm2` (that path is not executed for this shape — wasted a cycle there).
- Structure to change:
  - LDS sizing line ~3090: `lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)` (2-buffer).
  - Prologue barrier ~4285; a0/a1 prefetch ~4291.
  - K-loop ~4327: `for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2)` (2-tile ping-pong, Python-unrolled).
  - **2 `gpu.barrier()` per pair** at ~4361 and ~4405; tail at ~4417/4451/4476.
  - `num_k_tiles_py = num_k_tiles_per_batch`.

## Do NOT retry (already tested, no win)

- Grouped-K / full-K-resident LDS to cut barriers: TESTED (full-K = +31%). Barriers are not
  the limiter; the 2-buffer ping-pong's load/compute overlap is worth more than the barriers.
- Raising occupancy via `cu_num_mul` / `waves_per_eu` (the `..._async_w4_cumul3` variant): TESTED, ~0%.
- If ever changing `lds_x_bytes` @3090, ALSO fix the hardcoded `2 *` at ~3256
  (`lds_x_b = 2 * tile_m * lds_stride * a_elem_bytes`) that places `lds_tid`/`lds_tw`; leaving
  it stale overlaps the token-id table with extra X buffers -> HSA aperture violation.

## CRITICAL gotchas (these burned cycles — always do them)

1. **FlyDSL JIT cache silently serves the old kernel.** After ANY edit:
   `rm -rf /sgl-workspace/aiter_dev/aiter/jit/flydsl_cache/*` before testing. Clearing only
   `launch_mixed_moe_gemm2_*` was NOT enough in practice.
2. **`logits_diff` varies run-to-run** (random fp4 inputs) — it is NOT a reliable before/after delta.
   Judge correctness by a stable relative threshold and judge PERF by rocprof `MfmaUtil` / gemm2 us.
3. **Confirm the build actually ran**: add a `sys.stderr.write("GEMM2SHAPE ...")` in `compile_mixed_moe_gemm2`
   and verify it prints, else you're editing a non-executed path.
4. **rocprofv3 multi-metric `--pmc A B C ...` HANGS** (ROCTracer). Use single-metric passes
   (one `--pmc X` per run) and `--kernel-trace` (fast). Analyze with `utilities/pmc_pick.py` / `utilities/kt_analyze.py`.

## Dev workflow (worktree; /sgl-workspace/aiter stays untouched)

- Box `smci355-ccs-aus-m12-17.cs-aus.dcgpu`, container `glm5`. Box aiter = `9127c94a1` (detached).
- Scratch dev copy: `/sgl-workspace/aiter_dev` (cp of `/sgl-workspace/aiter`), imported via
  `PYTHONPATH=/sgl-workspace/aiter_dev`. Carry the final patch in the fork `/home/macui/aiter`
  (branch off `9127c94a1`; `moe_gemm_2stage.py` is identical across box/local, but edit the RIGHT file).
- Microbench (fast iteration): `cd /sgl-workspace/aiter_dev && PYTHONPATH=/sgl-workspace/aiter_dev python3 op_tests/test_moe_2stage.py -q 4 -dim 6144,2048 -e 257 -k 9 -t 512 --no-flydsl-csv`
  - `-t 512` dispatches the `t64x128x256` down GEMM (ATOMIC variant); production uses REDUCE (same mainloop, different epilogue).
- e2e validation: `utilities/run_glm52.sh --profile` + `bench_serving` i1k/o1k, EXTEND trace via
  `utilities/glm5_prof_csv.py` + `utilities/glm5_bucket.py`; GSM8K parity; keep all-reduce row carried-over.
