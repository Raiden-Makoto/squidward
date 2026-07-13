# GLM-5.2 MoE down-GEMM rewrite — working notes (PINNED)

Reference doc for the upcoming aiter kernel rewrite of the MoE stage-2 (down) GEMM.
Goal: close down GEMM 46.7 ms/fwd (MI355X) vs 26.2 ms (B200), i.e. the ~20 ms prefill gap.

## Diagnosis (rocprofv3, real production forward via `bench_one_batch`, M≈573)

Kernel `mfma_moe2 ... t64x128x256 ... cshuffle ... persist`:

| metric | value | reading |
| --- | ---: | --- |
| MfmaUtil | 13% | matrix cores idle ~87% |
| VALUBusy | 26% | not dequant-throughput-bound |
| MemUnitStalled | 0.6% | NOT HBM-bandwidth-bound |
| OccupancyPercent | 37% | ~3 waves; VGPR 64 / LDS 32.5 KB allow ~8 |

Verdict: **latency/dependency-bound, under-pipelined.** Not BW-, compute-roofline-, or
dequant-throughput-bound. The per-K-tile `s_barrier` (LDS-A store->read sync) serializes
the load->fp4 dequant/scale->MFMA->cshuffle->write chain at low effective occupancy.
`grid` is full and VGPR/LDS are not the cap, so it is not a resource/grid wall.

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

## Rewrite plan (grouped-K to amortize the barrier)

Load G K-tiles into G distinct LDS buffers -> 1 barrier -> compute G tiles streaming B, repeat.
Barriers drop from ~num_k_tiles to ~num_k_tiles/G.
- Raise `lds_x_bytes` @3090 to `G * tile_m * lds_stride * a_elem_bytes` (G=4 -> LDS ~64 KB, still <=2 WG/CU on 160 KB).
- Restructure the 4327 loop; carry the extra state per tile: `a0/a1_prefetch`, `a_scale/b_scale`, `b_hi_loader`, packed K/N.
- Keep numerics: fp4 dequant + per-32 e8m0 scale layout is fixed; tile_k for fp4 is limited to {128, 256} (registry).

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
