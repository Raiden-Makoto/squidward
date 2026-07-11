# GLM-5.2 prefill — i1k/o1k conc64, TP4 — MI355X (MXFP4) vs B200 (NVFP4)

GPU-busy per prefill forward: MI355X 363 ms vs B200 331 ms → 0.91x. All kernel times are **ms per prefill forward pass** (both captured 4 forwards, graphs-off eager, dense-MHA at i1k, `--mem-fraction-static 0.85 --chunked-prefill-size 131072`). MI355X = single dense+tuned capture (`run_glm52.sh --profile` + `AITER_CONFIG_FMOE=utilities/glm_fp8fp4_tuned_fmoe.csv`, INT4 QuickReduce). B200 = fresh apples-to-apples capture (GLM-5.2-NVFP4, tp4 modelopt_fp4, kv fp8_e4m3, trtllm DSA, flashinfer_trtllm MoE, dense `fmhaSm100f...H256...Causal`).

| Section | MI355X kernel | MI355X ms | B200 kernel | B200 ms | B200/MI355X |
| --- | --- | ---: | --- | ---: | ---: |
| Attention | `ck_tile FmhaFwd` dense MHA (gfx950) | 27.0 | `fmhaSm100f ...H256...Causal` dense MHA | 9.9 | |
| Attention | `set_mla_kv_buffer` + concat/cast (KV-cache write) | 6.2 | `set_mla_kv_buffer` + `float8_copy` + concat `direct_copy` | 11.0 | |
| Attention | `fused_qk_rmsnorm` (q/k norm) | 4.3 | flashinfer `LayerNorm`+`RMSNorm` | 3.1 | |
| Attention | `RotaryEmbedding` (rope) | 1.3 | `fused_rope_kernel` | 1.1 | |
| Attention | DSA indexer (`fast_hadamard` + `indexer_k_quant`) | 0.4 | `fast_hadamard` + `fused_store_indexer_cache` | 1.1 | |
| **Attention subtotal** | | **39.2** | | **26.3** | **0.67x** |
| MoE gate-up GEMM (gemm1) | `ck_moe_mxgemm` stage1 | 31.1 | `bmm_E2m1` (swiGlu) | 32.1 | 1.03x |
| MoE down GEMM (gemm2, tuned `t64x256`) | `mfma_moe2_afp4_wfp4` | 41.4 | `bmm_Bfloat16_E2m1` | 26.2 | 0.63x |
| MoE combine/finalize | `moe_reduction_kernel` | 22.3 | `finalizeKernelVecLoad` (fuses routed-scale) | 15.8 | 0.71x |
| MoE act quant | `dynamic_per_group_scaled_quant` (mxfp4) | 3.9 | `NVFP4Quantize` | 2.6 | 0.67x |
| **MoE subtotal** | | **98.7** | | **76.7** | **0.78x** |
| Dense GEMM | Tensile `MT256x256` (`unquant apply`) | 41.3 | `nvjet_sm100 128x256` | ≈40 | |
| Dense GEMM | Tensile `MT224x256` | 23.2 | `nvjet_sm100 176x128` | ≈22 | |
| Dense GEMM | `aiter::bf16gemm_256x256` (q_b_proj ‡) | 11.6 | `nvjet_sm100 256x128` | ≈12 | |
| Dense GEMM | `hgemm_bf16_128x128` (router gemm) | 4.1 | — | — | |
| **Dense GEMM subtotal** | | **80.2** | | **73.7** | **0.92x** |
| All-reduce | `quickreduce::allreduce_prototype_twoshot` (INT4 CodecQ4) | 109.9 | `ncclDevKernel_AllReduce ...RING_LL` | 68.6 | |
| All-reduce | `aiter::cross_device_reduce_2stage` | 20.8 | `mnnvl twoshotAllreduce` + `rmsNormLamport` + one-shot | 70.9 | |
| **All-reduce subtotal** | | **130.7** | | **139.5** | **1.07x** |
| RMSNorm | `aiter::add_rmsnorm_quant` (norm+quant fused) | 13.9 | `fused_add_rmsnorm` (norm only; quant in MoE) | 14.7 | |
| **RMSNorm subtotal** | | **13.9** | | **14.7** | **1.06x** |
| **TOTAL prefill** | | **363** | | **331** | **0.91x** |

‡ `q_b_proj` (ColumnParallelLinear, M×2048→4096/rank) and `o_proj` (RowParallelLinear, M×4096→6144/rank) are the two 128-aligned dense projections that `SGLANG_DSA_FP8_PROJ_GEMM=1` (gfx950, default off) converts from bf16 to the aiter FP8 CK GEMM. `fused_qkv_a` (M×6144→2624, not 128-aligned) and dense-MLP (layers 0-2) stay bf16.

## Levers

- **Dense-attention fallback at short prefill context (IMPLEMENTED + validated, commit `3235a7d271`).** MI355X previously ran triton `_sparse_mla_fwd_split_dim` **unconditionally** (128.8 ms/forward, the #1 prefill kernel) because the dense-MHA fallback was hard-gated to NVIDIA SM90/SM100 in `dsa_backend.py`. We extended the `use_mha` device gate to gfx950 and routed `_forward_standard_mha` through aiter `flash_attn_varlen_func` (ck_tile dense FA). Threshold-gated by `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` (effective value = model `index_topk` = 2048); off = set 0. Dense triggers when `max_kv_len ≤ 2048` (short prefill); long context stays sparse-MLA.
  - **Result at i1k (GLM-5.2-MXFP4, MI355X TP4):** the ≈764 ms sparse attention stack collapses to dense FA ≈27 ms/forward (vs B200 dense ≈9.9). GSM8K parity (**0.955**, 200 ex). e2e median **TTFT drops 14-32%** vs the sparse path:

| conc | sparse TTFT (ms) | dense TTFT (ms) | Δ |
| --- | ---: | ---: | ---: |
| 4  | 277.3  | 196.9  | −29% |
| 8  | 467.6  | 366.8  | −22% |
| 16 | 872.9  | 593.9  | −32% |
| 32 | 1371.1 | 1183.6 | −14% |
| 64 | 1920.5 | 1640.6 | −15% |
- **Tuned MoE tile CSV (applied above via `AITER_CONFIG_FMOE=utilities/glm_fp8fp4_tuned_fmoe.csv`).** Tuned `mfma_moe2 t64x256` tile (vs default `t64x128`).
- **All-reduce is NOT the gap at conc64.** With the corrected apples-to-apples B200 capture, MI355X INT4 QuickReduce (130.7 ms) is actually **faster** than B200's mnnvl twoshot + rmsNormLamport all-reduce (139.5 ms) at i1k/conc64. (The earlier "B200 wins all-reduce" claim came from a mismatched 8192-workload reference; it does not hold here.) Combined all-reduce+RMSNorm: MI355X 144.6 vs B200 154.2 — MI355X ahead.
- **Dense GEMM is near-parity (0.92x), not 3-4x** — the earlier 3-4x was against the bogus old B200 column. FP8 lever (`SGLANG_DSA_FP8_PROJ_GEMM=1`, ‡) A/B was captured but that run showed an anomalous all-reduce (degraded run); needs a clean redo before drawing a conclusion.
- **Router GEMM is NOT an anomaly (microbench-verified).** MI355X runs the MoE gate as a standalone bf16 GEMM (`aiter_dsv3_router_gemm` → `tgemm.mm`, shape M×6144→256, ≈4.4 ms/forward summed over 75 MoE layers); B200 fuses the gate matmul into another GEMM (only the split-K reduction `cublasLt::splitKreduce` ~50 µs/fwd is itemized). A standalone bench of the exact router shape via the same `tgemm.mm` path hits **547 TF at M=8k up to 1196 TF (~peak) at M=65k**, on par with `torch.matmul`/hipblaslt — the `SPK1` pick is near-optimal, not wasteful. The 4.1 ms/forward is legitimate work (~4.7 TFLOP/fwd of gate projection at ~1150 TFLOPS), the same work B200 does fused. No lever here — an earlier "SPK1 is wrong / 5% roofline" note was incorrect on both counts (B200 fuses it, and the MI355X kernel is already near-peak).
- **Attention (0.67x, dense FA 27 vs 9.9): no viable fork lever found — fp8 batch-prefill is accuracy-neutral but net-negative e2e.** The bf16 ck_tile FA is occupancy-limited by the `[BM,256]` fp32 online-softmax accumulator in VGPRs (B200 holds it in TMEM → ~96% of ~2.25 PFLOPS bf16 vs MI355X ~14% util). Two fork-side experiments:
  - *Tiling sweep exhausted (no win):* swept `BLOCK_M/N/warps` on the gfx950 triton `extend_attention_fwd` (11 configs) — best (128/64/8) = 1.534 ms = **1.00x** vs ck_tile 1.530; larger `BLOCK_M` is strictly worse (256/64/8 = 3.66 ms).
  - *Register/occupancy diagnosis (rocprofv3 + roofline):* ck_tile dense FA = **256 VGPR, 0 AGPR, 17 KB LDS** → ~2 waves/SIMD (VGPR-limited), AGPR file idle. Analytic roofline at 65536 tok/1.53 ms = **~360 TFLOPS (14% of ~2.5 PFLOPS bf16) and ~3.5 TB/s (~44% of ~8 TB/s)** → latency/occupancy-bound, neither compute nor BW saturated.
  - *Occupancy lever tested and DEAD:* the triton FA launches `waves_per_eu=1`; swept 1→8 (env-gated `SGLANG_FA_WAVES_PER_EU`, commit `4811ac0906`): wpe=2 = 1.01x (flat), **wpe=3 = 0.64x, wpe=4 = 0.18x, wpe=8 = 0.07x** (`num_stages=2` always slower). Forcing higher occupancy spills the `[BM,256]` fp32 accumulator → 5–13x slower. So 256 VGPR/2-wave is the *optimal* point, not a missed one; the idle AGPRs can't be repurposed (compiler spills instead). The FA2 online-softmax genuinely needs the `[BM,256]` fp32 accumulator (~256 VGPR), capping occupancy — a hard register-pressure wall on CDNA4 that only TMEM (B200) escapes. Same head_dim-256 wall for both AMD FA kernels; a fork-side win would require an algorithmic accumulator reduction (bf16 O-accum or split-D 2-pass), both compute/accuracy trade-offs, not a config/occupancy knob.
  - *fp8 batch-prefill (IMPLEMENTED, env-gated `SGLANG_DSA_FP8_DENSE_ATTN`, default off, commits `c410a22111`/`9c966c21a0`/`6cea8561a2`): accuracy-neutral, net-negative e2e.* `aiter.mha_batch_prefill_func` takes fp8 q/k/v + descales; page_size=16 is the only D=256-supported variant (32/64/128/256 → `no matching kernel`). GLM-5.2's `kv_b_proj` is bf16 (quark `exclude`), so the fused `fused_gemm_afp4wfp4_split_cat` fp8-K/V path (`forward_mha.py:299`) never fires and `_forward_standard_mha_fp8` must quantize the decompressed bf16 q/k/v to fp8 itself each forward.
    - **The FA *kernel* is genuinely faster (isolated bench, all batch sizes):** ck_tile bf16 vs mha_batch_prefill fp8 (page-16, fp8 inputs, aligned view): B8 0.216→0.154 (1.40x), B16 0.414→0.301 (1.38x), B32 0.781→0.635 (1.23x), B64 1.524→1.329 (1.15x), B128 3.008→2.671 (1.13x). So the kernel is **not** the problem — it's the surrounding per-forward work.
    - **e2e TTFT A/B (i1k/o1k, MI355X TP4)** across three implementations (scatter → scatter-free+fused-amax → per-forward-paging-cache+fused `scaled_fp8_quant`):

| conc | bf16 | fp8 v1 (scatter) | fp8 v2 (scatter-free) | fp8 v3 (cache+fused-quant) | Δ v3 vs bf16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4  | 201.7 | 228.0 | 227.1 | 213.6 | +6% |
| 8  | 264.7 | 511.3 | 452.5 | 377.1 | +42% |
| 16 | 478.8 | 638.3 | 645.7 | 597.0 | +25% |
| 32 | 798.2 | 1135.1 | 1126.3 | **OOM** | — |
| 64 | 1186.8 | 2130.8 | 2061.7 | **OOM** | — |

    - **Verdict: dead lever for this checkpoint; kept default-off.** The overhead fixes (per-forward paging cache to kill ~78 redundant `.item()` syncs/forward, commit `9c966c21a0`; fused `scaled_fp8_quant` to drop the bf16 temp, commit `6cea8561a2`) closed a lot of the gap (conc8 452→377) but fp8 stays **+25–42%** at conc ≤16 and **OOMs at conc32+** — even with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the extra per-forward q/k/v fp8 buffers don't fit under the required `--mem-fraction-static 0.85` (bf16 passes q/k/v straight to FA with zero copies). Root cause is irreducible on this path: q/k/v are **activations**, so they must be re-quantized **every forward** (78 layers × 3 tensors), and that memory-bound quant cost exceeds the ~0.2 ms/layer FA-kernel win. The only way to avoid the per-forward quant is fp8/paged K/V straight from the decompression GEMM (`fused_gemm_afp4wfp4_split_cat`), which needs MXFP4 `kv_b_proj` — structurally excluded for GLM (decode absorb + integration faults, see `SGLANG_FORCE_MXFP4_KVB` experiment). Accuracy was never the issue (1319-q GSM8K fp8 0.928 vs bf16 0.929, parity). B200 side `fmhaSm100f...` is a trtllm-gen precompiled cubin (no portable source).
- **MoE (0.78x):** within MoE the gate-up GEMM is at parity (31.1 vs 32.1).
  - **Down GEMM (`mfma_moe2 t64x256`, 41.4 vs 26.2): ceiling-limited, not a config lever.** It is *already* the tuned tile and near the CDNA4 mxfp4 roofline — closing it needs a **kernel rewrite** (CK/stream-K, vendor/upstream), not re-tuning. Not our lane.
  - **Expert combine (`moe_reduction` 22.3): scrutinized — at parity, near roofline, not a fork lever.** It's a plain sum `[M,9,6144]→[M,6144]` (routed-scale is applied in the stage2 GEMM, not here; the shared expert is top-9 slot #9, so 22.3 ms already includes it). Fair combine-only comparison: MI355X 22.3 vs B200 `finalizeKernelVecLoad` 15.8 + routed-scale/shared-add `CUDAFunctor_add` 5.4 = 21.2 → **near parity** (the raw 0.71x row omits B200's shared-add). Traffic (`M×9×6144` read + `M×6144` write) sits near the gfx950 HBM3e roofline at the captured M; the two-pass cost (stage2 writes `[M,9,6144]`, reduction re-reads it) is inherent to reduce-mode. Kernel is aiter FlyDSL-JIT (not fork-modifiable); only lever is aiter-side — a stage2 `atomic_persist` combine (drops the separate reduction) or folding the sum into the stage2 epilogue upstream. Not our lane.
  - all-reduce / dense-GEMM / RMSNorm / router are at/above parity.
