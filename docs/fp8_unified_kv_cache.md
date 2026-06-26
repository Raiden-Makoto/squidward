# FP8 Unified-KV Cache for DeepSeek-V4 (gfx942 / gfx950)

End-to-end design notes for the fp8 KV cache: the storage format, the custom
write/read kernels, the aiter OPUS prefill variant, and the fusion/cleanup work
that pays down fp8's overhead. Opt-in via `SGLANG_UNIFIED_KV_FP8=1`; the bf16
path is the untouched default.

## TL;DR

Upstream gives you a **bf16** unified-KV cache. Making it fp8 requires five things:

1. A **storage format** every kernel agrees on (fp8 dtype + per-1×64-group fp32 scales, arch-gated).
2. A **write path** that quantizes-on-store (in-kernel amax → scale → cast → scatter).
3. A native **decode read** path (fp8-MFMA, no bf16 materialization).
4. A native **prefill read** path (the aiter OPUS fp8 variant, no bf16 materialization).
5. **Fusion / cleanup** to claw back the store-side overhead, because the store
   path — not attention — is where fp8's cost actually lands.

The recurring design rule: **never materialize a bf16 copy of the fp8 data.**
Every read kernel dequantizes on-read into registers right before the MFMA. A
bf16 materialization both throws away the ~2× KV-capacity win (the entire point
of fp8) and is VALU-bound, which is what made the early approaches 2.4–4.3×
slower than bf16.

---

## 1. Storage format / contract

**File:** `unified_kv_kernels/paged_decode.py` (`_FP8_DTYPE`, `_FP8_GROUP_SIZE`)

The cache stores fp8 values with a **per-1×64-group fp32 block scale**
(`dequant: bf16 = fp8 * scale[page, d // 64]`). Only the **prefix**
(`unified_kv`) region is fp8; the **extend** (`kv`) region stays bf16/fp16.

Arch-gating (the two AMD archs use different fp8 encodings):

| Arch | GPU | fp8 dtype |
|---|---|---|
| gfx942 (CDNA3) | MI300X / MI325X | `e4m3fnuz` (fnuz max ≈ 224) |
| gfx950 (CDNA4) | MI350X / MI355X | `e4m3fn` (OCP, max = 448) |

Nothing upstream defines a unified-KV pool in fp8 with this scale granularity,
so the layout is ours. It is the single contract that the write kernels, the
decode kernel, and the aiter OPUS prefill kernel all read and write identically.

> **What is and isn't fp8 (read this to avoid the obvious confusion):**
>
> - **Feature on vs off.** With `SGLANG_UNIFIED_KV_FP8=0` (default) you get the
>   original upstream **bf16** cache, untouched. With `=1` the **persistent KV
>   cache is stored as fp8** — that is the 2× capacity win and it is already
>   implemented, not pending. "bf16 is the default" refers only to the
>   *opt-out*, not to what fp8 mode uses.
> - **Prefix vs extend (both present in an attention call, only one is the cache):**
>   - **prefix (`unified_kv`)** = the long-lived paged KV cache that accumulates
>     over the sequence → **stored as fp8.** This dominates KV memory.
>   - **extend (`kv`)** = only the handful of *brand-new* tokens computed this
>     step, in scratch before they are written into the cache → transient bf16.
>     The write path quantizes them into the fp8 cache on store.
> - **Lifecycle:** new tokens computed in bf16 → **quantized to fp8 on store**
>   (`_quant_scatter_by_loc_kernel`) → read back **as fp8** by both the decode
>   and OPUS prefill kernels (dequant-on-read into registers, never
>   re-materialized as a bf16 cache). The stored cache is fp8 end to end; the
>   only bf16 in fp8 mode is the transient new-token vectors and the in-register
>   dequantized values right at the MFMA.

---

## 2. Custom write/read kernels (HIP-safe, authored by us)

Upstream's unified-KV path is **bf16-only** — there is no fp8 store/scatter and
no fp8 decode for this pool. Three custom kernels fill the gap.

### 2a. Write path — `unified_kv_kernels/runtime.py`

- `_quant_scatter_by_loc_kernel` — in-kernel **amax → scale → cast → scatter**
  into the paged cache, plus writes the group scale, all in one pass.
- `_swa_scatter_kernel` — the sliding-window (SWA) scatter.

Custom because the alternative (a torch quantize op followed by a separate
scatter/copy) would materialize an fp8 tensor and add kernel launches. Fusing
the quant into the scatter keeps it single-pass and HIP-safe.

### 2b. Decode read — `unified_kv_kernels/paged_decode.py`

`_paged_decode_fp8mfma_split_kernel` — generate-time attention reads the fp8
cache **directly via native fp8-MFMA, no bf16 materialization**. Split-K because
the per-token decode shape is skinny and needs K-dimension parallelism. This is
the same "dequant-on-read, don't materialize" philosophy that drove the prefill
decision (§3).

### 2c. Store call sites

- `layers/fused_qk_norm_rope_store.py` — the decode SWA store.
- `dsv4/compressor_v2.py` — the compressed-K store.

These are where the model actually invokes the write path; both were taught to
emit fp8 + scales instead of bf16.

---

## 3. FP8 variant of the aiter OPUS prefill kernel

**File:** `unified_kv_kernels/paged_prefill.py` — routes the fp8 unified-KV
prefill into aiter's hand-tuned `pa_sparse_prefill_opus` kernel
(`sparse_attn_v4_paged_prefill`), reading our 8-bit format directly. (The kernel
change lives in the **aiter** repo — submitted as ROCm/aiter#3815, which was
**closed without merge**. The sglang-side gfx942 fnuz fix, sgl-project/sglang#28455,
*was* merged — these are two separate PRs.)

We added H>32 head-blocking for DP attention, then collapsed it back to a single
OPUS call at any head count.

### Why we wrote fp8 OPUS natively instead of a shim

Three approaches were tried, in order:

1. **The shim** (dequant fp8→bf16 into a temp buffer, then call the stock bf16
   OPUS) — **abandoned.**
   - Had a **random/intermittent GPU memory-access fault** that only reproduced
     during E2E benchmark sweeps (not single requests, not GSM8K) and could not
     be pinned cleanly. The shim commits were reverted to net-zero code.
   - Materializes a full bf16 copy of the fp8 prefix → throws away the ~2×
     capacity/bandwidth win right before the dot.
   - Forces abandoning `async_load` (direct global→LDS DMA), which is the whole
     reason OPUS is fast. The bf16 cast is VALU-bound → measured **2.4–4.3×
     slower than bf16 OPUS**.

2. **Triton fp8-dequant prefill** (Phase E wiring) — kept only as a correctness
   fallback (e.g. odd head counts). Same fundamental cost (materializes a bf16
   tile before the MFMA) and Triton can't match aiter's hand-tuned ASM MFMA
   scheduling.

3. **Native fp8 in aiter OPUS** — **chosen.**
   - Store the fp8 bytes **directly in the prefix LDS tile** and **dequant-on-read
     into registers** immediately before the MFMA, compile-time gated via
     `IS_FP8_SRC` so the bf16 path isn't even instantiated when fp8 is off.
   - Keeps `async_load` DMA overlap *and* aiter's MFMA pipeline intact → dequant
     becomes a cheap per-tile register op instead of a materialized bf16 tile.
   - Prototype decode went from **2.4–3.8× slower → ~0.79–0.92× of bf16**, while
     keeping the 2× KV-capacity win.

One line: the shim was both **unstable** (intermittent OOB fault) and
**architecturally self-defeating** (materialize bf16 → lose DMA overlap and the
fp8 memory win), so native dequant-on-read in OPUS was the only path to
parity-or-better.

### OPUS fp8 vs the Triton fp8 fallback (measured)

The OPUS fp8 kernel replaces the Triton fp8-dequant fallback. Direct head-to-head
at **H=128** (the DP-attention case — DSV4 is TP-sharded, so each rank carries all
128 heads and production prefill actually runs at this head count):

| T / ctx | OPUS fp8 (new) | Triton fp8 (old fallback) | speedup |
|---|---|---|---|
| 512 / 8192 | 3,766 µs | 78,008 µs | **20.7×** |
| 2048 / 2048 | 4,173 µs | 78,269 µs | **18.8×** |

Correctness matched (`opus_fp8_err ≈ 1.5e-2`, consistent with the GSM8K 0.948 run;
head-blocking is per-head-independent → identical math). In the live profile this
Triton fp8 prefill was **52.6% of GPU time, ~16.3 ms/call, ~2.7× TTFT vs bf16** —
the gap that motivated the kernel.

Caveats on what the number means:

- **18–21× is the H=128 case**, where the Triton kernel is pathologically slow
  (~78 ms/call, no good tiling for that head count).
- **vs OPUS bf16 is a different axis.** At H≤32 OPUS fp8 is ~0.73–0.78× of OPUS
  bf16 (i.e. ~22–27% *faster*). At H=128 it is ~3× slower than bf16 OPUS (4×32-head
  launches vs one 128-head bf16 launch) — but still ~19× faster than the Triton
  fp8 it replaces.

So against the path it actually replaces (Triton), OPUS fp8 is ~18–21× faster at
H=128; against bf16 OPUS it is faster at small H and ~3× slower at H=128.

---

## 4. Fusion / cleanup — paying down the fp8 tax

fp8 is not free. Profiling showed the **compressed-K store** is the dominant
fp8-only cost (`_quant_scatter_by_loc_kernel` ~2.1 ms + its elementwise unroll),
and it scales with batch×context — which is exactly why the ITL regression grows
with concurrency.

- **`jit_kernel/dsv4/compress.py`** — fold the fp8 compressed-K quant **into**
  the existing `compress_norm_rope_store` JIT kernel. One kernel instead of
  compress+norm+rope+store followed by a separate quant pass — no extra global
  round-trip.
- **`unified_kv_kernels/runtime.py` (cleanup)** — strip redundant `.contiguous()`
  and int64 index casts from the decode write that were spawning extra kernel
  launches. Profiling attributed **~4 ms/decode window of pure GPU-idle
  plumbing** to those; removing them closed the c=2 ITL gap from ~4.2% → 1.4%.

The decode read (`_paged_decode`) was already optimized; the residual regression
that remains is the compressed-K store scaling with concurrency.

---

## 5. Results (gfx950, DeepSeek-V4)

Accuracy GSM8K = **0.955**. fp8 vs bf16 baseline:

| Metric | fp8 vs bf16 |
|---|---|
| Accuracy (GSM8K) | 0.955 |
| TTT | within 3.3% |
| E2E | < 3.6% slower |
| TTFT | < 2.8% worse |
| ITL | 1.7–5.8% worse (highest regression at c=64) |

The ITL regression scales with concurrency (compressed-K store cost grows with
batch×context); attention reads are at/near bf16 parity.

---

## File index

| Path | Role |
|---|---|
| `unified_kv_kernels/paged_decode.py` | fp8 format (`_FP8_DTYPE`, `_FP8_GROUP_SIZE`) + native fp8-MFMA decode read (`_paged_decode_fp8mfma_split_kernel`) |
| `unified_kv_kernels/runtime.py` | write path (`_quant_scatter_by_loc_kernel`, `_swa_scatter_kernel`); decode-write plumbing cleanup |
| `unified_kv_kernels/paged_prefill.py` | routes fp8 prefill into aiter OPUS (`sparse_attn_v4_paged_prefill`) |
| `layers/fused_qk_norm_rope_store.py` | decode SWA quant-on-store call site |
| `dsv4/compressor_v2.py` | compressed-K quant-on-store call site |
| `jit_kernel/dsv4/compress.py` | fused fp8 compressed-K store into `compress_norm_rope_store` |
| aiter `pa_sparse_prefill_opus` (ROCm/aiter#3815, **closed, not merged**) | fp8 variant of the OPUS prefill kernel (dequant-on-read) |
