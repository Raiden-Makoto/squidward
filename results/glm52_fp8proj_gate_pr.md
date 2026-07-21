# PR draft — [AMD] [GLM5] Opt-in prefill-only FP8 dense projection GEMM for MLA q_b/o_proj (gfx950)

- PR: sgl-project/sglang#31955 (draft)
- Head: `Raiden-Makoto:RM/fp8-proj-gemm-prefill-gate`
- Commit: `47060060ea` (3 files, +130)
- Dependency: ROCm/aiter#4243 (tuned GLM-5.2 `a8w8_blockscale_bpreshuffle` configs) — MERGED

## Title

[AMD] [GLM5] Opt-in prefill-only FP8 dense projection GEMM for MLA q_b/o_proj (gfx950)

## Motivation

On gfx950 (MI355X), the GLM-5.2 MLA dense projections `q_b_proj` / `o_proj` are quark-excluded and run in bf16. For prefill (large M) these are dense-GEMM bound. Running them on the aiter FP8 `a8w8_blockscale_bpreshuffle` CK GEMM for prefill only — while decode (small-M cuda-graph batch sizes) stays bf16 — reduces the dense-GEMM section without regressing decode.

Opt-in via `SGLANG_DSA_FP8_PROJ_GEMM` (default off), gfx950-gated. No effect off-arch or with the flag unset.

Depends on the tuned `a8w8_blockscale_bpreshuffle` GLM-5.2 configs in aiter (ROCm/aiter#4243); with those present aiter selects the tuned tiles automatically — no config override needed. Untuned, the small-M FP8 GEMM regresses vs bf16, which is why the path is prefill-only (M > 512).

## Modifications

- `layers/quantization/unquant.py`: `UnquantizedLinearMethod` gains an opt-in FP8 path for layers tagged `_fp8_proj_gemm`. At load, the bf16 weight is repacked into a private FP8 e4m3 (+128×128 UE8M0 scale, bpreshuffle) copy; `layer.weight` stays bf16. `apply()` gates on token count: M > 512 → FP8 CK GEMM (`aiter_w8a8_block_fp8_linear`), M ≤ 512 → bf16.
- `models/deepseek_v2.py`: mark `q_b_proj`/`o_proj` with `_fp8_proj_gemm` (128-aligned; `fused_qkv_a` out=2624 is not 128-aligned, `kv_b_proj` is the absorbed-bmm path).
- `layers/quantization/fp8_utils.py`: add `quant_weight_ue8m0()` helper.



## Accuracy Tests

GSM8K (400 questions, 5-shot), `SGLANG_DSA_FP8_PROJ_GEMM=1`, TP4:


| config                     | accuracy |
| -------------------------- | -------- |
| prefill-only FP8 proj gate | 0.943    |


Pass bar ≥ 0.92.

## Speed Benchmarks

`sglang.bench_serving` (random, input=1024, output=1024, num-prompts=conc×4), TP4 MI355X, graphs-on. All values averaged over 3 reps after dropping the obvious outlier (for example 255.6, 255.8, 256.7, 399.1 --> 399.1 gets dropped) to account for noise and bad rng.

Baseline = flag off (bf16); Feature = `SGLANG_DSA_FP8_PROJ_GEMM=1`.

A. Baseline (flag off)*:


| concurrency | TTFT (ms) | ITL (ms) | E2EL (ms) | output tok/s |
| ----------- | --------- | -------- | --------- | ------------ |
| 4           | 199.0     | 11.81    | 12293     | 333.1        |
| 8           | 257.1     | 13.70    | 14312     | 572.0        |
| 16          | 525.0     | 16.40    | 17338     | 942.1        |
| 32          | 755.4     | 20.35    | 21900     | 1498.0       |
| 64          | 1176.8    | 27.08    | 29723     | 2205.8       |


B. Feature on (Δ vs baseline):


| concurrency | TTFT (ms) | Δ     | ITL (ms) | Δ     | E2EL (ms) | Δ     | output tok/s | Δ     |
| ----------- | --------- | ----- | -------- | ----- | --------- | ----- | ------------ | ----- |
| 4           | 197.2     | −0.9% | 11.81    | −0.0% | 12283     | −0.1% | 332.9        | −0.1% |
| 8           | 255.4     | −0.6% | 13.69    | −0.0% | 14310     | −0.0% | 572.2        | +0.0% |
| 16          | 498.8     | −5.0% | 16.39    | −0.0% | 17291     | −0.3% | 946.9        | +0.5% |
| 32          | 716.0     | −5.2% | 20.25    | −0.5% | 21781     | −0.5% | 1505.6       | +0.5% |
| 64          | 1110.5    | −5.6% | 27.05    | −0.1% | 29630     | −0.3% | 2210.5       | +0.2% |


TTFT improves monotonically with concurrency (−0.9% → −5.6%) as more prefill GEMM amortizes the FP8 path; ITL, E2EL, and output-throughput deltas are within run-to-run noise (decode stays bf16).

*Baseline and Feature both measured with ++[#30519](https://github.com/sgl-project/sglang/pull/30519)++, ++[#30715](https://github.com/sgl-project/sglang/pull/30715)++, ++[#31323](https://github.com/sgl-project/sglang/pull/31323)++, ++[#31324](https://github.com/sgl-project/sglang/pull/31324)++ and aiter tuned MoE configs for GLM5.2; we expect those will be merged first and aiter version updated in recent images.

## Checklist

- [x] Format your code according to the Contributor Guide (pre-commit).
- [x] Add unit tests as outlined in the Contributor Guide. (n/a — opt-in, arch-gated inference path; covered by GSM8K + bench above)
- [x] Update documentation as needed.



## Review and Merge Process

Requires ROCm/aiter#4243 (tuned GLM-5.2 `a8w8_blockscale_bpreshuffle` configs) for the tuned FP8 tiles.