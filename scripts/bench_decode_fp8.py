#!/usr/bin/env python3
"""Isolated microbenchmark for the DSv4 unified-KV paged-decode attention kernel.

Times ``sparse_attn_v4_paged_decode`` for the bf16 pool vs the MXFP8 (fp8 NoPE +
bf16 RoPE) pool on representative decode shapes, to attribute the fp8-vs-bf16 ITL
gap to its source (dequant ALU vs split fp8/E8M0/RoPE gathers vs the dots).

Run one variant per process so rocprof captures a clean kernel set, e.g.:

    PYTHONPATH=/sgl-workspace/squidward/python python3 scripts/bench_decode_fp8.py \
        --mode fp8 --batch 16 --heads 16 --kv-len 8192 --iters 200

Ablations are selected via the SGLANG_UNIFIED_KV_FP8_ABLATE env var, read by the
kernel wrapper (none|noscale|noropegather). ``--mode all`` runs every variant in
one process and prints a table (no rocprof).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

from sglang.srt.layers.attention.dsv4.unified_kv_kernels import mxfp8
from sglang.srt.layers.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
    sparse_attn_v4_paged_decode,
)
from sglang.srt.layers.attention.dsv4.unified_kv_kernels.runtime import (
    pack_mxfp8_dense,
)

_D = mxfp8.DIM_HEAD  # 512


def _build_inputs(T: int, H: int, kv_len: int, device: str, seed: int = 0):
    """Build a paged decode workload: T tokens, each attending to kv_len slots
    (page_size=1, distinct slots), plus the bf16 and MXFP8 views of one shared
    latent so both pools represent identical data."""
    g = torch.Generator(device=device).manual_seed(seed)
    pages = T * kv_len
    # Shared latent (NoPE 448 + RoPE 64). Small magnitude keeps fp8 in range.
    latent = (
        torch.randn(pages, _D, generator=g, device=device, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)

    nope_fp8, rope_bf16 = pack_mxfp8_dense(latent)  # [pages,512] fp8, [pages,64] bf16

    q = (torch.randn(T, H, _D, generator=g, device=device, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    attn_sink = torch.randn(H, generator=g, device=device, dtype=torch.float32)

    kv_indices = torch.arange(pages, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(0, pages + 1, kv_len, dtype=torch.int32, device=device)
    assert kv_indptr.numel() == T + 1

    softmax_scale = 1.0 / (_D ** 0.5)
    return {
        "q": q,
        "latent_bf16": latent,
        "nope_fp8": nope_fp8,
        "rope_bf16": rope_bf16,
        "kv_indices": kv_indices,
        "kv_indptr": kv_indptr,
        "attn_sink": attn_sink,
        "softmax_scale": softmax_scale,
    }


_FP8_CFG: dict = {}  # optional forced escape hatches for fp8 (profiling)


def _call(mode: str, inp: dict) -> torch.Tensor:
    if mode == "bf16":
        return sparse_attn_v4_paged_decode(
            inp["q"],
            inp["latent_bf16"],
            inp["kv_indices"],
            inp["kv_indptr"],
            inp["attn_sink"],
            inp["softmax_scale"],
            unified_kv_rope=None,
        )
    if _FP8_CFG:
        return _call_cfg(inp, **_FP8_CFG)
    return sparse_attn_v4_paged_decode(
        inp["q"],
        inp["nope_fp8"],
        inp["kv_indices"],
        inp["kv_indptr"],
        inp["attn_sink"],
        inp["softmax_scale"],
        unified_kv_rope=inp["rope_bf16"],
    )


def _call_cfg(inp: dict, **cfg) -> torch.Tensor:
    """fp8 decode via the internal wrapper with explicit tuning escape hatches
    (block_k / num_k_stages / kv_splits / block_h)."""
    return _sparse_attn_v4_paged_decode_triton(
        inp["q"],
        inp["nope_fp8"],
        inp["kv_indices"],
        inp["kv_indptr"],
        inp["attn_sink"],
        inp["softmax_scale"],
        unified_kv_rope=inp["rope_bf16"],
        **{k: v for k, v in cfg.items() if v is not None},
    )


def _time_fn(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)  # ms -> us
    return statistics.median(samples)


def _time_us(mode: str, inp: dict, iters: int, warmup: int) -> float:
    return _time_fn(lambda: _call(mode, inp), iters, warmup)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.to(torch.float32).reshape(-1)
    b32 = b.to(torch.float32).reshape(-1)
    return float(
        torch.dot(a32, b32) / (a32.norm() * b32.norm()).clamp_min(1e-30)
    )


def _check(inp: dict) -> None:
    """Correctness gate: fp8 decode (with whatever fp8 path the env selects --
    QK/PV native or the bf16-reconstruction default) vs the bf16 pool reference.
    Reports cosine similarity + max abs error over the full output."""
    ref = _call("bf16", inp)
    got = _call("fp8", inp)
    cos = _cosine(got, ref)
    max_abs = float((got.to(torch.float32) - ref.to(torch.float32)).abs().max())
    rel = max_abs / float(ref.to(torch.float32).abs().max().clamp_min(1e-30))
    print(f"  cosine(fp8, bf16) = {cos:.6f}")
    print(f"  max_abs_err       = {max_abs:.6e}  (rel {rel:.4%})")


def _sweep(inp: dict, iters: int, warmup: int) -> None:
    """Grid over the fp8 decode tuning knobs to see if a config beats the
    'safe-to-fit' default (block_k=32, num_k_stages=2). Configs that overflow
    LDS / fail to compile are reported as FAIL."""
    bf16 = _time_us("bf16", inp, iters, warmup)
    fp8_default = _time_us("fp8", inp, iters, warmup)
    print(f"  bf16            : {bf16:8.2f} us")
    print(f"  fp8 (default)   : {fp8_default:8.2f} us  (delta {fp8_default-bf16:+.2f})")
    print("  --- fp8 sweep (block_k / stages / kv_splits / block_h) ---")
    block_ks = [16, 32, 64]
    stages = [2, 3]
    kv_splits_opts = [None, 8, 16]
    block_hs = [None, 32]
    best = (fp8_default, "default")
    for bk in block_ks:
        for st in stages:
            for ks in kv_splits_opts:
                for bh in block_hs:
                    cfg = dict(block_k=bk, num_k_stages=st, kv_splits=ks, block_h=bh)
                    tag = f"bk={bk} st={st} ks={ks or 'auto'} bh={bh or 'auto'}"
                    try:
                        us = _time_fn(lambda: _call_cfg(inp, **cfg), iters, warmup)
                    except Exception as e:  # noqa: BLE001
                        msg = str(e).splitlines()[0][:60]
                        print(f"  {tag:42s}:   FAIL ({msg})")
                        continue
                    mark = ""
                    if us < best[0]:
                        best = (us, tag)
                        mark = "  <-- best"
                    print(f"  {tag:42s}: {us:8.2f} us (delta {us-bf16:+.2f}){mark}")
    print(f"  === best fp8: {best[0]:.2f} us [{best[1]}] vs bf16 {bf16:.2f} "
          f"(gap {best[0]-bf16:+.2f}, default gap {fp8_default-bf16:+.2f}) ===")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["bf16", "fp8", "all", "sweep", "check"], default="all"
    )
    ap.add_argument("--batch", type=int, default=16, help="T (decode tokens)")
    ap.add_argument("--heads", type=int, default=16, help="H (q heads per rank)")
    ap.add_argument("--kv-len", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-k", type=int, default=None)
    ap.add_argument("--num-k-stages", type=int, default=None)
    ap.add_argument("--kv-splits", dest="kv_splits", type=int, default=None)
    ap.add_argument("--block-h", type=int, default=None)
    args = ap.parse_args()

    for k, v in (
        ("block_k", args.block_k),
        ("num_k_stages", args.num_k_stages),
        ("kv_splits", args.kv_splits),
        ("block_h", args.block_h),
    ):
        if v is not None:
            _FP8_CFG[k] = v

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required")
    device = "cuda"
    ablate = os.environ.get("SGLANG_UNIFIED_KV_FP8_ABLATE", "none")
    qk_native = os.environ.get("SGLANG_UNIFIED_KV_FP8_QK_NATIVE", "0")
    pv_native = os.environ.get("SGLANG_UNIFIED_KV_FP8_PV_NATIVE", "0")

    inp = _build_inputs(args.batch, args.heads, args.kv_len, device, args.seed)
    arch = torch.cuda.get_device_properties(0).gcnArchName

    print(
        f"arch={arch} T={args.batch} H={args.heads} kv_len={args.kv_len} D={_D} "
        f"iters={args.iters} ablate={ablate} qk_native={qk_native} "
        f"pv_native={pv_native}"
    )

    if args.mode == "check":
        _check(inp)
        return

    if args.mode == "sweep":
        _sweep(inp, args.iters, args.warmup)
        return

    modes = ["bf16", "fp8"] if args.mode == "all" else [args.mode]
    results = {}
    for mode in modes:
        us = _time_us(mode, inp, args.iters, args.warmup)
        results[mode] = us
        print(f"{mode:>5}: {us:8.2f} us/call")

    if "bf16" in results and "fp8" in results:
        d = results["fp8"] - results["bf16"]
        pct = 100.0 * d / results["bf16"]
        print(f"delta(fp8-bf16): {d:+.2f} us ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
