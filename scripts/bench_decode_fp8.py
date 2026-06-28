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
    sparse_attn_v4_paged_decode,
)
from sglang.srt.layers.attention.dsv4.unified_kv_kernels.runtime import (
    pack_mxfp8_dense,
    pack_mxfp8_rope8_dense,
)

_D = mxfp8.DIM_HEAD  # 512


def _build_inputs(T: int, H: int, kv_len: int, device: str, seed: int = 0,
                  layout: str = "split"):
    """Build a paged decode workload: T tokens, each attending to kv_len slots
    (page_size=1, distinct slots), plus the bf16 and MXFP8 views of one shared
    latent so both pools represent identical data.

    layout="split"    : the production two-buffer pool ([pages,512] fp8 NoPE +
                        separate [pages,64] bf16 RoPE).
    layout="combined" : NoPE pack + RoPE co-located in one contiguous
                        [pages,640]-byte row; NoPE/RoPE are passed as strided
                        views (stride 640/320), so each slot's data lives in one
                        cache-line region. The decode kernel already honours
                        rope_stride_n, so this needs no kernel change."""
    g = torch.Generator(device=device).manual_seed(seed)
    pages = T * kv_len
    # Shared latent (NoPE 448 + RoPE 64). Small magnitude keeps fp8 in range.
    latent = (
        torch.randn(pages, _D, generator=g, device=device, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)

    nope_fp8, rope_bf16 = pack_mxfp8_dense(latent)  # [pages,512] fp8, [pages,64] bf16

    if layout == "fp8rope":
        # All-fp8 co-located layout: kv_fp8 [pages,512] (NoPE 0:448 + RoPE
        # 448:512), scale [pages,16] uint8 E8M0. The scale buffer is passed via
        # the unified_kv_rope slot; the kernel engages the fused 2-gather read
        # when SGLANG_UNIFIED_KV_FP8_ROPE8=1. Requires the fp8 backend gates set.
        kv_fp8, scale = pack_mxfp8_rope8_dense(latent)
        nope_fp8 = kv_fp8
        rope_bf16 = scale  # [pages,16] uint8 (carried in the rope slot)
        assert nope_fp8.shape == (pages, 512) and rope_bf16.shape == (pages, 16)

    if layout == "combined":
        # One contiguous [pages, 512 + 128] byte buffer: [0:512] NoPE pack,
        # [512:640] RoPE bf16. Strided views keep dtype/shape but co-locate rows.
        combined = torch.empty(pages, 512 + 2 * mxfp8.DIM_ROPE, dtype=torch.uint8,
                               device=device)
        combined[:, :512] = nope_fp8.view(torch.uint8)
        combined[:, 512:] = rope_bf16.view(torch.uint8)
        nope_fp8 = combined[:, :512].view(mxfp8.FP8_DTYPE)
        rope_bf16 = combined[:, 512:].view(torch.bfloat16)
        assert nope_fp8.shape == (pages, 512) and rope_bf16.shape == (pages, mxfp8.DIM_ROPE)
        assert nope_fp8.stride(0) == 640 and rope_bf16.stride(0) == 320

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
    return sparse_attn_v4_paged_decode(
        inp["q"],
        inp["nope_fp8"],
        inp["kv_indices"],
        inp["kv_indptr"],
        inp["attn_sink"],
        inp["softmax_scale"],
        unified_kv_rope=inp["rope_bf16"],
    )


def _time_us(mode: str, inp: dict, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        _call(mode, inp)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _call(mode, inp)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)  # ms -> us
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bf16", "fp8", "all"], default="all")
    ap.add_argument("--batch", type=int, default=16, help="T (decode tokens)")
    ap.add_argument("--heads", type=int, default=16, help="H (q heads per rank)")
    ap.add_argument("--kv-len", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", choices=["split", "combined", "fp8rope"], default="split",
                    help="fp8 pool layout: split (two buffers), combined "
                         "(co-located NoPE+RoPE bf16 per row), or fp8rope "
                         "(all-fp8 co-located NoPE+RoPE + E8M0 scale, fused read)")
    ap.add_argument("--check", action="store_true",
                    help="also report accuracy of the fp8 output vs the bf16 reference")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required")
    device = "cuda"
    ablate = os.environ.get("SGLANG_UNIFIED_KV_FP8_ABLATE", "none")
    qk_native = os.environ.get("SGLANG_UNIFIED_KV_FP8_QK_NATIVE", "0")

    inp = _build_inputs(args.batch, args.heads, args.kv_len, device, args.seed,
                        layout=args.layout)
    arch = torch.cuda.get_device_properties(0).gcnArchName

    print(
        f"arch={arch} T={args.batch} H={args.heads} kv_len={args.kv_len} D={_D} "
        f"iters={args.iters} ablate={ablate} qk_native={qk_native} layout={args.layout}"
    )

    if args.check:
        out_ref = _call("bf16", inp).float()
        out_fp8 = _call("fp8", inp).float()
        cos = (out_ref.flatten() @ out_fp8.flatten()) / (
            out_ref.norm() * out_fp8.norm() + 1e-30
        )
        rel = (out_fp8 - out_ref).norm() / (out_ref.norm() + 1e-30)
        print(f"accuracy vs bf16: cosine={cos.item():.6f} rel_l2={rel.item():.6e}")

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
