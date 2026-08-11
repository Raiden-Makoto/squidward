"""Renorm cost as a function of distribution sharpness.

A single batch shape hides the thing that actually drives renorm cost. The
pivot search has a fast path bounded by a 1024-entry prefix and a full-sort
fallback when the nucleus does not fit, so the same code is either flat or
dominated by torch.sort depending only on how peaked the rows are. FlashInfer's
AOT kernel has the opposite sensitivity: its ternary pivot search needs more
rounds as a row gets peakier.

Sweeping a logit scale exposes both curves. Reporting the overflow fraction
alongside them is what makes two runs comparable: softmax(randn) over a 100K
vocabulary is nearly uniform and overflows every row, which is not a regime any
real model produces.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.sampling.renorm import (
    _TOP_P_PREFIX,
    top_p_pivots,
    top_k_renorm_probs_torch,
    top_p_renorm_probs_torch,
)
from sglang.kernels.ops.sampling.renorm_triton import (
    _BLOCK_SIZE,
    _masked_row_sum_kernel,
    _masked_scale_kernel,
    apply_pivot_triton,
    top_k_renorm_probs_triton,
    top_p_renorm_probs_triton,
)

DEV = torch.device("cuda")


def load_aot():
    """Probe with a real call: the wrapper imports fine when the op is absent.
    Both spellings are tried because the wheel lags the in-tree rename."""
    for k_name, p_name in (
        ("top_k_renorm_probs", "top_p_renorm_probs"),
        ("top_k_renorm_prob", "top_p_renorm_prob"),
    ):
        try:
            import sgl_kernel

            fn_k = getattr(sgl_kernel, k_name)
            fn_p = getattr(sgl_kernel, p_name)
            probs = torch.softmax(torch.randn(2, 128, device=DEV), dim=-1)
            fn_k(probs, torch.full((2,), 8, dtype=torch.int64, device=DEV))
            fn_p(probs, torch.full((2,), 0.9, device=DEV))
            return fn_k, fn_p
        except Exception:
            continue
    return None


def timeit(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def latency_samples(fn, iters: int) -> list[float]:
    for _ in range(3):
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
        samples.append(start.elapsed_time(end))
    return samples


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def load_capture_rows(capture_dir: Path):
    softmax = []
    probs = []
    expected = []
    top_ps = []
    top_ks = []
    for path in sorted(capture_dir.glob("rank0_*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        num_draft = record["metadata"]["num_draft_tokens"]
        softmax.append(record["softmax_probs"].flatten(0, 1))
        probs.append(record["top_p_input_probs"].flatten(0, 1))
        expected.append(record["renormalized_probs"].flatten(0, 1))
        top_ps.append(record["top_ps"].reshape(-1).repeat_interleave(num_draft))
        top_ks.append(record["top_ks"].reshape(-1).repeat_interleave(num_draft))
    if not probs:
        raise ValueError(f"No rank0_*.pt captures found under {capture_dir}")
    return tuple(torch.cat(x) for x in (softmax, probs, expected, top_ps, top_ks))


def run_capture_bench(args, aot) -> None:
    captured_softmax, captured, captured_expected, captured_top_ps, captured_top_ks = (
        load_capture_rows(args.capture_dir)
    )
    output = {
        "device": torch.cuda.get_device_name(0),
        "capture_dir": str(args.capture_dir),
        "captured_rows": captured.shape[0],
        "vocab_size": captured.shape[1],
        "num_draft_tokens": args.num_draft,
        "rows": [],
    }
    print(
        f"{output['device']}  capture_rows={captured.shape[0]}  "
        f"vocab={captured.shape[1]}"
    )
    print(
        f"{'bs':>5} {'rows':>6} {'nuc_p50':>8} {'nuc_max':>8} {'ovf':>5} "
        f"{'pivot':>8} {'sum':>8} {'scale':>8} {'apply':>8} "
        f"{'full_p50':>9} {'full_p95':>9}"
    )

    for batch_size in args.batches:
        rows = batch_size * args.num_draft
        selected = torch.arange(rows) % captured.shape[0]
        softmax_probs = captured_softmax[selected].to(DEV)
        probs = captured[selected].to(DEV)
        expected = captured_expected[selected].to(DEV)
        top_ps = captured_top_ps[selected].to(DEV)
        top_ks = captured_top_ks[selected].to(DEV)
        values = probs.sort(dim=-1, descending=True).values
        budgets = probs.sum(dim=-1) - (1.0 - top_ps)
        nucleus = ((values.cumsum(dim=-1) - values) <= budgets[:, None]).sum(dim=-1)
        overflow_rate = float((nucleus > _TOP_P_PREFIX).float().mean())

        pivots = top_p_pivots(probs, top_ps)
        num_chunks = triton.cdiv(probs.shape[1], _BLOCK_SIZE)
        grid = (rows, num_chunks)
        partial = torch.empty((rows, num_chunks), device=DEV, dtype=torch.float32)
        row_sums = torch.empty(rows, device=DEV, dtype=torch.float32)
        out = torch.empty_like(probs)

        pivot_samples = latency_samples(
            lambda: top_p_pivots(probs, top_ps), args.iters
        )
        sum_samples = latency_samples(
            lambda: _masked_row_sum_kernel[grid](
                probs,
                pivots,
                partial,
                probs.shape[1],
                num_chunks,
                BLOCK_SIZE=_BLOCK_SIZE,
            ),
            args.iters,
        )
        torch.sum(partial, dim=1, out=row_sums)
        scale_samples = latency_samples(
            lambda: _masked_scale_kernel[grid](
                probs,
                pivots,
                row_sums,
                out,
                probs.shape[1],
                BLOCK_SIZE=_BLOCK_SIZE,
            ),
            args.iters,
        )
        apply_samples = latency_samples(
            lambda: apply_pivot_triton(probs, pivots), args.iters
        )
        full_samples = latency_samples(
            lambda: top_p_renorm_probs_triton(probs, top_ps), args.iters
        )
        got = top_p_renorm_probs_triton(probs, top_ps)
        torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)

        row = {
            "batch_size": batch_size,
            "rows": rows,
            "max_token_probability_median": float(probs.max(dim=-1).values.median()),
            "nucleus_p50": int(nucleus.median()),
            "nucleus_max": int(nucleus.max()),
            "overflow_rate": overflow_rate,
            "top_p_pivot_p50_ms": statistics.median(pivot_samples),
            "masked_row_sum_p50_ms": statistics.median(sum_samples),
            "masked_scale_p50_ms": statistics.median(scale_samples),
            "apply_pivot_p50_ms": statistics.median(apply_samples),
            "full_top_p_p50_ms": statistics.median(full_samples),
            "full_top_p_p95_ms": percentile(full_samples, 0.95),
        }
        top_k_is_active = bool((top_ks < probs.shape[1]).any())
        if top_k_is_active:
            top_k_samples = latency_samples(
                lambda: top_k_renorm_probs_triton(softmax_probs, top_ks), args.iters
            )
            row["full_top_k_p50_ms"] = statistics.median(top_k_samples)
            torch.testing.assert_close(
                top_k_renorm_probs_triton(softmax_probs, top_ks),
                probs,
                rtol=2e-5,
                atol=2e-6,
            )
        if aot is not None:
            aot_samples = latency_samples(lambda: aot[1](probs, top_ps), args.iters)
            row["aot_top_p_p50_ms"] = statistics.median(aot_samples)
        output["rows"].append(row)
        print(
            f"{batch_size:>5} {rows:>6} {row['nucleus_p50']:>8} "
            f"{row['nucleus_max']:>8} {overflow_rate:>5.0%} "
            f"{row['top_p_pivot_p50_ms']:>8.3f} "
            f"{row['masked_row_sum_p50_ms']:>8.3f} "
            f"{row['masked_scale_p50_ms']:>8.3f} "
            f"{row['apply_pivot_p50_ms']:>8.3f} "
            f"{row['full_top_p_p50_ms']:>9.3f} "
            f"{row['full_top_p_p95_ms']:>9.3f}"
        )

    if args.output_json:
        args.output_json.write_text(json.dumps(output, indent=2))


def describe(probs: torch.Tensor, top_p: float, sample_rows: int = 64):
    """Mean top-1 mass, median nucleus size, and the fraction of rows whose
    nucleus exceeds the prefix, which is what triggers the sort fallback."""
    sub = probs[:sample_rows]
    values, _ = torch.sort(sub, dim=-1, descending=True)
    cumsum = values.cumsum(dim=-1)
    nucleus = (cumsum < top_p).sum(dim=-1) + 1
    return (
        float(values[:, 0].mean()),
        int(nucleus.median()),
        float((nucleus > _TOP_P_PREFIX).float().mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=151936, help="GLM-5.2 vocabulary")
    parser.add_argument("--rows", type=int, default=1536, help="bs 256 x 6 draft tokens")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--num-draft", type=int, default=6)
    parser.add_argument(
        "--batches", type=int, nargs="+", default=(1, 2, 4, 8, 32, 128, 256)
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0),
        help="logit scale; 1.0 is the unscaled softmax(randn) that overflows every row",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    aot = load_aot()
    if args.capture_dir:
        run_capture_bench(args, aot)
        return
    name = torch.cuda.get_device_name(0)
    print(f"{name}  vocab={args.vocab}  rows={args.rows}  top_p={args.top_p}  top_k={args.top_k}")
    print(f"AOT renorm kernel: {'available' if aot else 'unavailable (expected on ROCm)'}")
    print(f"prefix={_TOP_P_PREFIX} (nucleus above this falls back to a full sort)\n")

    header = (
        f"{'scale':>6} {'top1':>7} {'nucleus':>8} {'ovf':>6} "
        f"{'k_torch':>8} {'k_triton':>9} {'p_torch':>8} {'p_triton':>9}"
    )
    if aot is not None:
        header += f" {'k_aot':>7} {'p_aot':>7}"
    print(header)

    for scale in args.scales:
        probs = torch.softmax(
            torch.randn(args.rows, args.vocab, device=DEV) * scale, dim=-1
        )
        top1, nucleus, ovf = describe(probs, args.top_p)
        top_ks = torch.full((args.rows,), args.top_k, dtype=torch.int64, device=DEV)
        top_ps = torch.full((args.rows,), args.top_p, dtype=torch.float32, device=DEV)

        kt = timeit(lambda: top_k_renorm_probs_torch(probs, top_ks), args.iters)
        kr = timeit(lambda: top_k_renorm_probs_triton(probs, top_ks), args.iters)
        pt = timeit(lambda: top_p_renorm_probs_torch(probs, top_ps), args.iters)
        pr = timeit(lambda: top_p_renorm_probs_triton(probs, top_ps), args.iters)

        row = (
            f"{scale:>6.1f} {top1:>7.3f} {nucleus:>8} {ovf:>6.0%} "
            f"{kt:>8.3f} {kr:>9.3f} {pt:>8.3f} {pr:>9.3f}"
        )
        if aot is not None:
            ka = timeit(lambda: aot[0](probs, top_ks), args.iters)
            pa = timeit(lambda: aot[1](probs, top_ps), args.iters)
            row += f" {ka:>7.3f} {pa:>7.3f}"
        print(row)

        del probs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
