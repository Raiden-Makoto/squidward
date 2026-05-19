#!/usr/bin/env python3
"""
Analyze SGLang torch profiler trace files (.trace.json.gz).

Usage:
    python3 scripts/analyze_trace.py <trace_dir> [--threshold-ms N] [--op-filter PATTERN]
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

# Noisy internal profiler/watchdog/threading ops to suppress by default
_NOISE_PATTERNS = re.compile(
    r"^(ProfilerStep|Record|RecordFunction|"
    r"Thread|python_event|[Pp]ython_|"
    r"cudaStream|cudaEvent|cudaDevice|cudaLaunch|"
    r"[Cc]upti|CUPTI|nvtx|NVTX|"
    r"[Ww]atchdog|[Ss]ignal|aten::_record_function|"
    r"autograd::engine|[Ee]ngine::evaluate|"
    r"Optimizer\.step|zero_grad|"
    r"__\w+__|<built-in).*"
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _iter_events(path: Path) -> Iterator[dict]:
    """Stream-parse a gzip JSON trace, yielding ph=X events that have a dur."""
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rb") as fh:
        # The file can be either {"traceEvents": [...]} or a bare array.
        # ijson would be ideal for true streaming but we avoid extra deps.
        # Instead, load lazily and iterate only traceEvents.
        raw = fh.read()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  [WARN] Could not parse {path.name}: {exc}", file=sys.stderr)
        return

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    for ev in events:
        if ev.get("ph") == "X" and "dur" in ev:
            yield ev


def _collect_stats(path: Path) -> dict[str, dict]:
    """Return {op_name: {count, total_us, p99_us, max_us}} for one file."""
    durations: defaultdict[str, list[float]] = defaultdict(list)

    for ev in _iter_events(path):
        name = ev.get("name", "")
        if not name or _NOISE_PATTERNS.match(name):
            continue
        durations[name].append(float(ev["dur"]))

    stats: dict[str, dict] = {}
    for name, durs in durations.items():
        durs.sort()
        n = len(durs)
        p99_idx = max(0, int(n * 0.99) - 1)
        stats[name] = {
            "count": n,
            "avg_us": sum(durs) / n,
            "p99_us": durs[p99_idx],
            "max_us": durs[-1],
        }
    return stats


def _us_to_ms(us: float) -> float:
    return us / 1_000.0


# ── display ──────────────────────────────────────────────────────────────────

_COL_OP   = 52
_COL_CNT  = 8
_COL_AVG  = 10
_COL_P99  = 10
_COL_MAX  = 10

def _header() -> str:
    return (
        f"{'Op Name':<{_COL_OP}} {'Count':>{_COL_CNT}} "
        f"{'Avg(ms)':>{_COL_AVG}} {'P99(ms)':>{_COL_P99}} {'Max(ms)':>{_COL_MAX}}"
    )

def _sep() -> str:
    return "-" * (_COL_OP + _COL_CNT + _COL_AVG + _COL_P99 + _COL_MAX + 4)

def _row(name: str, s: dict) -> str:
    short = name if len(name) <= _COL_OP else name[:_COL_OP - 1] + "…"
    return (
        f"{short:<{_COL_OP}} {s['count']:>{_COL_CNT},d} "
        f"{_us_to_ms(s['avg_us']):>{_COL_AVG}.3f} "
        f"{_us_to_ms(s['p99_us']):>{_COL_P99}.3f} "
        f"{_us_to_ms(s['max_us']):>{_COL_MAX}.3f}"
    )


def _print_file_summary(
    path: Path,
    stats: dict[str, dict],
    threshold_us: float,
    op_filter: re.Pattern | None,
    top_n: int = 30,
) -> None:
    filtered = {
        name: s
        for name, s in stats.items()
        if s["max_us"] >= threshold_us
        and (op_filter is None or op_filter.search(name))
    }
    if not filtered:
        print(f"\n{'='*70}")
        print(f"  {path.name}  — no ops above threshold / matching filter")
        return

    ranked = sorted(filtered.items(), key=lambda kv: kv[1]["max_us"], reverse=True)[:top_n]

    print(f"\n{'='*70}")
    print(f"  {path}")
    print(f"  {len(stats)} unique ops total; showing top {len(ranked)} by max_ms")
    print(_sep())
    print(_header())
    print(_sep())
    for name, s in ranked:
        print(_row(name, s))
    print(_sep())


# ── cross-file summary ────────────────────────────────────────────────────────

def _print_cross_summary(all_results: list[tuple[Path, dict[str, dict]]]) -> None:
    worst: tuple[float, str, Path, dict] | None = None  # (max_us, name, path, stats)
    for path, stats in all_results:
        for name, s in stats.items():
            if worst is None or s["max_us"] > worst[0]:
                worst = (s["max_us"], name, path, s)

    print(f"\n{'='*70}")
    print("CROSS-FILE SUMMARY")
    print(_sep())

    if worst is None:
        print("  (no data)")
    else:
        max_us, name, path, s = worst
        print(f"  Worst single op across all traces:")
        print(f"    Op   : {name}")
        print(f"    File : {path}")
        print(f"    Avg  : {_us_to_ms(s['avg_us']):.3f} ms")
        print(f"    P99  : {_us_to_ms(s['p99_us']):.3f} ms")
        print(f"    Max  : {_us_to_ms(s['max_us']):.3f} ms")
        print(f"    Count: {s['count']:,d}")

    print(_sep())


# ── main ──────────────────────────────────────────────────────────────────────

def _find_traces(root: Path) -> list[Path]:
    return sorted(root.rglob("*.trace.json.gz"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SGLang torch profiler trace files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Directory to search recursively for *.trace.json.gz files",
    )
    parser.add_argument(
        "--threshold-ms",
        type=float,
        default=100.0,
        metavar="MS",
        help="Only show ops whose max duration exceeds this value (default: 100 ms)",
    )
    parser.add_argument(
        "--op-filter",
        type=str,
        default=None,
        metavar="PATTERN",
        help="Regex pattern to filter op names (case-insensitive substring match)",
    )
    args = parser.parse_args()

    trace_dir: Path = args.trace_dir.expanduser().resolve()
    if not trace_dir.is_dir():
        sys.exit(f"Error: '{trace_dir}' is not a directory.")

    threshold_us = args.threshold_ms * 1_000.0
    op_filter = (
        re.compile(args.op_filter, re.IGNORECASE) if args.op_filter else None
    )

    traces = _find_traces(trace_dir)
    if not traces:
        sys.exit(f"No *.trace.json.gz files found under '{trace_dir}'.")

    print(f"Found {len(traces)} trace file(s) under {trace_dir}")

    all_results: list[tuple[Path, dict[str, dict]]] = []
    for i, path in enumerate(traces, 1):
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"\n[{i}/{len(traces)}] Parsing {path.name} ({size_mb:.1f} MB)…", end="", flush=True)
        stats = _collect_stats(path)
        print(f" {len(stats)} unique ops.", flush=True)
        all_results.append((path, stats))
        _print_file_summary(path, stats, threshold_us, op_filter)

    if len(all_results) > 1 or (len(all_results) == 1 and op_filter is None):
        _print_cross_summary(all_results)


if __name__ == "__main__":
    main()
