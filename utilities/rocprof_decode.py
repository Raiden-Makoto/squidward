#!/usr/bin/env python3
"""Aggregate a rocprofv3 kernel_trace.csv into per-kernel GPU time over a time
window (tail fraction), optionally filtered to high-call-count (per-step)
kernels. Buckets kernels by role.

Usage: rocprof_decode.py <kernel_trace.csv> [frac] [min_calls]
  frac=1.0 -> whole trace; min_calls filters out low-count (e.g. prefill) kernels
"""
import collections
import csv
import sys


def cat_of(n):
    s = n.lower()
    if "cross_device_reduce" in s or "allreduce" in s or "all_reduce" in s or "rccl" in s or "twoshot" in s or "local_device_load_rmsnorm" in s:
        return "allreduce/fused"
    if "main_kernel" in s or n == "main" or "sparse_mla" in s:
        return "attn_sparse"
    if "mfma_moe" in s or "moe_sort" in s or "moe1" in s or "moe2" in s or "flydsl_moe" in s or "scaled_quant" in s:
        return "moe"
    if "grouped_topk" in s or "shared_expert" in s:
        return "moe_route"
    if "hadamard" in s or "act_quant" in s or "indexer" in s or "mqa_logits" in s or "topk_transform" in s or "wv_splitk" in s:
        return "indexer"
    if "rmsnorm" in s or "rms_norm" in s or "layernorm" in s or "fused_qk" in s:
        return "norm"
    if "rope" in s or "rotary" in s or "cached_positions" in s:
        return "rope"
    if "cijk" in s or "hgemm" in s or "gemm" in s or "bf16gemm" in s:
        return "gemm"
    if "elementwise" in s or "copy" in s or "fill" in s or "memcpy" in s or "memset" in s or "cast" in s or "cat" in s.split("array")[0]:
        return "elementwise"
    return "other"


path = sys.argv[1]
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
min_calls = int(sys.argv[3]) if len(sys.argv) > 3 else 0

rows = []
tmin = tmax = None
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        st = int(r["Start_Timestamp"]); en = int(r["End_Timestamp"])
        rows.append((st, en, r["Kernel_Name"]))
        tmin = st if tmin is None else min(tmin, st)
        tmax = en if tmax is None else max(tmax, en)

win_start = tmax - int(frac * (tmax - tmin))
dur = collections.defaultdict(float)
cnt = collections.defaultdict(int)
for st, en, name in rows:
    if st < win_start:
        continue
    dur[name] += en - st
    cnt[name] += 1
if min_calls:
    for k in list(dur):
        if cnt[k] < min_calls:
            del dur[k]; del cnt[k]

catdur = collections.defaultdict(float)
for k in dur:
    catdur[cat_of(k)] += dur[k]
tot = sum(dur.values())
print("trace:", path, " window=last %.0f%%  min_calls=%d" % (frac * 100, min_calls))
print("total kernel GPU us in window = %.1f" % (tot / 1e3))
print("\n== per-category ==")
for c, v in sorted(catdur.items(), key=lambda kv: -kv[1]):
    print("  %-16s %12.1f us  %5.1f%%" % (c, v / 1e3, 100 * v / tot if tot else 0))
print("\n== top 30 kernels (total_us, calls, avg_us) ==")
for v, c, k in sorted([(dur[k], cnt[k], k) for k in dur], key=lambda r: -r[0])[:30]:
    print("%11.1f %7d %10.2f  %s" % (v / 1e3, c, v / c / 1e3, k[:88]))
