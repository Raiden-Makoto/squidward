#!/usr/bin/env python3
"""Pivot a rocprofv3 counter_collection.csv (long format) per dispatch and group
by kernel signature (VGPR_Count, Grid_Size, Workgroup_Size) to separate kernel
variants (e.g. sparse-decode partial vs combine, prefill vs decode). Reports
per-group dispatch count, mean GPU duration, and the mean of each PMC.

Usage: rocprof_pmc_analyze.py <counter_collection.csv>
"""
import collections
import csv
import sys

path = sys.argv[1]
disp = {}  # dispatch_id -> dict
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        did = r["Dispatch_Id"]
        d = disp.setdefault(did, {
            "grid": int(r["Grid_Size"]), "wg": int(r["Workgroup_Size"]),
            "vgpr": int(r["VGPR_Count"]), "sgpr": int(r["SGPR_Count"]),
            "lds": int(r["LDS_Block_Size"]), "scratch": int(r["Scratch_Size"]),
            "dur": (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3,
            "ctr": {},
        })
        d["ctr"][r["Counter_Name"]] = float(r["Counter_Value"])

groups = collections.defaultdict(list)
for d in disp.values():
    groups[(d["vgpr"], d["grid"], d["wg"])].append(d)

ctr_names = sorted({c for d in disp.values() for c in d["ctr"]})
print("file:", path, " dispatches:", len(disp))
print("\n%-5s %-5s %-7s %-7s %-11s %-4s %5s %9s %8s  %s" % (
    "VGPR", "SGPR", "LDS_B", "Scratch", "GridThreads", "WG", "count",
    "mean_us", "blocks",
    "  ".join("%-14s" % c for c in ctr_names)))
for (vgpr, grid, wg), ds in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    n = len(ds)
    mean_dur = sum(d["dur"] for d in ds) / n
    blocks = grid // wg if wg else 0
    lds = ds[0]["lds"]
    sgpr = ds[0]["sgpr"]
    scratch = ds[0]["scratch"]
    means = []
    for c in ctr_names:
        vals = [d["ctr"][c] for d in ds if c in d["ctr"]]
        means.append(sum(vals) / len(vals) if vals else 0.0)
    print("%-5d %-5d %-7d %-7d %-11d %-4d %5d %9.2f %8d  %s" % (
        vgpr, sgpr, lds, scratch, grid, wg, n, mean_dur, blocks,
        "  ".join("%-14.3f" % m for m in means)))
