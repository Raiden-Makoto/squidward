#!/usr/bin/env python3
"""Print actual baseline vs flag e2e numbers per concurrency.
Usage: e2e_table.py <base_dir> <flag_dir>
"""
import glob
import os
import re
import sys

M = {
    "tot": r"Total token throughput \(tok/s\):\s*([\d.]+)",
    "out": r"Output token throughput \(tok/s\):\s*([\d.]+)",
    "ttft": r"Median TTFT \(ms\):\s*([\d.]+)",
    "itl": r"Median ITL \(ms\):\s*([\d.]+)",
    "e2el": r"Median E2E Latency \(ms\):\s*([\d.]+)",
}


def parse(d):
    # Average every metric across all rep logs found for each concurrency
    # (CLT smoothing). Each bench run writes its own file, so multiple files
    # per concurrency = multiple reps.
    acc = {}
    for f in glob.glob(os.path.join(d, "*.log")):
        m = re.search(r"_c-(\d+)_", os.path.basename(f))
        if not m:
            continue
        c = int(m.group(1))
        t = open(f, errors="ignore").read()
        slot = acc.setdefault(c, {k: [] for k in M})
        for k, p in M.items():
            r = re.findall(p, t)
            if r:
                slot[k].append(float(r[-1]))
    o = {}
    for c, slot in acc.items():
        v = {k: (sum(vs) / len(vs) if vs else float("nan")) for k, vs in slot.items()}
        v["n"] = max((len(vs) for vs in slot.values()), default=0)
        o[c] = v
    return o


b = parse(sys.argv[1])
f = parse(sys.argv[2])
cols = ["tot", "ttft", "itl", "e2el"]
hdr = (
    ["conc", "n"]
    + ["base_" + k for k in cols]
    + ["patch_" + k for k in cols]
    + ["ttft_delta_%"]
)
print(",".join(hdr))
for c in sorted(set(b) | set(f)):
    bb = b.get(c, {})
    ff = f.get(c, {})
    n = min(bb.get("n", 0), ff.get("n", 0))
    bt, pt = bb.get("ttft"), ff.get("ttft")
    dl = (
        (pt - bt) / bt * 100.0
        if isinstance(bt, float) and bt == bt and bt and isinstance(pt, float) and pt == pt
        else float("nan")
    )
    row = [c, n] + [bb.get(k) for k in cols] + [ff.get(k) for k in cols] + [dl]

    def fmt(x):
        if isinstance(x, float) and x == x:
            return ("%.2f" % x) if abs(x) < 100 else str(int(x))
        return str(x)
    print(",".join(fmt(x) for x in row))
