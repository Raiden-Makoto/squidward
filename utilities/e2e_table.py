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
    o = {}
    for f in glob.glob(os.path.join(d, "*.log")):
        m = re.search(r"_c-(\d+)_", os.path.basename(f))
        if not m:
            continue
        c = int(m.group(1))
        t = open(f, errors="ignore").read()
        v = {}
        for k, p in M.items():
            r = re.findall(p, t)
            v[k] = float(r[-1]) if r else float("nan")
        o[c] = v
    return o


b = parse(sys.argv[1])
f = parse(sys.argv[2])
cols = ["tot", "ttft", "itl", "e2el"]
hdr = ["conc"] + ["base_" + k for k in cols] + ["patch_" + k for k in cols]
print(",".join(hdr))
for c in sorted(set(b) | set(f)):
    bb = b.get(c, {})
    ff = f.get(c, {})
    row = [c] + [bb.get(k) for k in cols] + [ff.get(k) for k in cols]

    def fmt(x):
        if isinstance(x, float) and x == x:
            return ("%.2f" % x) if x < 100 else str(int(x))
        return str(x)
    print(",".join(fmt(x) for x in row))
