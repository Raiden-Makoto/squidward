#!/usr/bin/env python3
"""Read one or more rocprofv3 counter_collection.csv files and, for MoE stage
kernels, report the mean counter value (+ mean GPU us, VGPR/LDS/grid)."""
import csv
import sys
import collections

pats = ("mfma_moe1", "mfma_moe2")
for path in sys.argv[1:]:
    # dispatch_id -> (name, dur_us, vgpr, lds, wg, grid, {counter: value})
    disp = {}
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        continue
    for r in rows:
        name = r.get("Kernel_Name", "")
        low = name.lower()
        if not any(p in low for p in pats):
            continue
        did = r["Dispatch_Id"]
        d = disp.setdefault(did, {
            "name": name[:55],
            "dur": (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3,
            "vgpr": r["VGPR_Count"], "lds": r["LDS_Block_Size"],
            "wg": int(r["Workgroup_Size"]),
            "grid": int(r["Grid_Size"]), "ctr": {},
        })
        d["ctr"][r["Counter_Name"]] = float(r["Counter_Value"])
    agg = collections.defaultdict(lambda: [0.0, 0, collections.defaultdict(float), collections.defaultdict(int), None])
    for d in disp.values():
        a = agg[d["name"]]
        a[0] += d["dur"]; a[1] += 1
        for c, v in d["ctr"].items():
            a[2][c] += v; a[3][c] += 1
        if a[4] is None:
            a[4] = (d["vgpr"], d["lds"], d["wg"], d["grid"])
    for name, (us, n, csum, ccnt, meta) in sorted(agg.items()):
        cstr = " ".join("%s=%.2f" % (c, csum[c] / ccnt[c]) for c in sorted(csum))
        vgpr, lds, wg, grid = meta
        print("%-56s x%-3d meanus=%.1f vgpr=%s lds=%s wg=%d  %s"
              % (name, n, us / n, vgpr, lds, wg, cstr))
