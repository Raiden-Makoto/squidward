#!/usr/bin/env python3
"""Aggregate a rocprofv3 kernel_trace.csv: for MoE stage kernels, report count,
total us, and dispatch metadata (VGPR/AGPR/LDS/scratch/grid/wg) for occupancy."""
import csv
import sys
import collections

path = sys.argv[1]
pats = ("mfma_moe1", "mfma_moe2", "moe_gather", "reduction", "moe_sort",
        "scaled_quant", "grouped_topk")
agg = collections.defaultdict(lambda: [0.0, 0, None])
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        name = r["Kernel_Name"]
        low = name.lower()
        if not any(p in low for p in pats):
            continue
        dur = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3
        key = name[:70]
        a = agg[key]
        a[0] += dur
        a[1] += 1
        if a[2] is None:
            a[2] = dict(vgpr=r["VGPR_Count"], agpr=r["Accum_VGPR_Count"],
                        sgpr=r["SGPR_Count"], lds=r["LDS_Block_Size"],
                        scr=r["Scratch_Size"],
                        wg=int(r["Workgroup_Size_X"]) * int(r["Workgroup_Size_Y"]) * int(r["Workgroup_Size_Z"]),
                        grid=int(r["Grid_Size_X"]) * int(r["Grid_Size_Y"]) * int(r["Grid_Size_Z"]))
for k, (us, n, m) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    blk = m["grid"] // m["wg"] if m["wg"] else 0
    print("%9.1f us  x%-4d vgpr=%s agpr=%s sgpr=%s lds=%s scr=%s wg=%d blocks=%d  %s"
          % (us, n, m["vgpr"], m["agpr"], m["sgpr"], m["lds"], m["scr"], m["wg"], blk, k))
