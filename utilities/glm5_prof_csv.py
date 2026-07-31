#!/usr/bin/env python3
"""Aggregate an SGLang torch-profiler trace into a per-Block / per-Module /
per-kernel CSV (MI355X only). Attributes each GPU kernel to its leaf nn.Module
via correlation -> host launch -> enclosing nn.Module annotation interval.

Requires CUDA graphs DISABLED (each kernel launched eagerly so the host
cuda_runtime launch carries the correlation and sits inside the module
annotations). Per-layer microseconds = total / (layers-in-block * forwards).

Usage:
  python3 glm5_prof_csv.py <trace.json[.gz]> [out.csv]
"""
import bisect
import collections
import gzip
import json
import sys


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path) as f:
        return json.load(f)


def cls_of(name):
    body = name.split(":", 1)[1].strip()
    base, _, idx = body.rpartition("_")
    try:
        return base, int(idx)
    except ValueError:
        return body, -1


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    d = load(path)
    ev = d["traceEvents"]

    tid_counts = collections.Counter()
    for e in ev:
        if e.get("name", "").startswith("nn.Module:"):
            tid_counts[e.get("tid")] += 1
    mod_tid = tid_counts.most_common(1)[0][0] if tid_counts else None

    intervals = []
    inst_count = collections.Counter()
    for e in ev:
        n = e.get("name", "")
        if not n.startswith("nn.Module:") or e.get("tid") != mod_tid:
            continue
        cls, idx = cls_of(n)
        ts = e.get("ts", 0.0)
        intervals.append((ts, ts + e.get("dur", 0.0), cls, idx))
        inst_count[cls] += 1
    intervals.sort()
    starts = [iv[0] for iv in intervals]

    def stack_at(ts):
        hi = bisect.bisect_right(starts, ts)
        anc = []
        for k in range(hi - 1, -1, -1):
            s, eend, cls, idx = intervals[k]
            if s <= ts < eend:
                anc.append((cls, idx))
            if ts - s > 5e6:
                break
        anc.reverse()
        return anc

    launch_ts = {}
    for e in ev:
        if e.get("cat") == "cuda_runtime":
            c = e.get("args", {}).get("correlation")
            if c is not None and c not in launch_ts:
                launch_ts[c] = e.get("ts", 0.0)

    GPU_CATS = ("kernel", "Kernel", "gpu_memcpy", "gpu_memset")
    agg = collections.defaultdict(lambda: [0.0, 0])
    unattributed = [0.0, 0]
    seen_dense_layers = set()
    seen_moe_layers = set()

    for e in ev:
        if e.get("cat") not in GPU_CATS:
            continue
        dur = e.get("dur", 0.0)
        kern = e.get("name", "?")
        corr = e.get("args", {}).get("correlation")
        ts = launch_ts.get(corr)
        if ts is None:
            unattributed[0] += dur
            unattributed[1] += 1
            continue
        anc = stack_at(ts)
        if not anc:
            unattributed[0] += dur
            unattributed[1] += 1
            continue
        leaf = anc[-1][0]
        anc_cls = {c for c, _ in anc}
        layer_idx = None
        for cls, idx in anc:
            if cls == "DeepseekV2DecoderLayer":
                layer_idx = idx
        if anc_cls & {"DeepseekV2MoE", "FusedMoE", "MoEGate", "TopK"}:
            block, sub = "B", "MoE"
            if layer_idx is not None:
                seen_moe_layers.add(layer_idx)
        elif anc_cls & {"DeepseekV2MLP", "SiluAndMul", "MergedColumnParallelLinear"}:
            block, sub = "A", "DenseMLP"
            if layer_idx is not None:
                seen_dense_layers.add(layer_idx)
        elif "DeepseekV2AttentionMLA" in anc_cls:
            block, sub = "L", "Attn(MLA)"
        elif layer_idx is not None:
            block, sub = "L", "Norm/Residual"
        else:
            block, sub = "G", leaf
        kl = kern.lower()
        if "cross_device_reduce" in kl or "all_reduce" in kl or "allreduce" in kl \
                or "rccl" in kl or "nccl" in kl:
            sub = "AllReduce(TP)"
        agg[(block, sub, leaf, kern)][0] += dur
        agg[(block, sub, leaf, kern)][1] += 1

    n_dense = len(seen_dense_layers) or 3
    n_moe = len(seen_moe_layers) or 75
    rows = [(block, sub, leaf, kern, us, cnt)
            for (block, sub, leaf, kern), (us, cnt) in agg.items()]
    nlayers = {"A": n_dense, "B": n_moe, "L": n_dense + n_moe, "G": 1}
    n_fwd = inst_count.get("DeepseekV2Model", 1) or 1

    header = ["Block", "BlockLayers", "DecoderSubmodule", "LeafModule",
              "KernelName", "Calls", "Total_us", "PerLayer_us"]
    block_order = {"A": 0, "L": 1, "B": 2, "G": 3}
    rows.sort(key=lambda r: (block_order.get(r[0], 9), r[1], r[2], -r[4]))
    lines = [",".join(header)]
    for block, sub, leaf, kern, us, cnt in rows:
        nlyr = nlayers.get(block, 1)
        per = us / (nlyr * n_fwd) if nlyr else us
        kn = '"' + kern.replace('"', "'")[:200] + '"'
        lines.append("%s,%d,%s,%s,%s,%d,%.1f,%.2f" %
                      (block, nlyr, sub, leaf, kn, cnt, us, per))
    csv = "\n".join(lines) + "\n"
    if out:
        with open(out, "w") as f:
            f.write(csv)

    print("trace:", path)
    print("module thread tid:", mod_tid, " module intervals:", len(intervals))
    print("dense layers:", n_dense, " moe layers:", n_moe, " forward passes:", n_fwd,
          " unattributed kernel us=%.1f (n=%d)" % (unattributed[0], unattributed[1]))
    sub_us = collections.defaultdict(float)
    sub_cnt = collections.defaultdict(int)
    for block, sub, leaf, kern, us, cnt in rows:
        sub_us[(block, sub)] += us
        sub_cnt[(block, sub)] += cnt
    print("\n%-6s %-7s %-16s %12s %10s %12s" %
          ("Block", "#Lyr", "Submodule", "Total_us", "Calls", "PerLayer_us"))
    for (block, sub) in sorted(sub_us, key=lambda k: (block_order.get(k[0], 9), -sub_us[k])):
        nl = nlayers.get(block, 1) * n_fwd
        print("%-6s %-7d %-16s %12.1f %10d %12.2f" %
              (block, nlayers.get(block, 1), sub, sub_us[(block, sub)],
               sub_cnt[(block, sub)], sub_us[(block, sub)] / nl if nl else 0))
    tot = sum(sub_us.values())
    print("\nTOTAL attributed GPU kernel us = %.1f" % tot)
    if out:
        print("wrote:", out)


if __name__ == "__main__":
    main()
