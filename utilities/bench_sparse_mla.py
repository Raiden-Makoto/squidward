#!/usr/bin/env python3
"""A/B harness for the DSA triton sparse-MLA prefill kernel (#28975).

Automates the kernel dev loop: correctness vs a bf16 reference + latency /
effective-bandwidth timing for the current `triton_sparse_mla_fwd`, across the
GLM-5.2 prefill M-buckets. Run it before and after a kernel change (e.g. a gluon
port) and diff the numbers.

    PYTHONPATH=/sgl-workspace/squidward/python python3 utilities/bench_sparse_mla.py
    ... --seqs 8192,24501 --topk 2048 --iters 30
"""
import argparse
import time

import torch

from sglang.srt.layers.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd

H, D_V, D_TAIL, DIM = 16, 512, 64, 576  # GLM-5.2 @ TP4 indexer/MLA geometry


def _ref(q_nope, q_rope, kv, idx, sm_scale):
    """bf16 dense reference over the selected topk (correctness oracle)."""
    seq = q_nope.shape[0]
    qm = q_nope.float()
    qt = q_rope.float()
    kvf = kv.float().squeeze(1)  # [pages, DIM]
    out = torch.empty(seq, H, D_V, device=q_nope.device, dtype=torch.float32)
    for i in range(seq):
        page = idx[i, 0].long()
        valid = page >= 0
        p = page.clamp(min=0)
        km = kvf[p, :D_V]  # [topk, D_V]
        kt = kvf[p, D_V:]  # [topk, D_TAIL]
        qk = (qm[i] @ km.T + qt[i] @ kt.T) * sm_scale  # [H, topk]
        qk = qk.masked_fill(~valid[None, :], float("-inf"))
        w = torch.softmax(qk, dim=-1)
        out[i] = w @ km
    return out


def _mk(seq, topk, npages, dev):
    torch.manual_seed(0)
    qn = (torch.randn(seq, H, D_V, device=dev) * 0.1).to(torch.float8_e4m3fn)
    qr = (torch.randn(seq, H, D_TAIL, device=dev) * 0.1).to(torch.float8_e4m3fn)
    kv = (torch.randn(npages, 1, DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    idx = torch.randint(0, npages, (seq, 1, topk), device=dev, dtype=torch.int32)
    return qn, qr, kv, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="8192,24501")
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--npages", type=int, default=65536)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--check", action="store_true", help="run bf16 correctness (slow)")
    a = ap.parse_args()
    dev = "cuda"
    sm_scale = 1.0 / (D_V**0.5)
    print(f"{'seq':>7} {'ms/call':>9} {'gatherGB':>9} {'eff TB/s':>9} {'maxdiff':>9}")
    for seq in (int(s) for s in a.seqs.split(",")):
        qn, qr, kv, idx = _mk(seq, a.topk, a.npages, dev)
        maxdiff = float("nan")
        if a.check:
            n = min(seq, 256)  # reference is O(seq*topk*D); sample rows
            o = triton_sparse_mla_fwd(qn[:n], qr[:n], kv, idx[:n], sm_scale, d_v=D_V)[0]
            r = _ref(qn[:n], qr[:n], kv, idx[:n], sm_scale)
            maxdiff = (o.float() - r).abs().max().item()
        for _ in range(10):
            triton_sparse_mla_fwd(qn, qr, kv, idx, sm_scale, d_v=D_V)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(a.iters):
            triton_sparse_mla_fwd(qn, qr, kv, idx, sm_scale, d_v=D_V)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / a.iters * 1000
        gb = seq * a.topk * DIM / 1e9
        eff = gb / (ms / 1000) / 1e3
        print(f"{seq:>7} {ms:>9.3f} {gb:>9.2f} {eff:>9.2f} {maxdiff:>9.4f}")


if __name__ == "__main__":
    main()
