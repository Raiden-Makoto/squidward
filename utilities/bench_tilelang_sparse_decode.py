#!/usr/bin/env python3
"""Isolated bench/PMC driver for the GLM-5.1 sparse-MLA decode TileLang kernel.

The GLM-5 FP4 decode attention `main_kernel` (profile: tilelang_kernel.py:1333
`tilelang_sparse_fwd` -> `sparse_mla_fwd_decode_partial_fp8`) is the dominant
decode attention cost (~8.6us/call, the biggest addressable gap vs B200's single
`fmhaSm100fKernel`). This drives that exact two-stage kernel (partial + combine)
standalone for timing and rocprofv3 PMC counter collection.

Shapes (GLM-5.1, tp4): num_heads=16/GPU, qk=576 (d_v=512 + rope_tail=64),
topk=2048, page_size=1, fp8_e4m3 q+kv.

Timing:
  PYTHONPATH=/sgl-workspace/squidward/python \
    python3 /tmp/bench_tilelang_sparse_decode.py --bs 4
PMC (single shot, no host-side timing loop noise):
  rocprofv3 -i /tmp/pmc.txt -d /tmp/tl_pmc -- \
    env PYTHONPATH=/sgl-workspace/squidward/python \
    python3 /tmp/bench_tilelang_sparse_decode.py --bs 4 --iters 100 --warmup 20 --no-timing
"""
import argparse
import statistics

import torch

NUM_HEADS = 16  # GLM-5.1 64 heads / tp4
DIM = 576  # kv_lora_rank(512) + qk_rope(64)
D_V = 512
TOPK = 2048
FP8 = torch.float8_e4m3fn  # gfx950 / MI350 e4m3fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bs-list",
        type=int,
        nargs="+",
        default=[4],
        help="decode-token counts (concurrency) to sweep, all in one process",
    )
    ap.add_argument("--num-pages", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument(
        "--no-timing",
        action="store_true",
        help="run a fixed iters loop with no events (for rocprof PMC)",
    )
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    from sglang.srt.layers.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    sm_scale = DIM**-0.5
    kv = torch.randn((args.num_pages, 1, DIM), dtype=torch.bfloat16, device=dev).to(FP8)

    def make(bs):
        q = torch.randn((bs, NUM_HEADS, DIM), dtype=torch.bfloat16, device=dev).to(FP8)
        idx = torch.randint(
            0, args.num_pages, (bs, 1, TOPK), dtype=torch.int32, device=dev
        )
        return q, idx

    if not args.no_timing:
        print(f"heads={NUM_HEADS} dim={DIM} topk={TOPK} pages={args.num_pages}")

    for bs in args.bs_list:
        q, idx = make(bs)

        def run():
            return tilelang_sparse_fwd(q, kv, idx, sm_scale, d_v=D_V)

        for _ in range(args.warmup):
            out = run()
        torch.cuda.synchronize()
        assert torch.isfinite(out.float()).all(), f"non-finite output bs={bs}"

        if args.no_timing:
            for _ in range(args.iters):
                run()
            torch.cuda.synchronize()
            continue

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(args.iters):
            s.record()
            run()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1e3)
        ts.sort()
        print(
            f"bs={bs:>4}  p50_us={ts[len(ts)//2]:8.2f}  min={ts[0]:8.2f}  "
            f"mean={statistics.mean(ts):8.2f}  (partial+combine, host-bound)"
        )

    if args.no_timing:
        torch.cuda.synchronize()
        print(f"PMC loop done: bs_list={args.bs_list}, iters={args.iters}")


if __name__ == "__main__":
    main()
