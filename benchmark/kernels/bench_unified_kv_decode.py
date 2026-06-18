"""Microbenchmark for the V4 unified-KV sparse paged-DECODE kernel.

Isolates the fp8 "dequant tax" on the decode attention read at V4-Pro shapes
(H = num_heads/TP = 16, D = 512). Unlike prefill, decode uses the SAME Triton
kernel (``sparse_attn_v4_paged_decode``) for both bf16 and fp8 — the only
difference is ``kv_scales`` (None → bf16 pool; set → fp8 pool + 1x64 dequant
in-kernel). So the bf16↔fp8 gap here is purely:

  - extra per-tile fp32 scale load,
  - fp8→q.dtype cast,
  - per-group multiply (broadcast over GROUP_SIZE),
  - the BLOCK_K=32/NUM_K_STAGES=2 (fp8) vs BLOCK_K=16/NUM_K_STAGES=3 (bf16)
    config split chosen in ``_sparse_attn_v4_paged_decode_triton``.

fp8 reads HALF the value bytes of bf16 (512 vs 1024 B/slot) + 32 B of scales,
so on a bandwidth-bound decode fp8 *should* win; a ratio > 1 means the dequant
ALU / reg pressure is eating the bandwidth saving.

Sweeps (T, kv_len): low-conc (T=2) through high-conc (T=64), each over a range
of per-token sparse kv_len. T maps to decode batch size (DP-sharded at high
conc); kv_len is the number of committed/selected slots each decode token
attends (constant per token here).

Run on the gfx950 box inside the dsv4 container:
    PYTHONPATH=/sgl-workspace/squidward/python:$PYTHONPATH \
        python /sgl-workspace/squidward/benchmark/kernels/bench_unified_kv_decode.py

Add ``--check`` to validate Triton(fp8)/Triton(bf16) against a torch reference.
"""

from __future__ import annotations

import argparse

import torch

from sglang.srt.layers.attention.dsv4.unified_kv_kernels.paged_decode import (
    _FP8_DTYPE,
    _FP8_GROUP_SIZE,
    sparse_attn_v4_paged_decode,
)

NEG = -3.4028234663852886e38


def quantize_fp8_pertoken(kv_bf16: torch.Tensor, num_groups: int | None = None):
    """Per-token (1xD) fp8 quant matching the store-kernel contract: ONE scale
    per slot (= amax(|slot|)/fp8_max), replicated across ``num_groups`` scale
    columns. Prefill/OPUS read all columns (1x64 dequant); the native fp8-MFMA
    decode kernel reads column 0 as the per-token scale."""
    P, D = kv_bf16.shape
    if num_groups is None:
        num_groups = D // _FP8_GROUP_SIZE
    x = kv_bf16.float()
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [P, 1]
    fmax = torch.finfo(_FP8_DTYPE).max
    scale = (amax / fmax).clamp(min=1e-12)  # [P, 1]
    q = (x / scale).clamp(-fmax, fmax).to(_FP8_DTYPE)
    scales = scale.repeat(1, num_groups).to(torch.float32).contiguous()  # [P, num_groups]
    return q.contiguous(), scales


def dequant_fp8_pertoken(kv_fp8: torch.Tensor, scales: torch.Tensor):
    # All scale columns are equal (per-token); dequant with column 0.
    return kv_fp8.float() * scales[:, :1]


def make_inputs(T, H, D, kv_len, n_pages, device, seed=0):
    """Random V4-decode inputs. Each of T query tokens attends ``kv_len``
    distinct (random) slots into a pool of ``n_pages`` to defeat L1 reuse."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(T, H, D, device=device, dtype=torch.bfloat16, generator=g) * 0.1
    kv_pool = torch.randn(n_pages, D, device=device, dtype=torch.bfloat16, generator=g) * 0.1
    kv_indices = torch.randint(
        0, n_pages, (T * kv_len,), device=device, dtype=torch.int32, generator=g
    )
    kv_indptr = torch.arange(
        0, (T + 1) * kv_len, kv_len, device=device, dtype=torch.int32
    )
    attn_sink = torch.randn(H, device=device, dtype=torch.float32, generator=g) * 0.5
    return q, kv_pool, kv_indices, kv_indptr, attn_sink


def ref_decode(q, kv_pool, kv_indices, kv_indptr, attn_sink, scale):
    """fp32 dense reference (online-softmax-equivalent) per token."""
    T, H, D = q.shape
    out = torch.empty(T, H, D, device=q.device, dtype=torch.float32)
    qf = q.float()
    kpf = kv_pool.float()
    for t in range(T):
        slots = kv_indices[kv_indptr[t] : kv_indptr[t + 1]].long()
        K = kpf[slots]  # [L, D]
        s = (qf[t] @ K.T) * scale  # [H, L]
        m = torch.maximum(s.max(dim=1).values, attn_sink)  # [H]
        p = torch.exp(s - m[:, None])
        denom = p.sum(dim=1) + torch.exp(attn_sink - m)
        out[t] = (p @ K) / denom[:, None]
    return out


def _bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


def run_point(T, H, D, kv_len, n_pages, iters, warmup, do_check):
    dev = "cuda"
    q, kv_pool, kv_indices, kv_indptr, sink = make_inputs(T, H, D, kv_len, n_pages, dev)
    scale = float(D) ** -0.5
    kv_fp8, kv_scales = quantize_fp8_pertoken(kv_pool)

    res = {"T": T, "kv_len": kv_len}

    def f_bf16():
        return sparse_attn_v4_paged_decode(
            q, kv_pool, kv_indices, kv_indptr, sink, scale
        )

    res["bf16_us"] = _bench(f_bf16, iters, warmup)

    def f_fp8():
        return sparse_attn_v4_paged_decode(
            q, kv_fp8, kv_indices, kv_indptr, sink, scale, kv_scales=kv_scales
        )

    res["fp8_us"] = _bench(f_fp8, iters, warmup)

    if do_check:
        ref = ref_decode(q, kv_pool, kv_indices, kv_indptr, sink, scale)
        o_bf16 = f_bf16().float()
        o_fp8 = f_fp8().float()
        res["bf16_max_abs"] = (o_bf16 - ref).abs().max().item()
        ref_fp8 = ref_decode(
            q, dequant_fp8_pertoken(kv_fp8, kv_scales), kv_indices, kv_indptr, sink, scale
        )
        res["fp8_max_abs"] = (o_fp8 - ref_fp8).abs().max().item()
    return res


def run_store_point(T, D, iters, warmup):
    """Time the per-layer decode KV-store (``store_swa_into_unified``) for the
    bf16 (plain masked copy) vs fp8 (per-token amax reduction + scale write)
    paths at decode batch sizes. This kernel runs every layer / every step and
    is NOT covered by the attention microbench above; at decode batch (T rows,
    few CTAs) it is latency-bound, so its bf16->fp8 delta is a roughly fixed
    per-layer tax on the decode step."""
    from sglang.srt.layers.attention.dsv4.unified_kv_kernels import runtime

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    win = 4096
    ring_stride = win
    n_pages = T * ring_stride + ring_stride
    kv = torch.randn(T, D, device=dev, dtype=torch.bfloat16, generator=g) * 0.1
    state_slot = torch.arange(T, device=dev, dtype=torch.int32)
    positions = torch.randint(0, ring_stride, (T,), device=dev, dtype=torch.int32, generator=g)

    pool_bf16 = torch.zeros(n_pages, D, device=dev, dtype=torch.bfloat16)
    pool_fp8 = torch.zeros(n_pages, D, device=dev, dtype=_FP8_DTYPE)
    scales = torch.zeros(n_pages, D // _FP8_GROUP_SIZE, device=dev, dtype=torch.float32)

    def f_bf16():
        runtime.store_swa_into_unified(
            kv=kv, state_slot=state_slot, positions=positions,
            unified_kv=pool_bf16, win=win, ring_stride=ring_stride,
        )

    def f_fp8():
        runtime.store_swa_into_unified(
            kv=kv, state_slot=state_slot, positions=positions,
            unified_kv=pool_fp8, win=win, ring_stride=ring_stride,
            unified_kv_scales=scales,
        )

    return {
        "T": T,
        "bf16_us": _bench(f_bf16, iters, warmup),
        "fp8_us": _bench(f_fp8, iters, warmup),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--n-pages", type=int, default=131072)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--store", action="store_true", help="bench the KV-store kernel (bf16 copy vs fp8 quant) instead of attention")
    ap.add_argument(
        "--shapes",
        type=str,
        default=(
            "2:1024,2:2048,2:4096,2:8192,"
            "32:1024,32:2048,32:4096,32:8192,"
            "64:1024,64:2048,64:4096,64:8192"
        ),
        help="comma-sep T:kv_len pairs",
    )
    args = ap.parse_args()

    print(
        f"_FP8_DTYPE={_FP8_DTYPE}  "
        f"arch={torch.cuda.get_device_properties(0).gcnArchName}"
    )

    if args.store:
        srows = [
            run_store_point(T, args.D, args.iters, args.warmup)
            for T in (1, 2, 4, 8, 16, 32, 64, 128)
        ]
        print(f"{'T':>6} {'bf16_us':>10} {'fp8_us':>10} {'fp8-bf16_us':>12} {'fp8/bf16':>9}")
        for r in srows:
            d = r["fp8_us"] - r["bf16_us"]
            print(
                f"{r['T']:>6} {r['bf16_us']:>10.2f} {r['fp8_us']:>10.2f} "
                f"{d:>12.2f} {r['fp8_us']/r['bf16_us']:>9.2f}"
            )
        return

    rows = []
    if args.check:
        rows.append(run_point(8, args.H, args.D, 512, 4096, args.iters, args.warmup, True))
    for tok in args.shapes.split(","):
        T, kv_len = (int(x) for x in tok.split(":"))
        rows.append(
            run_point(T, args.H, args.D, kv_len, args.n_pages, args.iters, args.warmup, args.check)
        )

    hdr = f"{'T':>6} {'kv_len':>7} {'bf16_us':>10} {'fp8_us':>10} {'fp8/bf16':>9}"
    print(hdr)
    for r in rows:
        ratio = r["fp8_us"] / r["bf16_us"]
        line = (
            f"{r['T']:>6} {r['kv_len']:>7} "
            f"{r['bf16_us']:>10.1f} {r['fp8_us']:>10.1f} {ratio:>9.2f}"
        )
        if "fp8_max_abs" in r:
            line += f"   bf16_err={r['bf16_max_abs']:.4e} fp8_err={r['fp8_max_abs']:.4e}"
        print(line)


if __name__ == "__main__":
    main()
