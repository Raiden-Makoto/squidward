"""Microbenchmark for the V4 unified-KV sparse paged-prefill kernel.

Decomposes the prefill cost into three comparable points at V4-Pro shapes
(H = num_heads/TP = 16, D = 512):

  - OPUS  (bf16): hand-tuned aiter ``pa_sparse_prefill_opus`` (gfx950 only).
  - Triton(bf16): ``_sparse_attn_v4_paged_prefill_triton`` with kv_scales=None.
  - Triton(fp8) : same Triton kernel with the fp8 unified pool + 1x64 scales.

The OPUS↔Triton(bf16) gap is the "kernel-quality" tax; the
Triton(bf16)↔Triton(fp8) gap is the "fp8 dequant" tax. The fp8 path is forced
onto Triton because OPUS is bf16-only, so closing the Triton↔OPUS gap is what
matters for the fp8 deployment.

Run on the gfx950 box inside the dsv4 container:
    PYTHONPATH=/sgl-workspace/squidward/python:$PYTHONPATH \
        python /sgl-workspace/squidward/benchmark/kernels/bench_unified_kv_prefill.py

Add ``--check`` to validate Triton(fp8)/Triton(bf16) against a torch reference.
"""

from __future__ import annotations

import argparse

import torch

from sglang.srt.layers.attention.dsv4.unified_kv_kernels.paged_decode import (
    _FP8_DTYPE,
    _FP8_GROUP_SIZE,
)
from sglang.srt.layers.attention.dsv4.unified_kv_kernels.paged_prefill import (
    _HAS_OPUS,
    _sparse_attn_v4_paged_prefill_triton,
    sparse_attn_v4_paged_prefill,
)

NEG = -3.4028234663852886e38


def quantize_fp8_1xg(kv_bf16: torch.Tensor, group: int = _FP8_GROUP_SIZE):
    """1xGROUP block-scale quant matching the kernel's dequant contract."""
    P, D = kv_bf16.shape
    g = D // group
    x = kv_bf16.float().reshape(P, g, group)
    amax = x.abs().amax(dim=-1).clamp(min=1e-8)
    fmax = torch.finfo(_FP8_DTYPE).max
    scale = (amax / fmax).clamp(min=1e-12)
    q = (x / scale[..., None]).clamp(-fmax, fmax).to(_FP8_DTYPE)
    return q.reshape(P, D).contiguous(), scale.to(torch.float32).contiguous()


def dequant_fp8_1xg(kv_fp8: torch.Tensor, scale: torch.Tensor, group: int = _FP8_GROUP_SIZE):
    P, D = kv_fp8.shape
    g = D // group
    x = kv_fp8.float().reshape(P, g, group)
    return (x * scale[..., None]).reshape(P, D)


def make_inputs(T, H, D, p_len, e_len, n_pages, device, seed=0):
    """Random V4-prefill inputs. Each query token attends ``p_len`` distinct
    (random) prefix slots into a pool of ``n_pages`` to defeat L1 reuse, plus
    ``e_len`` extend rows from the live per-fwd ``kv``."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(T, H, D, device=device, dtype=torch.bfloat16, generator=g) * 0.1

    kv_pool = torch.randn(n_pages, D, device=device, dtype=torch.bfloat16, generator=g) * 0.1

    # prefix: T*p_len random valid slots, true prefix-sum indptr.
    kv_indices_prefix = torch.randint(
        0, n_pages, (T * p_len,), device=device, dtype=torch.int32, generator=g
    )
    kv_indptr_prefix = torch.arange(
        0, (T + 1) * p_len, p_len, device=device, dtype=torch.int32
    )

    # extend: e_len rows per token from a flat per-fwd kv.
    n_ext = max(e_len * T, 1)
    kv_extend = torch.randn(n_ext, D, device=device, dtype=torch.bfloat16, generator=g) * 0.1
    if e_len > 0:
        kv_indices_extend = torch.randint(
            0, n_ext, (T * e_len,), device=device, dtype=torch.int32, generator=g
        )
        kv_indptr_extend = torch.arange(
            0, (T + 1) * e_len, e_len, device=device, dtype=torch.int32
        )
    else:
        kv_indices_extend = torch.zeros(1, device=device, dtype=torch.int32)
        kv_indptr_extend = torch.zeros(T + 1, device=device, dtype=torch.int32)

    attn_sink = torch.randn(H, device=device, dtype=torch.float32, generator=g) * 0.5
    return (
        q,
        kv_pool,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_extend,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
    )


def ref_prefill(q, kv_pool, ipre, ppre, kv_ext, iext, pext, attn_sink, scale):
    """fp32 dense reference (online-softmax-equivalent) per token."""
    T, H, D = q.shape
    out = torch.empty(T, H, D, device=q.device, dtype=torch.float32)
    qf = q.float()
    kpf = kv_pool.float()
    kef = kv_ext.float()
    for t in range(T):
        slots = ipre[ppre[t] : ppre[t + 1]].long()
        slots = slots[slots >= 0]
        K = kpf[slots]
        es = iext[pext[t] : pext[t + 1]].long()
        es = es[es >= 0]
        if es.numel() > 0:
            K = torch.cat([K, kef[es]], dim=0)
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


def run_point(T, H, D, p_len, e_len, n_pages, iters, warmup, do_check):
    dev = "cuda"
    (q, kv_pool, ipre, ppre, kv_ext, iext, pext, sink) = make_inputs(
        T, H, D, p_len, e_len, n_pages, dev
    )
    scale = float(D) ** -0.5
    kv_fp8, kv_scales = quantize_fp8_1xg(kv_pool)

    res = {"T": T, "p_len": p_len, "e_len": e_len}

    # Triton bf16 (force triton; bypasses OPUS dispatch).
    def f_tri_bf16():
        return _sparse_attn_v4_paged_prefill_triton(
            q, kv_pool, ipre, ppre, kv_ext, iext, pext, sink, scale
        )

    res["tri_bf16_us"] = _bench(f_tri_bf16, iters, warmup)

    # Triton fp8.
    def f_tri_fp8():
        return _sparse_attn_v4_paged_prefill_triton(
            q, kv_fp8, ipre, ppre, kv_ext, iext, pext, sink, scale, kv_scales=kv_scales
        )

    res["tri_fp8_us"] = _bench(f_tri_fp8, iters, warmup)

    # OPUS bf16 (only when present; public dispatcher routes bf16→OPUS).
    if _HAS_OPUS:
        def f_opus():
            return sparse_attn_v4_paged_prefill(
                q, kv_pool, ipre, ppre, kv_ext, iext, pext, sink, scale
            )

        res["opus_us"] = _bench(f_opus, iters, warmup)
    else:
        res["opus_us"] = float("nan")

    if do_check:
        ref = ref_prefill(q, kv_pool, ipre, ppre, kv_ext, iext, pext, sink, scale)
        o_bf16 = f_tri_bf16().float()
        o_fp8 = f_tri_fp8().float()
        res["bf16_max_abs"] = (o_bf16 - ref).abs().max().item()
        ref_fp8 = ref_prefill(
            q, dequant_fp8_1xg(kv_fp8, kv_scales), ipre, ppre, kv_ext, iext, pext, sink, scale
        )
        res["fp8_max_abs"] = (o_fp8 - ref_fp8).abs().max().item()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--n-pages", type=int, default=65536)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--shapes",
        type=str,
        default="512:2048:0,512:8192:0,2048:2048:0,2048:2048:64",
        help="comma-sep T:p_len:e_len triples",
    )
    args = ap.parse_args()

    print(f"_HAS_OPUS={_HAS_OPUS}  _FP8_DTYPE={_FP8_DTYPE}  arch={torch.cuda.get_device_properties(0).gcnArchName}")
    rows = []
    if args.check:
        rows.append(run_point(8, args.H, args.D, 64, 4, 4096, args.iters, args.warmup, True))
    for tok in args.shapes.split(","):
        T, p_len, e_len = (int(x) for x in tok.split(":"))
        rows.append(
            run_point(T, args.H, args.D, p_len, e_len, args.n_pages, args.iters, args.warmup, args.check)
        )

    hdr = f"{'T':>6} {'p_len':>6} {'e_len':>5} {'opus_us':>10} {'tri_bf16_us':>12} {'tri_fp8_us':>11} {'fp8/opus':>9} {'fp8/tri':>8}"
    print(hdr)
    for r in rows:
        ratio_opus = r["tri_fp8_us"] / r["opus_us"] if r["opus_us"] == r["opus_us"] else float("nan")
        ratio_tri = r["tri_fp8_us"] / r["tri_bf16_us"]
        line = (
            f"{r['T']:>6} {r['p_len']:>6} {r['e_len']:>5} "
            f"{r['opus_us']:>10.1f} {r['tri_bf16_us']:>12.1f} {r['tri_fp8_us']:>11.1f} "
            f"{ratio_opus:>9.2f} {ratio_tri:>8.2f}"
        )
        if "fp8_max_abs" in r:
            line += f"   bf16_err={r['bf16_max_abs']:.4e} fp8_err={r['fp8_max_abs']:.4e}"
        print(line)


if __name__ == "__main__":
    main()
