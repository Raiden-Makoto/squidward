"""Microbenchmark for the target-only tree verifier, Triton vs CUDA AOT.

Runs on ROCm (Triton only) and on CUDA (both, so the ratio isolates the port from
the hardware -- timings from two different GPUs are not comparable).

Two things this is careful about:

- Real tree topologies. A chain placeholder leaves `retrive_next_sibling` all -1,
  so the sibling walk that the tree verifier exists for never executes.
- Scratch state. The kernel writes rejected candidates back into `draft_probs`, so
  a second call on the same buffer takes different branches. The buffer is reset
  between iterations, outside the timed region -- folding a vocabulary-sized memset
  into the measurement would swamp the kernel itself.

Usage:  python3 bench_tree_sampling.py [--vocab 151936] [--iters 20]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsa.index_buf_accessor import MoveKAndS
from sglang.kernels.ops.speculative.tree_sampling import (
    tree_speculative_sampling_target_only_triton,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

DEVICE = torch.device("cuda")


def load_aot_kernel():
    """The python wrapper imports fine on ROCm; only the torch op is missing, so
    probe with a real call rather than trusting the import."""
    try:
        from sgl_kernel import tree_speculative_sampling_target_only as aot
    except (ImportError, AttributeError):
        return None
    try:
        args = make_inputs(batch_size=1, num_draft_tokens=2, width=1, vocab_size=8)
        aot(**args, threshold_single=1.0, threshold_acc=1.0, deterministic=True)
    except (AttributeError, NotImplementedError, RuntimeError):
        return None
    return aot


class Topology:
    """First-child / next-sibling encoding of a breadth-first k-ary tree.

    A uniform tree is a proxy: EAGLE-2 prunes globally by confidence, so a real
    draft tree is ragged. Width still bounds the branching, which is the axis that
    decides whether the sibling walk does any work.

    ``parent`` and ``rank`` are kept because the verifier scores a child against
    the *parent's* target row, so candidates have to be built per parent.
    """

    def __init__(self, num_nodes: int, width: int):
        children: list[list[int]] = [[] for _ in range(num_nodes)]
        self.parent = [0] * num_nodes
        self.rank = [0] * num_nodes
        node_depth = [0] * num_nodes

        frontier = [0]
        next_id = 1
        while frontier and next_id < num_nodes:
            p = frontier.pop(0)
            for slot in range(width):
                if next_id >= num_nodes:
                    break
                children[p].append(next_id)
                self.parent[next_id] = p
                self.rank[next_id] = slot
                node_depth[next_id] = node_depth[p] + 1
                frontier.append(next_id)
                next_id += 1

        self.next_token = [-1] * num_nodes
        self.next_sibling = [-1] * num_nodes
        for p, kids in enumerate(children):
            if kids:
                self.next_token[p] = kids[0]
                for older, younger in zip(kids, kids[1:]):
                    self.next_sibling[older] = younger

        self.depth = max(node_depth) + 1
        self.width = width


def make_candidates(
    target_probs: torch.Tensor,
    topology: Topology,
    agreement: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draft candidates the target mostly agrees with.

    Uniformly random token ids sit at p ~ 1/vocab, so every candidate is rejected at
    the root and the traversal the verifier exists for never runs. The verifier
    scores a child against its parent's target row, so siblings are taken as that
    row's top-`width` tokens -- which is also how EAGLE builds them -- and a
    fraction is corrupted to model draft/target drift.
    """
    batch_size, num_draft_tokens, vocab_size = target_probs.shape
    top_tokens = target_probs.topk(max(topology.width, 1), dim=-1).indices

    parent = torch.tensor(topology.parent, dtype=torch.int64, device=DEVICE)
    rank = torch.tensor(topology.rank, dtype=torch.int64, device=DEVICE)
    agreed = top_tokens[:, parent, rank]

    corrupted = torch.randint(
        0,
        vocab_size,
        (batch_size, num_draft_tokens),
        dtype=torch.int64,
        device=DEVICE,
        generator=generator,
    )
    agrees = (
        torch.rand((batch_size, num_draft_tokens), device=DEVICE, generator=generator)
        < agreement
    )
    return torch.where(agrees, agreed, corrupted)


def make_inputs(
    *,
    batch_size: int,
    num_draft_tokens: int,
    width: int,
    vocab_size: int,
    seed: int = 0,
    agreement: float = 0.7,
    logit_scale: float = 6.0,
) -> dict:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    tree = Topology(num_draft_tokens, width)

    def per_request(values):
        return (
            torch.tensor(values, dtype=torch.int64, device=DEVICE)
            .expand(batch_size, -1)
            .contiguous()
        )

    # Peaked, like a trained model's next-token distribution. softmax(randn) over a
    # 150K vocabulary is nearly uniform and would make every acceptance test fail.
    target_probs = torch.softmax(
        torch.randn(
            (batch_size, num_draft_tokens, vocab_size),
            dtype=torch.float32,
            device=DEVICE,
            generator=generator,
        )
        * logit_scale,
        dim=-1,
    )

    return dict(
        predicts=torch.full(
            (batch_size * num_draft_tokens,), -1, dtype=torch.int32, device=DEVICE
        ),
        accept_index=torch.full(
            (batch_size, tree.depth), -1, dtype=torch.int32, device=DEVICE
        ),
        accept_token_num=torch.zeros(batch_size, dtype=torch.int32, device=DEVICE),
        candidates=make_candidates(target_probs, tree, agreement, generator),
        retrive_index=torch.arange(
            batch_size * num_draft_tokens, dtype=torch.int64, device=DEVICE
        ).view(batch_size, num_draft_tokens),
        retrive_next_token=per_request(tree.next_token),
        retrive_next_sibling=per_request(tree.next_sibling),
        uniform_samples=torch.rand(
            (batch_size, num_draft_tokens),
            dtype=torch.float32,
            device=DEVICE,
            generator=generator,
        ),
        uniform_samples_for_final_sampling=torch.rand(
            (batch_size,), dtype=torch.float32, device=DEVICE, generator=generator
        ),
        target_probs=target_probs,
        draft_probs=torch.zeros(
            (batch_size, num_draft_tokens, vocab_size),
            dtype=torch.float32,
            device=DEVICE,
        ),
    )


def time_kernel(kernel, args: dict, iters: int) -> tuple[float, float]:
    """Median milliseconds, and the mean accepted draft length."""

    def call():
        kernel(**args, threshold_single=1.0, threshold_acc=1.0, deterministic=True)

    for _ in range(3):
        args["draft_probs"].zero_()
        call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        args["draft_probs"].zero_()
        torch.cuda.synchronize()
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), args["accept_token_num"].float().mean().item()


def latency_samples(fn, iters: int, prepare=None) -> list[float]:
    for _ in range(3):
        if prepare is not None:
            prepare()
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        if prepare is not None:
            prepare()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def load_capture_requests(capture_dir: Path):
    requests = []
    for path in sorted(capture_dir.glob("rank0_*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        for index, acceptance_length in enumerate(
            record["metadata"]["acceptance_length"]
        ):
            requests.append(
                {
                    "target_probs": record["renormalized_probs"][index],
                    "candidates": record["candidates"][index],
                    "next_token": record["retrieve_next_token"][index],
                    "next_sibling": record["retrieve_next_sibling"][index],
                    "coins": record["uniform_samples"][index],
                    "final_coin": record["uniform_samples_for_final_sampling"][index],
                    "acceptance_length": acceptance_length,
                    "max_tree_depth": record["metadata"]["max_tree_depth"],
                }
            )
    if not requests:
        raise ValueError(f"No rank0_*.pt captures found under {capture_dir}")
    return requests


def make_captured_inputs(requests, batch_size: int):
    selected = [requests[index % len(requests)] for index in range(batch_size)]
    num_draft_tokens = selected[0]["candidates"].numel()
    max_tree_depth = selected[0]["max_tree_depth"]
    return (
        {
            "predicts": torch.full(
                (batch_size * num_draft_tokens,),
                -1,
                dtype=torch.int32,
                device=DEVICE,
            ),
            "accept_index": torch.full(
                (batch_size, max_tree_depth),
                -1,
                dtype=torch.int32,
                device=DEVICE,
            ),
            "accept_token_num": torch.zeros(
                batch_size, dtype=torch.int32, device=DEVICE
            ),
            "candidates": torch.stack([x["candidates"] for x in selected]).to(DEVICE),
            "retrive_index": torch.arange(
                batch_size * num_draft_tokens, dtype=torch.int64, device=DEVICE
            ).view(batch_size, num_draft_tokens),
            "retrive_next_token": torch.stack([x["next_token"] for x in selected]).to(
                DEVICE
            ),
            "retrive_next_sibling": torch.stack(
                [x["next_sibling"] for x in selected]
            ).to(DEVICE),
            "uniform_samples": torch.stack([x["coins"] for x in selected]).to(DEVICE),
            "uniform_samples_for_final_sampling": torch.stack(
                [x["final_coin"] for x in selected]
            ).to(DEVICE),
            "target_probs": torch.stack([x["target_probs"] for x in selected]).to(
                DEVICE
            ),
            "draft_probs": torch.zeros(
                (
                    batch_size,
                    num_draft_tokens,
                    selected[0]["target_probs"].shape[-1],
                ),
                dtype=torch.float32,
                device=DEVICE,
            ),
        },
        torch.tensor(
            [x["acceptance_length"] for x in selected],
            dtype=torch.int32,
            device=DEVICE,
        ),
    )


def run_capture_bench(args, aot) -> None:
    requests = load_capture_requests(args.capture_dir)
    output = {
        "device": torch.cuda.get_device_name(0),
        "capture_dir": str(args.capture_dir),
        "captured_requests": len(requests),
        "rows": [],
    }
    print(
        f"{output['device']}  capture_requests={len(requests)}  " f"iters={args.iters}"
    )
    print(
        f"{'bs':>5} {'tree_p50':>10} {'tree_p95':>10} {'accept':>8} "
        f"{'move_slots':>10} {'index/1L':>10} {'old/78L':>10} "
        f"{'batch/78L':>10} {'speedup':>8}"
    )

    for batch_size in args.batches:
        inputs, expected_acceptance = make_captured_inputs(requests, batch_size)

        def call_tree():
            tree_speculative_sampling_target_only_triton(
                **inputs,
                threshold_single=1.0,
                threshold_acc=1.0,
                deterministic=True,
            )

        inputs["draft_probs"].zero_()
        call_tree()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            inputs["accept_token_num"], expected_acceptance, rtol=0, atol=0
        )
        tree_samples = latency_samples(
            call_tree, args.iters, prepare=inputs["draft_probs"].zero_
        )

        moved_slots = batch_size * inputs["candidates"].shape[1]
        pool = SimpleNamespace(
            page_size=64,
            index_head_dim=128,
            quant_block_size=128,
        )
        src_loc = torch.arange(moved_slots, device=DEVICE, dtype=torch.int64) + 64
        tgt_loc = src_loc + moved_slots + 64
        num_pages = int(tgt_loc.max().item() // pool.page_size + 1)
        index_buffer = torch.zeros(
            (num_pages, pool.page_size * (pool.index_head_dim + 4)),
            dtype=torch.uint8,
            device=DEVICE,
        )
        scratch = torch.empty(
            (moved_slots, pool.index_head_dim + 4),
            dtype=torch.uint8,
            device=DEVICE,
        )
        move_samples = latency_samples(
            lambda: MoveKAndS.execute(
                pool,
                index_buffer,
                tgt_loc=tgt_loc,
                src_loc=src_loc,
                scratch=scratch,
            ),
            args.iters,
        )
        full_pool = object.__new__(DSATokenToKVPool)
        full_pool.page_size = pool.page_size
        full_pool.index_head_dim = pool.index_head_dim
        full_pool.quant_block_size = pool.quant_block_size
        full_pool.size = int(tgt_loc.max().item()) + 1
        full_pool.kv_buffer = [
            torch.zeros(
                (full_pool.size + full_pool.page_size, 1, 656),
                dtype=torch.uint8,
                device=DEVICE,
            )
            for _ in range(78)
        ]
        full_pool.index_k_with_scale_buffer = [
            torch.zeros_like(index_buffer) for _ in range(78)
        ]
        full_pool._init_dsa_move_metadata()

        def old_full_move():
            for kv_cache in full_pool.kv_buffer:
                kv_cache[tgt_loc] = kv_cache[src_loc]
            full_pool._move_index_k_cache(tgt_loc, src_loc)

        old_full_move_samples = latency_samples(
            old_full_move,
            args.iters,
        )
        full_pool.move_kv_cache(tgt_loc, src_loc)
        batched_full_move_samples = latency_samples(
            lambda: full_pool.move_kv_cache(tgt_loc, src_loc),
            args.iters,
        )

        row = {
            "batch_size": batch_size,
            "num_draft_tokens": inputs["candidates"].shape[1],
            "tree_p50_ms": statistics.median(tree_samples),
            "tree_p95_ms": percentile(tree_samples, 0.95),
            "acceptance_length_mean": float(expected_acceptance.float().mean()),
            "relocation_slots": moved_slots,
            "relocation_p50_ms_per_layer": statistics.median(move_samples),
            "relocation_p95_ms_per_layer": percentile(move_samples, 0.95),
            "old_full_relocation_p50_ms_78_layers": statistics.median(
                old_full_move_samples
            ),
            "old_full_relocation_p95_ms_78_layers": percentile(
                old_full_move_samples, 0.95
            ),
            "batched_full_relocation_p50_ms_78_layers": statistics.median(
                batched_full_move_samples
            ),
            "batched_full_relocation_p95_ms_78_layers": percentile(
                batched_full_move_samples, 0.95
            ),
        }
        row["batched_full_relocation_speedup"] = (
            row["old_full_relocation_p50_ms_78_layers"]
            / row["batched_full_relocation_p50_ms_78_layers"]
        )
        if aot is not None:
            aot_samples = latency_samples(
                lambda: aot(
                    **inputs,
                    threshold_single=1.0,
                    threshold_acc=1.0,
                    deterministic=True,
                ),
                args.iters,
                prepare=inputs["draft_probs"].zero_,
            )
            row["aot_tree_p50_ms"] = statistics.median(aot_samples)
        output["rows"].append(row)
        print(
            f"{batch_size:>5} {row['tree_p50_ms']:>10.4f} "
            f"{row['tree_p95_ms']:>10.4f} "
            f"{row['acceptance_length_mean']:>8.2f} "
            f"{moved_slots:>10} {row['relocation_p50_ms_per_layer']:>10.4f} "
            f"{row['old_full_relocation_p50_ms_78_layers']:>10.3f} "
            f"{row['batched_full_relocation_p50_ms_78_layers']:>10.3f} "
            f"{row['batched_full_relocation_speedup']:>7.2f}x"
        )

    if args.output_json:
        args.output_json.write_text(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=151936, help="GLM-5.2 vocabulary")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--batches", type=int, nargs="+", default=(1, 2, 4, 8, 32, 128, 256)
    )
    parser.add_argument(
        "--agreement",
        type=float,
        default=0.7,
        help="fraction of candidates drawn from the target distribution",
    )
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=12.0,
        help="how peaked the target rows are; drives the accept length",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="sweep logit scale and report accept length only",
    )
    args = parser.parse_args()

    if args.capture_dir:
        run_capture_bench(args, load_aot_kernel())
        return

    if args.calibrate:
        print(f"{torch.cuda.get_device_name(0)}   vocab={args.vocab}")
        print(
            f"{'logit_scale':>12} {'top1_prob':>10} {'width=1':>9} {'width=2':>9} "
            f"{'width=4':>9}   (accept_len, ndt=8, bs=128)"
        )
        for scale in (6.0, 9.0, 12.0, 16.0, 20.0):
            row = f"{scale:>12.1f}"
            top1 = None
            lengths = []
            for width in (1, 2, 4):
                inputs = make_inputs(
                    batch_size=128,
                    num_draft_tokens=8,
                    width=width,
                    vocab_size=args.vocab,
                    agreement=args.agreement,
                    logit_scale=scale,
                )
                if top1 is None:
                    top1 = inputs["target_probs"].max(dim=-1).values.mean().item()
                _, accept = time_kernel(
                    tree_speculative_sampling_target_only_triton, inputs, 3
                )
                lengths.append(accept)
                del inputs
                torch.cuda.empty_cache()
            print(row + f" {top1:>10.3f}" + "".join(f" {v:>9.2f}" for v in lengths))
        return

    aot = load_aot_kernel()
    print(
        f"{torch.cuda.get_device_name(0)}   vocab={args.vocab}   iters={args.iters}"
        f"   agreement={args.agreement}"
    )
    print(
        f"AOT tree kernel: {'available' if aot else 'unavailable (expected on ROCm)'}"
    )

    header = f"{'bs':>5} {'ndt':>4} {'width':>6} {'depth':>6} {'triton(ms)':>11}"
    if aot:
        header += f" {'aot(ms)':>9} {'triton/aot':>11}"
    header += f" {'accept_len':>11}"

    for num_draft_tokens in (8, 16):
        print(f"\n--- num_draft_tokens={num_draft_tokens} ---")
        print(header)
        for batch_size in (1, 8, 32, 128, 256):
            for width in (1, 2, 4):
                inputs = make_inputs(
                    batch_size=batch_size,
                    num_draft_tokens=num_draft_tokens,
                    width=width,
                    vocab_size=args.vocab,
                    agreement=args.agreement,
                    logit_scale=args.logit_scale,
                )
                depth = inputs["accept_index"].shape[1]
                t_triton, accept = time_kernel(
                    tree_speculative_sampling_target_only_triton, inputs, args.iters
                )
                row = (
                    f"{batch_size:>5} {num_draft_tokens:>4} {width:>6} {depth:>6} "
                    f"{t_triton:>11.4f}"
                )
                if aot:
                    t_aot, _ = time_kernel(aot, inputs, args.iters)
                    row += f" {t_aot:>9.4f} {t_triton / t_aot:>10.2f}x"
                row += f" {accept:>11.2f}"
                print(row)
                del inputs
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
