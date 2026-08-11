from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from sglang.srt.environ import envs

_CAPTURE: Optional["TreeProfileCapture"] = None


def _cpu_copy(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", copy=True)


def _select_per_request(
    value: Any, request_indices: torch.Tensor, batch_size: int
) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    flat = value.detach().reshape(batch_size, -1)
    return _cpu_copy(flat[request_indices])


def _is_primary_process() -> bool:
    return not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    )


@dataclass
class PendingTreeProfileCapture:
    call_index: int
    batch_size: int
    batch_bucket: int
    num_draft_tokens: int
    request_indices: torch.Tensor
    softmax_probs: torch.Tensor
    candidates: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    top_ks: Any
    top_ps: Any
    temperatures: Any
    tree_width: int
    max_tree_depth: int
    top_p_input_probs: Optional[torch.Tensor] = None


class TreeProfileCapture:
    def __init__(self, output_dir: str, max_rows: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_rows = max_rows
        self.saved_rows = 0
        self.call_index = 0
        self.last_batch_bucket: Optional[int] = None

    @staticmethod
    def _batch_bucket(batch_size: int) -> int:
        return 1 << (batch_size - 1).bit_length()

    def _should_capture(self, call_index: int, batch_bucket: int) -> bool:
        is_power_of_two = call_index == 0 or not (call_index & (call_index - 1))
        return is_power_of_two or batch_bucket != self.last_batch_bucket

    def begin(
        self,
        *,
        softmax_probs: torch.Tensor,
        candidates: torch.Tensor,
        verify_input: Any,
        sampling_info: Any,
    ) -> Optional[PendingTreeProfileCapture]:
        batch_size, num_draft_tokens, _ = softmax_probs.shape
        call_index = self.call_index
        self.call_index += 1
        batch_bucket = self._batch_bucket(batch_size)
        should_capture = self._should_capture(call_index, batch_bucket)
        self.last_batch_bucket = batch_bucket

        remaining_rows = self.max_rows - self.saved_rows
        max_requests = remaining_rows // num_draft_tokens
        if not should_capture or max_requests <= 0:
            return None

        num_requests = min(2, batch_size, max_requests)
        if num_requests == 1:
            request_indices = torch.tensor(
                [call_index % batch_size],
                dtype=torch.int64,
                device=softmax_probs.device,
            )
        else:
            request_indices = torch.linspace(
                0,
                batch_size - 1,
                num_requests,
                dtype=torch.int64,
                device=softmax_probs.device,
            )

        self.saved_rows += num_requests * num_draft_tokens
        return PendingTreeProfileCapture(
            call_index=call_index,
            batch_size=batch_size,
            batch_bucket=batch_bucket,
            num_draft_tokens=num_draft_tokens,
            request_indices=request_indices,
            softmax_probs=_cpu_copy(softmax_probs[request_indices]),
            candidates=_cpu_copy(candidates[request_indices]),
            retrieve_index=_cpu_copy(verify_input.retrieve_index[request_indices]),
            retrieve_next_token=_cpu_copy(
                verify_input.retrieve_next_token[request_indices]
            ),
            retrieve_next_sibling=_cpu_copy(
                verify_input.retrieve_next_sibling[request_indices]
            ),
            top_ks=_select_per_request(
                sampling_info.top_ks, request_indices, batch_size
            ),
            top_ps=_select_per_request(
                sampling_info.top_ps, request_indices, batch_size
            ),
            temperatures=_select_per_request(
                sampling_info.temperatures, request_indices, batch_size
            ),
            tree_width=int(verify_input.tree_topk),
            max_tree_depth=int(verify_input.max_tree_depth),
        )

    def set_top_p_input(
        self,
        pending: Optional[PendingTreeProfileCapture],
        top_p_input_probs: torch.Tensor,
    ) -> None:
        if pending is not None:
            pending.top_p_input_probs = _cpu_copy(
                top_p_input_probs[pending.request_indices]
            )

    def finish(
        self,
        pending: Optional[PendingTreeProfileCapture],
        *,
        renormalized_probs: torch.Tensor,
        coins: torch.Tensor,
        final_coins: torch.Tensor,
        accept_token_num: torch.Tensor,
    ) -> None:
        if pending is None:
            return

        request_indices = pending.request_indices
        nucleus_input = (
            pending.top_p_input_probs
            if pending.top_p_input_probs is not None
            else pending.softmax_probs
        )
        flat_nucleus_input = nucleus_input.flatten(0, 1)
        top_ps = pending.top_ps
        if isinstance(top_ps, torch.Tensor):
            row_top_ps = top_ps.reshape(-1).repeat_interleave(
                pending.num_draft_tokens
            )
        else:
            row_top_ps = torch.full(
                (flat_nucleus_input.shape[0],), float(top_ps), dtype=torch.float32
            )

        sorted_probs = flat_nucleus_input.sort(dim=-1, descending=True).values
        budgets = flat_nucleus_input.sum(dim=-1) - (1.0 - row_top_ps)
        nucleus_sizes = (
            (sorted_probs.cumsum(dim=-1) - sorted_probs) <= budgets.unsqueeze(1)
        ).sum(dim=-1)
        max_token_probability = pending.softmax_probs.amax(dim=-1).flatten()

        metadata = {
            "call_index": pending.call_index,
            "batch_size": pending.batch_size,
            "batch_bucket": pending.batch_bucket,
            "num_draft_tokens": pending.num_draft_tokens,
            "captured_rows": int(flat_nucleus_input.shape[0]),
            "selected_request_indices": request_indices.tolist(),
            "tree_width": pending.tree_width,
            "max_tree_depth": pending.max_tree_depth,
            "max_token_probability": max_token_probability.tolist(),
            "nucleus_size": nucleus_sizes.tolist(),
            "prefix_overflow": (nucleus_sizes > 4096).tolist(),
            "acceptance_length": _cpu_copy(
                accept_token_num[request_indices]
            ).tolist(),
        }
        record = {
            "metadata": metadata,
            "softmax_probs": pending.softmax_probs,
            "top_p_input_probs": nucleus_input,
            "renormalized_probs": _cpu_copy(
                renormalized_probs[request_indices]
            ),
            "candidates": pending.candidates,
            "retrieve_index": pending.retrieve_index,
            "retrieve_next_token": pending.retrieve_next_token,
            "retrieve_next_sibling": pending.retrieve_next_sibling,
            "uniform_samples": _cpu_copy(coins[request_indices]),
            "uniform_samples_for_final_sampling": _cpu_copy(
                final_coins[request_indices]
            ),
            "top_ks": pending.top_ks,
            "top_ps": pending.top_ps,
            "temperatures": pending.temperatures,
        }

        stem = (
            f"rank0_call{pending.call_index:08d}_"
            f"bs{pending.batch_size}_rows{metadata['captured_rows']}"
        )
        output_path = self.output_dir / f"{stem}.pt"
        temporary_path = self.output_dir / f".{stem}.{os.getpid()}.tmp"
        torch.save(record, temporary_path)
        os.replace(temporary_path, output_path)
        metadata["file"] = output_path.name
        with (self.output_dir / "manifest.jsonl").open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(metadata, sort_keys=True) + "\n")


def get_tree_profile_capture() -> Optional[TreeProfileCapture]:
    global _CAPTURE

    output_dir = envs.SGLANG_DEBUG_SPEC_PROBS_CAPTURE_DIR.get()
    max_rows = envs.SGLANG_DEBUG_SPEC_PROBS_CAPTURE_MAX_ROWS.get()
    if not output_dir or max_rows <= 0 or not _is_primary_process():
        return None
    if _CAPTURE is None:
        _CAPTURE = TreeProfileCapture(output_dir, max_rows)
    return _CAPTURE
