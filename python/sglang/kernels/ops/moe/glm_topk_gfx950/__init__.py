"""gfx950 one-launch GLM-5.x routing from bf16 router logits: aiter biased_grouped_topk (one group, top-8,
renormalized) and fused_append_shared_experts (slot 8 = shared expert 288) in one kernel, bit-identical to that
pair, written as the [M, 9] fp32 weights / int32 ids the aiter MoE consumes. A build/load failure disables it."""

import ctypes
import logging
import os
import subprocess
from typing import Optional, Tuple

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils.hipcc import HipKernel, build_gfx950_module

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glm_topk.hip")
E, TOPK = 288, 8
_kernels: dict = {}
_disabled = False
logger = logging.getLogger(__name__)


class TArgs(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        (n, ctypes.c_void_p) for n in ("logits", "bias", "ids", "wts", "n_valid")
    ] + [
        ("tok", ctypes.c_int32),
        ("stride", ctypes.c_int32),
        ("rsf", ctypes.c_float),
        ("shared_w", ctypes.c_float),
    ]


def _load() -> bool:
    global _disabled
    if _kernels:
        return True
    if _disabled or torch.cuda.is_current_stream_capturing():
        return False
    try:
        mod = build_gfx950_module(_SRC, "sglang_glm_topk_")
        _kernels.update({nw: HipKernel(mod, f"glm_topk_w{nw}") for nw in (1, 4)})
    except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
        logger.warning("glm_topk disabled: %s", e)
        _disabled = True
        return False
    return True


def glm_topk_supported(
    router_logits: torch.Tensor, correction_bias: Optional[torch.Tensor]
) -> bool:
    hip = current_arch().hip
    return (
        not _disabled
        and hip is not None
        and hip.gcn_arch == "gfx950"
        and router_logits.dim() == 2
        and router_logits.dtype == torch.bfloat16
        and router_logits.shape[1] == E
        and router_logits.shape[0] >= 1
        and router_logits.stride(1) == 1
        and correction_bias is not None
        and correction_bias.dtype == torch.bfloat16
        and correction_bias.shape == (E,)
        and correction_bias.is_contiguous()
    )


def glm_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    shared_weight: float,
    num_token_non_padded: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(topk_weights [M, 9] fp32, topk_ids [M, 9] int32), slot 8 = shared expert 288, or None if unavailable
    (build failure, first call under graph capture). Rows from num_token_non_padded (a 1-element int32/int64
    device tensor) on get routed ids 0 and zero weights."""
    if not _load():
        return None
    tok = router_logits.shape[0]
    ids = torch.empty(tok, TOPK + 1, dtype=torch.int32, device=router_logits.device)
    wts = torch.empty(tok, TOPK + 1, dtype=torch.float32, device=router_logits.device)
    n_valid = num_token_non_padded
    if n_valid is not None and (
        n_valid.dtype not in (torch.int32, torch.int64)
        or n_valid.device != router_logits.device
    ):
        n_valid = n_valid.to(device=router_logits.device, dtype=torch.int32)
    args = TArgs(
        router_logits.data_ptr(),
        correction_bias.data_ptr(),
        ids.data_ptr(),
        wts.data_ptr(),
        None if n_valid is None else n_valid.data_ptr(),
        tok,
        router_logits.stride(0),
        float(routed_scaling_factor),
        float(shared_weight),
    )
    nw = 1 if tok <= 512 else 4
    _kernels[nw].launch(
        -(-tok // nw), args, torch.cuda.current_stream().cuda_stream, block=64 * nw
    )
    return wts, ids
