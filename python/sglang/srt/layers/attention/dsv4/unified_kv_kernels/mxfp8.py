# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Canonical MXFP8 (E8M0 per-32-block) NoPE pack/dequant for the DSv4 unified KV.

This is the single source of truth for the on-disk fp8 layout the upstream aiter
``pa_sparse_prefill_fp8_opus`` op (ROCm/aiter #3751, commit ``bd003d9d``) consumes
and our decode kernel reads. It mirrors the reference packer
``op_tests/test_pa_sparse_prefill_opus.py::_quantize_nope`` byte-for-byte (verified
bit-identical on gfx950).

Per-token head dim is 512 = 448 NoPE + 64 RoPE. The NoPE stream is packed; the
RoPE stream stays bf16 in a separate buffer.

NoPE packed row layout (512 fp8 slots = 512 bytes):
    [   0 : 448 ]  fp8 (e4m3fn) quantized NoPE values
    [ 448 : 462 ]  14 uint8 E8M0 block exponents (one per 32-element block)
    [ 462 : 512 ]  50 bytes zero pad

Per 32-element block ``b`` the dequant scale is ``2 ** (E8M0[b] - 127)``; the
exponent is chosen as ``ceil(log2(amax_b / 448.0))`` so the block max maps into
the e4m3fn finite range (overflow -> NaN on cast is avoided), clamped to uint8.
"""

from __future__ import annotations

from typing import Tuple

import torch

# NoPE / RoPE split of the 512-d norm+rope'd K head.
DIM_NOPE = 448
DIM_ROPE = 64
DIM_HEAD = DIM_NOPE + DIM_ROPE  # 512

# MXFP8 block geometry.
FP8_BLOCK = 32
NUM_NOPE_BLOCKS = DIM_NOPE // FP8_BLOCK  # 14
# Packed NoPE row width in fp8 slots: 448 data + 14 E8M0 + 50 pad = 512.
NOPE_PACKED_WIDTH = 512
SCALE_OFFSET = DIM_NOPE  # E8M0 bytes start here within the packed row
PAD_OFFSET = DIM_NOPE + NUM_NOPE_BLOCKS  # zero pad starts here (462)

# e4m3fn max normal. The packed format is gfx950-only (e4m3fn); the op itself
# asserts gfx950, so we do NOT use the e4m3fnuz arch variant here.
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
E8M0_BIAS = 127


def pack_nope_mxfp8(real: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Quantize ``[..., 448]`` real NoPE values into a packed ``[..., 512]`` fp8
    row (NoPE fp8 + E8M0 block scales + zero pad).

    Returns the packed tensor viewed as ``float8_e4m3fn``. Accepts arbitrary
    leading dims (flattened internally).
    """
    if real.shape[-1] != DIM_NOPE:
        raise ValueError(f"pack_nope_mxfp8 expects last dim {DIM_NOPE}, got {real.shape[-1]}")
    lead = real.shape[:-1]
    r = int(torch.tensor(lead).prod().item()) if lead else 1
    flat = real.reshape(r, DIM_NOPE)

    blk = flat.reshape(r, NUM_NOPE_BLOCKS, FP8_BLOCK).to(torch.float32)
    amax = blk.abs().amax(dim=-1)  # [r, NBLK]
    e_unbiased = torch.ceil(torch.log2(amax.clamp(min=1e-30) / FP8_MAX)).to(torch.int32)
    e_unbiased = torch.where(amax == 0, torch.zeros_like(e_unbiased), e_unbiased)
    e_byte = (e_unbiased + E8M0_BIAS).clamp(0, 255).to(torch.uint8)  # [r, NBLK]
    s = torch.exp2(e_unbiased.to(torch.float32)).unsqueeze(-1)  # [r, NBLK, 1]
    q = (blk / s).to(FP8_DTYPE)  # [r, NBLK, BLOCK]

    if out is None:
        packed = torch.zeros(r, NOPE_PACKED_WIDTH, dtype=torch.uint8, device=real.device)
    else:
        packed = out.reshape(r, NOPE_PACKED_WIDTH).view(torch.uint8)
        packed.zero_()
    packed[:, :DIM_NOPE] = q.reshape(r, DIM_NOPE).view(torch.uint8)
    packed[:, SCALE_OFFSET : SCALE_OFFSET + NUM_NOPE_BLOCKS] = e_byte
    return packed.view(FP8_DTYPE).reshape(*lead, NOPE_PACKED_WIDTH)


def dequant_nope_mxfp8(packed: torch.Tensor) -> torch.Tensor:
    """Dequant a packed ``[..., 512]`` fp8 NoPE row back to ``[..., 448]`` fp32.

    Reference path used by tests and the PyTorch fallback; the decode Triton
    kernel performs the equivalent dequant inline.
    """
    lead = packed.shape[:-1]
    r = int(torch.tensor(lead).prod().item()) if lead else 1
    b = packed.reshape(r, NOPE_PACKED_WIDTH).view(torch.uint8)
    q = b[:, :DIM_NOPE].view(FP8_DTYPE).to(torch.float32)  # [r, 448]
    e_byte = b[:, SCALE_OFFSET : SCALE_OFFSET + NUM_NOPE_BLOCKS].to(torch.int32)  # [r,14]
    s = torch.exp2((e_byte - E8M0_BIAS).to(torch.float32))  # [r, 14]
    deq = (q.reshape(r, NUM_NOPE_BLOCKS, FP8_BLOCK) * s.unsqueeze(-1)).reshape(r, DIM_NOPE)
    return deq.reshape(*lead, DIM_NOPE)


def split_nope_rope(k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a ``[..., 512]`` norm+rope'd K into ``(nope[...,448], rope[...,64])``."""
    if k.shape[-1] != DIM_HEAD:
        raise ValueError(f"split_nope_rope expects last dim {DIM_HEAD}, got {k.shape[-1]}")
    return k[..., :DIM_NOPE], k[..., DIM_NOPE:]
