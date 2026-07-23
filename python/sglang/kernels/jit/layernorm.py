"""LayerNorm (mean-subtracted, affine weight + bias) JIT kernel.

Drop-in for the DSA indexer's k_norm (head_dim=128, bf16) on gfx950, replacing
aiter's ck_tile Layernorm2dFwd. Reduction + affine computed in fp32, rounded to
the activation dtype on store (matches torch F.layer_norm / aiter semantics).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_CTA_BLOCK_SIZE = 512
_WARP_SIZE = 32


def is_supported_layernorm_hidden_size(hidden_size: int) -> bool:
    """Return True iff the JIT layernorm kernel supports this hidden size.

    Two launch configs cover the practical range:
      - Warp kernel: ``[32, 512)`` in multiples of 32 (indexer k_norm=128).
      - CTA kernel: ``>= 512`` in multiples of 512 (token norms).
    """
    if _WARP_SIZE <= hidden_size < _CTA_BLOCK_SIZE and hidden_size % _WARP_SIZE == 0:
        return True
    return hidden_size >= _CTA_BLOCK_SIZE and hidden_size % _CTA_BLOCK_SIZE == 0


@cache_once
def _jit_layernorm_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    kernel_cls = (
        "LayerNormWarpKernel" if hidden_size < _CTA_BLOCK_SIZE else "LayerNormKernel"
    )
    return load_jit(
        "layernorm",
        *args,
        cuda_files=["elementwise/layernorm.cuh"],
        cuda_wrappers=[("layernorm", f"{kernel_cls}<{args}>::run")],
    )


def layernorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """LayerNorm: ``out = (x - mean) * rsqrt(var + eps) * weight + bias``.

    ``input`` must be 2D ``(num_tokens, hidden_size)``; callers with higher-rank
    tensors should reshape first. ``hidden_size`` must satisfy
    :func:`is_supported_layernorm_hidden_size`. Empty inputs return an empty
    output without launching the kernel.
    """
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"layernorm: input must be fp16 or bf16, got {input.dtype}")
    if input.dim() != 2:
        raise RuntimeError(f"layernorm: input must be 2D, got {input.dim()}D")
    hidden_size = input.size(-1)
    if not is_supported_layernorm_hidden_size(hidden_size):
        raise RuntimeError(
            f"layernorm: unsupported hidden_size={hidden_size} "
            f"(must be a multiple of {_WARP_SIZE} in [{_WARP_SIZE}, {_CTA_BLOCK_SIZE}) "
            f"or a multiple of {_CTA_BLOCK_SIZE})"
        )
    if out is None:
        out = torch.empty_like(input)
    if input.numel() == 0:
        return out
    module = _jit_layernorm_module(hidden_size, input.dtype)
    module.layernorm(input, weight, bias, out, eps)
    return out
