from __future__ import annotations

import functools
import os

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip


@functools.lru_cache(maxsize=1)
def is_unified_kv_triton() -> bool:
    # unified_kv_triton is only implemented on HIP (ROCm)
    return is_hip() and envs.SGLANG_HACK_FLASHMLA_BACKEND.get() == "unified_kv_triton"


# Truthy spellings accepted for the opt-in fp8 env var below.
_TRUTHY = ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=1)
def unified_kv_use_fp8() -> bool:
    """Whether the unified KV attention pool should use fp8 1x64 block-scale storage.

    Strictly opt-in and default OFF. Returns True only when ALL hold:
      (a) running on HIP (ROCm),
      (b) the unified_kv_triton backend is active (``is_unified_kv_triton()``),
      (c) ``SGLANG_UNIFIED_KV_FP8`` is set truthy (one of: 1/true/yes/on).

    This is deliberately decoupled from ``--kv-cache-dtype fp8_e4m3``: that flag
    only fp8s the indexer pool, so it must NOT by itself flip the unified
    attention pool to fp8 (the quant-on-store / read paths are not wired yet).
    """
    # ``is_unified_kv_triton()`` already requires HIP, so this implies is_hip().
    if not is_unified_kv_triton():
        return False
    return os.environ.get("SGLANG_UNIFIED_KV_FP8", "").strip().lower() in _TRUTHY
