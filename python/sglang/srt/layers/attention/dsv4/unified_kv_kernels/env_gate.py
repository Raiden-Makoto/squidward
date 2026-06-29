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
    """Whether the unified KV attention pool should use the upstream aiter MXFP8
    (E8M0 per-32-block NoPE-fp8 / RoPE-bf16) storage layout (#3751).

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


@functools.lru_cache(maxsize=1)
def unified_kv_fp8_qk_native() -> bool:
    """Whether the fp8 unified-KV decode should run the QK dot as a native CDNA4
    MXFP8 scaled MFMA (``tl.dot_scaled`` over the stored E8M0 NoPE pack) instead
    of dequantizing the KV tile to bf16 first.

    Strictly opt-in and default OFF. Returns True only when BOTH hold:
      (a) the fp8 unified-KV pool is active (``unified_kv_use_fp8()``),
      (b) ``SGLANG_UNIFIED_KV_FP8_QK_NATIVE`` is set truthy.

    gfx950-only in practice: the MXFP8 NoPE pack itself asserts gfx950 (e4m3fn),
    so this never engages on gfx942.
    """
    if not unified_kv_use_fp8():
        return False
    return envs.SGLANG_UNIFIED_KV_FP8_QK_NATIVE.get()


@functools.lru_cache(maxsize=1)
def unified_kv_fp8_pv_native() -> bool:
    """Whether the fp8 unified-KV decode should run the PV dot natively in fp8
    (single fp8 MFMA over the stored NoPE pack + a small bf16 RoPE dot) instead
    of reconstructing the bf16 KV tile for an fp32/bf16 PV dot.

    Strictly opt-in and default OFF. Returns True only when BOTH hold:
      (a) the fp8 unified-KV pool is active (``unified_kv_use_fp8()``),
      (b) ``SGLANG_UNIFIED_KV_FP8_PV_NATIVE`` is set truthy.

    PV-native implies QK-native: both consume the same single fp8 tile read, so
    the score path runs as a native MXFP8 ``dot_scaled`` rather than dequanting.
    gfx950-only in practice (the MXFP8 NoPE pack asserts gfx950 / e4m3fn).
    """
    if not unified_kv_use_fp8():
        return False
    return envs.SGLANG_UNIFIED_KV_FP8_PV_NATIVE.get()
