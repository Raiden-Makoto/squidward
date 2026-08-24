from unittest import mock

import torch

from sglang.kernels.ops.attention.dsa import triton_sparse_mla
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def _inputs(seq=3, heads=16, topk=128):
    q_nope = torch.empty(seq, heads, 512, dtype=torch.float8_e4m3fn)
    q_rope = torch.empty(seq, heads, 64, dtype=torch.float8_e4m3fn)
    kv = torch.empty(256, 1, 576, dtype=torch.float8_e4m3fn)
    indices = torch.zeros(seq, 1, topk, dtype=torch.int32)
    return q_nope, q_rope, kv, indices


def test_packed_output_abi_is_head_major_contiguous():
    q_nope, _, _, _ = _inputs(seq=5, heads=16)
    values, scales = triton_sparse_mla._allocate_output(q_nope, 5, 16, 512, True)

    assert values.shape == (16, 5, 256)
    assert values.stride() == (1280, 256, 1)
    assert values.dtype == torch.uint8
    assert scales is not None
    assert scales.shape == (16, 5, 16)
    assert scales.stride() == (80, 16, 1)
    assert scales.dtype == torch.uint8
    formatted = triton_sparse_mla._format_output(values, scales, True)
    assert formatted[0] is values
    assert formatted[1] is scales


def test_direct_v_up_compile_probe_uses_production_chunk_geometry():
    normalized = torch.empty(2, 16, 128, dtype=torch.float32)
    weight = torch.empty(16, 64, 256, dtype=torch.uint8)
    weight_scale = torch.empty(16, 4, 256, dtype=torch.uint8)
    kernel = _FakeKernel()
    with mock.patch.object(
        triton_sparse_mla, "_direct_v_up_dot_scaled_compile_probe_kernel", kernel
    ):
        output = triton_sparse_mla.direct_v_up_dot_scaled_compile_probe(
            normalized, weight, weight_scale
        )

    grid, launch_args, launch_kwargs = kernel.calls[0]
    assert grid == (2, 16, 16)
    assert launch_args[0] is normalized
    assert launch_args[1] is weight
    assert launch_args[2] is weight_scale
    assert launch_args[3] is output
    assert launch_kwargs["H"] == 16
    assert launch_kwargs["N"] == 256
    assert launch_kwargs["BLOCK_N"] == 16
    assert output.shape == (2, 16, 256)
    assert output.dtype == torch.bfloat16


def test_long_single_pass_requests_direct_packed_store():
    args = _inputs()
    kernel = _FakeKernel()
    with mock.patch.object(
        triton_sparse_mla, "_sparse_mla_fwd_split_dim_kernel", kernel
    ):
        values, scales = triton_sparse_mla._triton_sparse_mla_fwd_single(
            *args, sm_scale=0.125, return_mxfp4=True
        )

    _, launch_args, launch_kwargs = kernel.calls[0]
    assert launch_args[4] is values
    assert launch_args[5] is scales
    assert launch_kwargs["RETURN_MXFP4"] is True
    assert launch_kwargs["T"] == 3


def test_short_fused_requests_direct_packed_store():
    args = _inputs()
    kernel = _FakeKernel()
    with mock.patch.object(triton_sparse_mla, "_sparse_mla_fused_kernel", kernel):
        values, scales = triton_sparse_mla._triton_sparse_mla_fwd_splitk(
            *args, sm_scale=0.125, d_v=512, kv_splits=1, return_mxfp4=True
        )

    _, launch_args, launch_kwargs = kernel.calls[0]
    assert launch_args[4] is values
    assert launch_args[5] is scales
    assert launch_kwargs["RETURN_MXFP4"] is True


def test_split_k_keeps_bf16_partials_and_packs_only_final_reduce():
    args = _inputs(topk=256)
    partial_kernel = _FakeKernel()
    reduce_kernel = _FakeKernel()
    with (
        mock.patch.object(
            triton_sparse_mla, "_sparse_mla_split_k_kernel", partial_kernel
        ),
        mock.patch.object(
            triton_sparse_mla, "_sparse_mla_reduce_kernel", reduce_kernel
        ),
    ):
        values, scales = triton_sparse_mla._triton_sparse_mla_fwd_splitk(
            *args, sm_scale=0.125, d_v=512, kv_splits=2, return_mxfp4=True
        )

    _, partial_args, _ = partial_kernel.calls[0]
    assert partial_args[5].dtype == torch.bfloat16
    _, reduce_args, reduce_kwargs = reduce_kernel.calls[0]
    assert reduce_args[2] is values
    assert reduce_args[3] is scales
    assert reduce_kwargs["D_CHUNK"] == 32
    assert reduce_kwargs["RETURN_MXFP4"] is True


def test_public_gate_preserves_bf16_fallback_for_unsupported_geometry():
    args = _inputs(heads=8)
    sentinel = torch.empty(1, 3, 8, 512, dtype=torch.bfloat16)
    with (
        mock.patch.object(triton_sparse_mla, "_cu_count", return_value=1),
        mock.patch.object(
            triton_sparse_mla,
            "_triton_sparse_mla_fwd_single",
            return_value=sentinel,
        ) as single,
    ):
        result = triton_sparse_mla.triton_sparse_mla_fwd(
            *args, sm_scale=0.125, return_mxfp4=True
        )

    assert result is sentinel
    assert single.call_args.args[-1] is False


def test_public_default_preserves_bf16_fallback_for_supported_geometry():
    args = _inputs(heads=16)
    sentinel = torch.empty(1, 3, 16, 512, dtype=torch.bfloat16)
    with (
        mock.patch.object(triton_sparse_mla, "_cu_count", return_value=1),
        mock.patch.object(
            triton_sparse_mla,
            "_triton_sparse_mla_fwd_single",
            return_value=sentinel,
        ) as single,
    ):
        result = triton_sparse_mla.triton_sparse_mla_fwd(*args, sm_scale=0.125)

    assert result is sentinel
    assert single.call_args.args[-1] is False
