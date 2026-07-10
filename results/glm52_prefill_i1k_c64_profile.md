# GLM-5.2 prefill — i1k/o1k conc64, TP4 — MI355X (MXFP4)

GPU-busy/step (graph-on): prefill ~2740 ms (overlap 1.00x); decode 108 ms (1.00x). Workload `random 1024/1024 conc64 --profile-by-stage`; 78 layers × 4 steps captured (312 attention launches). MoE column uses the tuned MoE tile CSV. B200 comparison pending re-capture with the correct DSA (trtllm sparse-MLA) backend.

| Section | MI355X kernel | ms |
| --- | --- | ---: |
| Attention | `_sparse_mla_fwd_split_dim` (triton sparse MLA, #30575) | 515.4 |
| Attention | `_batched_gemm_a8w8` absorb_prepare (#30519) | 155.2 |
| Attention | `_batched_gemm_a8w8` absorb_core (#30519) | 94.0 |
| **Attention subtotal** | | **764.6** |
| MoE | `ck_moe_mxgemm` stage1 | 123.8 |
| MoE | `mfma_moe2_t64x256` (tuned tile) | 166.6 |
| MoE | `moe_reduction_kernel` | 92.7 |
| **MoE subtotal** | | **383.1** |
| Dense GEMM | Tensile ×5 (`unquant.py apply`) | 295.7 |
| **Dense GEMM subtotal** | | **295.7** |
| All-reduce | `ncclDevKernel_Generic` | 944.1 |
| **All-reduce subtotal** | | **944.1** |
| RMSNorm / hadamard-quant fusion (#30715) | `aiter::add_rmsnorm_quant` | 56.2 |
| **RMSNorm subtotal** | | **56.2** |
| **TOTAL** | | **~2444** |
