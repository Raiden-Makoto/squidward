# GLM-5.2 MoE gemm1 K=256

## Configuration

| Field | Value |
| --- | --- |
| Shape | a4w4 MXFP4, model_dim=6144, inter_dim=512, E=257, topk=9, SiLU |
| Hardware | gfx950, 256 CUs, box `smci355-ccs-aus-m12-17`, container `glm5` |
| Worktree | `/sgl-workspace/aiter_silu`, `PYTHONPATH` set to it |
| aiter branch / head | `RM/glm52-mxfp4-moe-ck-gemm1` / `a28c46d33` |
| CK branch / head | `RM/glm52-a4w4-k256-swizzle-fix` / `ebfc8bb74` |
| K=128 candidates | `256x64x128x128_1x4`, `256x128x128x128_1x4`, gufusion_v3 |
| K=256 candidates | `256x64x128x256_1x4`, `256x128x128x256_1x4`, gufusion_v3 and v5 |

## K=256 swizzle fix

| Field | Value |
| --- | --- |
| File | `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_gather_direct_load.hpp` |
| Broken swizzle | `(threadIdx.x % 64) / 8` |
| Fixed swizzle | `(threadIdx.x / block_slice_lengths.At(I0)) % block_slice_lengths.At(I0)` |
| K=128 behavior | Preserved |
| K=256 behavior | Runtime LDS-write swizzle matches the descriptor read swizzle |
| GSM8K after fix | 0.930 versus 0.929 for K=128 |

## Retained CK optimizations

Packed register fragments, reciprocal SiLU, native exp2, raw scale and B offsets. Median isolated
gemm1 `us_stage1`, three alternating full-sweep repetitions on GPU4:

| M | Pre-optimization µs | Retained µs | Delta |
| ---: | ---: | ---: | ---: |
| 1024 | 177.794 | 172.108 | −3.20% |
| 4096 | 308.900 | 307.401 | −0.49% |
| 8192 | 413.738 | 402.673 | −2.67% |
| 16384 | 644.110 | 620.169 | −3.72% |
| 32768 | 1052.040 | 1016.530 | −3.38% |

M=1024 single-counter passes, cumulative effect of all retained changes:

| Metric | Pre-optimization | Retained | Delta |
| --- | ---: | ---: | ---: |
| `SQ_INSTS_VALU` | 8869584 | 5826704 | −34.31% |
| `SQ_INSTS_SALU` | 1265680 | 1189856 | −5.99% |
| `SQ_INSTS_SMEM` | 76816 | 68592 | −10.71% |
| `SQ_INSTS_VALU_FMA_F32` | 1052672 | 0 | −100% |
| `SQ_INSTS_VALU_CVT` | 197376 | 65792 | −66.67% |
| `SQ_INSTS_MFMA` | 3158016 | 3158016 | 0% |

## CK versus FlyDSL

Median isolated gemm1 `us_stage1`, three alternating repetitions, retained CK K128 versus
production FlyDSL:

| M | CK µs | FlyDSL µs | CK delta |
| ---: | ---: | ---: | ---: |
| 1024 | 173.796 | 152.266 | +14.14% |
| 4096 | 306.226 | 273.363 | +12.02% |
| 8192 | 403.027 | 421.324 | −4.34% |
| 16384 | 620.879 | 687.496 | −9.69% |
| 32768 | 1022.100 | 1180.560 | −13.42% |

M=1024 structural differential, CK K128 M64 versus FlyDSL M64 BK256:

| Metric | CK | FlyDSL | CK delta |
| --- | ---: | ---: | ---: |
| Kernel-trace median | 172.401 µs | 151.241 µs | +13.99% |
| `SQ_INSTS_MFMA` | 3158016 | 3158016 | 0% |
| `SQ_INSTS_VALU` | 5826704 | 7383004 | −21.08% |
| `SQ_INSTS_SALU` | 1189856 | 762296 | +56.09% |
| `SQ_INSTS_SMEM` | 68592 | 35648 | +92.41% |
| `SQ_INSTS_VMEM` | 1414528 | 1449324 | −2.40% |
| `SQ_INSTS_LDS` | 888192 | 947604 | −6.27% |
| LDS per block | 32768 B | 55808 B | −41.29% |
| LDSBankConflict | 0.600% | 0.270% | +0.330 pp |
| MemUnitStalled | 0.096% | 0.132% | −0.036 pp |

## Dispatched kernel resources

Read from the rocprof dispatch record, not from code-object metadata: an instance object holds
several template instantiations and only one is launched.

| Arm | VGPR | AGPR | SGPR | LDS | Scratch | median µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 K128 M64 | 88 | 0 | 112 | 32768 B | 0 B | 172.0 |
| v3 K256 M64 | 128 | 0 | 112 | 32768 B | 136 B | 257.6 |
| v5 K256 M64, two stages, `ARegBuf` 2 | 108 | 0 | 112 | 32768 B | 0 B | 173.1 |
| v5 K256 M64, two stages, `ARegBuf` 1 | 84 | 0 | 112 | 32768 B | 0 B | 173.8 |
| v5 K256 M64, single-buffered stages | 84 | 0 | 112 | 32768 B | 0 B | 175.1 |
| FlyDSL M64 BK256 | 100 | 0 | — | 55808 B | 0 B | 150.8 |

## K=256 as a single register stage

gufusion v3 and its byte-identical v5 clone, M=1024, GPU4:

| Arm | median `us_stage1` | VGPR | Scratch |
| --- | ---: | ---: | ---: |
| v3 K128 M64 | 173.219 | 88 | 0 B |
| v3 K256 M64 | 256.498 | 128 | 136 B |
| v5 K256 M64 clone | 256.997 | 128 | 136 B |

Full-sweep K256 versus K128 before any staging work, best `block_m` per arm:

| M | Best K128 µs | Best K256 µs | K256 delta |
| ---: | ---: | ---: | ---: |
| 1024 | 177.068 | 217.045 | +22.6% |
| 4096 | 308.631 | 361.631 | +17.2% |
| 8192 | 421.297 | 490.984 | +16.5% |
| 16384 | 651.215 | 775.001 | +19.0% |
| 32768 | 1073.100 | 1354.160 | +26.2% |

## K=256 streamed as two K128 register stages

v5 stages one K256 tile through LDS and consumes it as two K128 register stages, no barrier between
stages. M=1024, GPU4, three alternating repetitions with K128 re-measured per session:

| v5 variant | CK head | K128 median µs | v5 median µs | v5 delta | v5 VGPR | Scratch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Two K128 stages | `150f06906` | 173.219 | 173.059 | −0.09% | 108 | 0 B |
| Per-stage scheduler | `46dfba42f` | 173.150 | 173.133 | −0.01% | 108 | 0 B |
| Single A register slot | `3018cea23` | 173.520 | 173.737 | +0.13% | 84 | 0 B |
| Single-buffered stage operands | `a10427aac` | 173.051 | 175.076 | +1.17% | 84 | 0 B |

M=1024 counters, two-K128-stage v5 versus K128, 25 dispatches per single-metric pass:

| Metric | K128 | v5 | Delta |
| --- | ---: | ---: | ---: |
| `SQ_INSTS_MFMA` | 3158016 | 3158016 | 0% |
| `SQ_INSTS_LDS` | 888192 | 888192 | 0% |
| `SQ_INSTS_SMEM` | 68592 | 68592 | 0% |
| `SQ_INSTS_VMEM` | 1414528 | 1422752 | +0.58% |
| `SQ_INSTS_VALU` | 5826704 | 5662224 | −2.82% |
| `SQ_INSTS_SALU` | 1189856 | 1041824 | −12.44% |
| OccupancyPercent | 18.499% | 18.420% | −0.08 pp |
| MfmaUtil | 12.234% | 12.000% | −0.23 pp |

Static instruction mix, all kernel variants per code object:

| Object | mfma | ds_read | buffer_load | scratch | s_waitcnt |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 K128 M64 | 320 | 112 | 144 | 0 | 182 |
| v3 K256 M64 | 640 | 192 | 288 | 128 | 298 |
| v5 K256 M64, two stages | 640 | 192 | 288 | 0 | 269 |

Fixed-seed MBSTAT, M=1024, seed 1234:

| Arm | absmean | std | sum |
| --- | ---: | ---: | ---: |
| v3 K128 M64 | 2.755572e4 | 3.468417e4 | 1.453110e8 |
| v5 K256 M64, two stages | 2.755573e4 | 3.468416e4 | 1.451541e8 |
| v3 K256 M64 | 2.755572e4 | 3.468417e4 | 1.452175e8 |

## M=1024 is work-starved, not resource-limited

Grid is 410624 threads, so 1604 workgroups of 256 threads. Residency is 5 blocks per CU (32 KB LDS
of 160 KB; 88 VGPR allows 5 waves per SIMD), so 1280 blocks are resident and the kernel is about
1.25 rounds deep. Runtime is therefore one workgroup's K loop, and register cuts cannot buy waves:

| Arm | VGPR | OccupancyPercent | MeanOccupancyPerCU | MfmaUtil | MemUnitStalled | median µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 K128 M64 | 88 | 18.344% | 5.877 | 12.038% | 0.092% | 172.0 |
| v5, double-buffered stages | 108 | 18.420% | — | 12.000% | — | 173.1 |
| v5, single-buffered stages | 84 | 25.748% | — | 11.780% | 0.080% | 175.1 |
| FlyDSL M64 BK256 | 100 | 18.044% | 5.785 | — | — | 150.8 |

Retained configuration is double-buffered.

## B read efficiency is already optimal

M=1024 single-metric passes, CK K128 M64 versus FlyDSL M64 BK256, medians over 25 dispatches:

| Metric | CK | FlyDSL | CK / FlyDSL |
| --- | ---: | ---: | ---: |
| `FETCH_SIZE` | 461104 KB | 453955 KB | 1.016 |
| `TCC_REQ_sum` | 8.02517e6 | 7.94028e6 | 1.011 |
| `TCC_MISS_sum` | 7.45009e6 | 7.30171e6 | 1.020 |
| `TCC_HIT_sum` | 575182 | 638565 | 0.901 |
| `TCP_TCC_READ_REQ_sum` | 7.71030e6 | 7.76164e6 | 0.993 |
| L2 hit rate | 7.2% | 8.0% | — |
| `SQ_BUSY_CYCLES` | 1.02718e7 | 9.38518e6 | 1.094 |
| `SQ_WAIT_ANY` | 3.07105e7 | 4.31132e7 | 0.712 |
| Achieved read bandwidth | 2.62 TB/s | 2.94 TB/s | 0.891 |

Analytic minimum B bytes at M=1024: 9216 token slots pad to 144 blocks of `block_m` 64, each
reading gate and up weights of 1024 rows by 6144 K at 0.5 B per element, so 144 × 3.146 MB =
453 MB. Both kernels fetch that, so neither wastes weight traffic and L2 supplies nothing because
each expert block is read once.

Ruled out as causes of the small-M gap: K tile granularity, LDS barrier count, spills, non-MFMA
instruction count, instruction scheduling, wave occupancy, fetched bytes, L2 behavior, and load
request count.

## The kernel is concurrency-limited: bandwidth scales with workgroup count

`block_m` 32 halves the M tile, so every m-block still reads the full B slice and weight traffic
grows. M=1024, GPU4, three alternating repetitions:

| Arm | Workgroups | `FETCH_SIZE` | median µs | Achieved read BW |
| --- | ---: | ---: | ---: | ---: |
| K128 M64, `block_m` 64 | 1604 | 461104 KB | 173.402 | 2.62 TB/s |
| K128 M32, `block_m` 32 | 2180 | 770920 KB | 219.480 | 3.60 TB/s |
| FlyDSL M64 BK256 | 1600 | 453955 KB | 150.800 | 2.94 TB/s |

1.36x the workgroups buys 1.37x the read bandwidth, so at `block_m` 64 the kernel cannot saturate
HBM for lack of concurrent requests. `block_m` 32 loses anyway because it fetches 1.67x the bytes.
Split-K adds the same concurrency without touching B traffic, since each split reads a different K
slice; only the partial outputs are extra, about 37.7 MB written and read back per split against
450 MB of weights.

## Split-K status for a4w4 gemm1

| Path | Split-K support | Note |
| --- | --- | --- |
| CK2stages, per_1x128 a8w8 blockscale | yes | `KBatch = K / (splitk * KPerBlock)`, `StrideE = 2N`, fp32 partials, `MulABScaleExpertWeightA8W8blkscaleSplitk` skips the fused activation |
| CK2stages, per_1x32 MXFP4 | no | `gemm_moe_ck2stages_common_mxfp4.cuh` hardcodes `KBatch = 1`; the python guard requires `quant_type == per_1x128` |
| CK-Tile, MXFP4 preshuffled | yes, A16W4 only | `ksplit > 1` with `q_dtype_a` fp4x2 sets `a1 = hidden_states.to(dtype)` and `a1_scale = None`, so activations become bf16; pinning an a4w4 CK2stages kernel with `ksplit > 1` faults with `hipErrorIllegalAddress` |

FlyDSL gemm1 does not split K either: grid 409600 threads is 1600 workgroups, exactly 200 m-blocks
by 8 n-blocks, and `mfma_moe1_silu_mul_afp4_wfp4_fp4_t64x128x256_pm1_fp4q_sort` fuses SiLU, the
gate-up multiply and fp4 quant. Its gemm2 names do carry `reduce` and `atomic_persist` markers.
FlyDSL therefore reaches 2.94 TB/s at the same concurrency CK gets 2.62 TB/s from.

Split-K traffic accounting at M=1024, 12800 sorted slots by 1024 gate-and-up columns:

| Item | Today | Split-K 2 |
| --- | ---: | ---: |
| gemm1 output | 3.3 MB fp4 | 52.4 MB fp32 |
| Zeroing memset | — | 52.4 MB |
| Atomic partial writes | — | 105 MB |
| Activation and quant pass | — | 52.4 MB read, 3.3 MB write |
| Extra versus the 450 MB weight stream | — | +210 MB, +47% |

Split-K gate: it changes nothing about gemm2, whose input contract stays fp4 `inter_states` plus
per-32 scales, and gemm1 is 173 µs of a 289 µs two-stage total, so a 10% gemm1 win is 6% end to
end. Do not implement split-K unless a projected gain of 5% or more on gemm1 survives the traffic
penalty above, which requires first measuring that the 52.4 MB partial buffer stays LLC-resident.
The only lever with proven headroom is the 12% bandwidth gap to FlyDSL at equal concurrency.

## LDS accounting, and why reclaiming it does not help

`GetSharedMemoryNumberOfByte` in `gridwise_gemm_xdl_cshuffle_common.hpp` returns
`max((A_aligned + B_aligned) * NumLdsBuffer, c_block_size * sizeof(F32))` with the B term forced to
zero under `BPreshuffle`, and `kernel_moe_mxgemm_2lds` declares two such arrays. For K128 M64:

| Term | Value |
| --- | ---: |
| A LDS tile, `(AK0 8, M 64, AK1 16)` strided | 4096 B |
| A term with `NumLdsBuffer` 2 | 8192 B |
| C-shuffle tile, `CShuffleMXdlPerWavePerShuffle 2 * MWave 1 * MPerXdl 16` by `2 * NWave 4 * NPerXdl 16`, fp32 | 16384 B |
| `GetSharedMemoryNumberOfByte` | 16384 B |
| Dispatched LDS, two arrays | 32768 B |

The C-shuffle epilogue buffer, not the A tile, sets the size, and it is paid twice. The model
reproduces the other instances: M128 K256 has a 32768 B A term giving 65536 B, and the single-buffer
v1 64x32x32 instance gives 4096 B.

Reclaiming it is pointless because LDS does not bind residency:

| Metric | Value |
| --- | ---: |
| Hardware LDS per CU | 160 KB |
| Blocks per CU allowed by LDS at 32768 B | 5 |
| Blocks per CU allowed by 88 VGPR | 5 |
| `Max_Waves_Per_Cu` | 32 |
| `MeanOccupancyPerCU` at M=1024, 1604 workgroups | 5.914 |
| `MeanOccupancyPerCU` at M=16384, 10244 workgroups | 7.455 |

Achieved residency is about 1.5 to 1.9 blocks per CU against a resource ceiling of 5, and 6.4x more
workgroups moves it only 26%, so neither LDS, registers, nor work availability is the constraint.
Halving `CShuffleMXdlPerWavePerShuffle` also fails to compile: the epilogue thread-cluster lengths
`S<1, CShuffleNLane, 1, CShuffleMLane>` no longer tile the smaller shuffle tile.

## Barrier frequency: CK 48 per workgroup, FlyDSL 1

FlyDSL gemm1 `mfma_moe1_silu_mul_afp4_wfp4_fp4_t64x128x256_pm1_fp4q_sort_async_v32`, from the
lowered IR in its `flydsl_cache` pickle:

| Structure | Count |
| --- | ---: |
| `scf.for` | 1 |
| `gpu.barrier` total | 4 |
| `gpu.barrier` inside the loop body | 1 |
| `rocdl.mfma` in the loop body | 896 |
| `rocdl.raw.ptr.buffer.load` | 405 |
| `rocdl.s.waitcnt` | 29 |
| `rocdl.sched.barrier` | 370 |
| `rocdl.s.setprio` | 208 |

896 MFMAs is the whole K=6144 reduction for this tile (48 K128 steps by 4 M by 2 N by gate and up
is 768, plus tail), so FlyDSL crosses one block barrier per K reduction. CK stages A through a
two-deep LDS ping-pong and calls `block_sync_lds` once per K tile: 48 barriers at K128, 24 at the
v5 K256 staging. Each barrier rendezvouses all four waves and its `s_waitcnt` drains outstanding
loads, which is the mechanism behind equal traffic and equal occupancy but 12% less bandwidth.

Lever: the LDS is already over-allocated 4x per buffer (16384 B holding a 4096 B A tile), so a
deeper A ring of 4 or 8 tiles fits in the same 32768 B and cuts barriers proportionally, to 6 to 12
per workgroup, at zero LDS or traffic cost. Both `kernel_moe_mxgemm_2lds` and the pipeline assume
exactly two buffers, so this is a real change, not a parameter flip.

## Rejected and neutral probes at M=1024

| Probe | Result | Disposition |
| --- | ---: | --- |
| M64 MFMA priority | −1.7% | retained, below promotion bar |
| Precomputed scale coordinate steps | +0.51% SALU | reverted |
| Precomputed B gate/up coordinate step | +0.32% SALU | reverted |
| Invariant A-gather offsets | +0.53% | reverted |
| MFMA-aligned A fragment, K256 M128 | +0.3% at 1024 | reverted |
| K256 gridwise B-buffer removal | 0% VGPR, 0% LDS | reverted |
| `ARegBuf` 2 to 1 | VGPR 108 to 84, +0.13% time | kept, no time effect |
| Single-buffered v5 stage operands | +1.17% | reverted |

## Invalid historical timings

| M | K=256 µs | Reason invalid |
| ---: | ---: | --- |
| 512 | 65.9 | Broken pre-fix K=256 kernel |
| 1024 | 123.2 | Broken pre-fix K=256 kernel |
| 4096 | 242.9 | Broken pre-fix K=256 kernel |
| 8192 | 391.0 | Broken pre-fix K=256 kernel |
| 1024 | 125.5 | `--csv-filter` comparison did not isolate valid arms |
| 16384 | 758.2 | `--csv-filter` comparison did not isolate valid arms |
| 32768 | 1342.9 | `--csv-filter` comparison did not isolate valid arms |

## Method

| Item | Value |
| --- | --- |
| Timing | `test_moe_2stage.py --kernel --no-legacy --csv-filter moe_ck2stages`, field `us_stage1` |
| Dispatch control | `AITER_CONFIG_FMOE` CSV pinning full `kernelName1` per arm |
| Repetitions | Three or more, arms alternating, same GPU and seed, identical FlyDSL gemm2 |
| Rebuild | Delete `aiter/jit/module_moe_ck2stages_..._mulWeightStage2.so` and its `build/` directory; `AITER_REBUILD=1` alone reuses the stale instance blob and falls back with `ck kernel not found` |
| Counters | Single-metric `rocprofv3 --pmc <one> --kernel-trace`; multi-metric hangs |
| Counter parsing | Filter `Kernel_Name` on `mxgemm`, sum per `Dispatch_Id`, take the median |
| Code-object metadata | `roc-obj-ls` the module, slice each `gfx950 offset/size`, then `llvm-objdump -t` with `c++filt` for the symbol and `llvm-readelf --notes` split on `.name:` |
| Tuner | `TUNE_ONLY=cktile python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py -i <in> -o <out> -o2 <profile> --all --mp 1 --shape_grouped` |
