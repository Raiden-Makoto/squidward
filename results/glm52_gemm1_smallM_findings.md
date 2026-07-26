# GLM-5.2 MoE gemm1 K=256

## Configuration

| Field | Value |
| --- | --- |
| Shape | a4w4 MXFP4, model_dim=6144, inter_dim=512, E=257, topk=9, SiLU |
| Hardware | gfx950, 256 CUs |
| aiter branch | `RM/glm52-mxfp4-moe-ck-gemm1` |
| aiter commit | `433445255` |
| CK branch | `RM/glm52-a4w4-k256-swizzle-fix` |
| CK commit | `b1fd91e44` |
| K=128 tile | `256x128x128x128_1x4`, gufusion_v3 |
| K=256 tile | `256x128x128x256_1x4`, gufusion_v3 |

## Correctness

| Metric | K=128 | K=256 fixed |
| --- | ---: | ---: |
| MBSTAT absmean, seed 1234 | 2.75533e4 | 2.75533e4 |
| MBSTAT std, seed 1234 | 3.46699e4 | 3.46699e4 |
| MBSTAT sum | 4.5938e7 | 4.5980e7 |
| GSM8K accuracy | 0.929 | 0.930 |
| GSM8K invalid | — | 0.001 |

## Fix

| Field | Value |
| --- | --- |
| File | `3rdparty/composable_kernel/include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_gather_direct_load.hpp` |
| Broken swizzle | `(threadIdx.x % 64) / 8` |
| Fixed swizzle | `(threadIdx.x / block_slice_lengths.At(I0)) % block_slice_lengths.At(I0)` |
| K=128 behavior | Preserved |
| K=256 behavior | Runtime LDS-write swizzle now matches the descriptor read swizzle |

## Performance status

| Measurement | Status |
| --- | --- |
| Corrected K=256 vs K=128 CK | Not measured |
| Corrected K=256 vs FlyDSL | Not measured |
| Full tuner attempt | Did not complete |
| `--kernel --csv-filter` K=256/K=128/FlyDSL comparison | Invalid dispatch/method |

## Invalid historical timings

| M | K=256 µs | Reason invalid |
| ---: | ---: | --- |
| 512 | 65.9 | Broken pre-fix K=256 kernel |
| 1024 | 123.2 | Broken pre-fix K=256 kernel |
| 4096 | 242.9 | Broken pre-fix K=256 kernel |
| 8192 | 391.0 | Broken pre-fix K=256 kernel |
| 1024 | 125.5 | Post-fix `--kernel --csv-filter` comparison did not isolate valid arms |
| 16384 | 758.2 | Post-fix `--kernel --csv-filter` comparison did not isolate valid arms |
| 32768 | 1342.9 | Post-fix `--kernel --csv-filter` comparison did not isolate valid arms |

## Tuner input

```csv
token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1
1024,6144,512,257,9,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
4096,6144,512,257,9,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
8192,6144,512,257,9,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
16384,6144,512,257,9,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
32768,6144,512,257,9,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
```

## Tuner command

```bash
PYTHONPATH=/sgl-workspace/aiter_dev \
HIP_VISIBLE_DEVICES=0 \
TUNE_ONLY=cktile \
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i /tmp/g1_untuned.csv \
  -o /tmp/g1_tuned.csv \
  -o2 /tmp/g1_profile.csv \
  --all --mp 1 --shape_grouped
```

## Result extraction

| Check | Requirement |
| --- | --- |
| Timing source | `us1` from `/tmp/g1_profile.csv` |
| Comparison | Best K=128 and best K=256 candidates at each M in the same run |
| Kernel identity | Full `kernelName1` must contain the expected KPerBlock value |
| Backend | CK Tile only via `TUNE_ONLY=cktile` |
| Method validation | K=128 curve must remain consistent with the established same-tuner baseline |
