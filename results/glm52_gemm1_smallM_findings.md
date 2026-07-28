# GLM-5.2 MoE gemm1 K=256

## Configuration


| Field            | Value                                                          |
| ---------------- | -------------------------------------------------------------- |
| Shape            | a4w4 MXFP4, model_dim=6144, inter_dim=512, E=257, topk=9, SiLU |
| Hardware         | gfx950, 256 CUs                                                |
| aiter branch     | `RM/glm52-mxfp4-moe-ck-gemm1`                                  |
| aiter commit     | `433445255`                                                    |
| CK branch        | `RM/glm52-a4w4-k256-swizzle-fix`                               |
| CK commit        | `b1fd91e44`                                                    |
| K=128 candidates | `256x64x128x128_1x4`, `256x128x128x128_1x4`, gufusion_v3       |
| K=256 candidates | `256x64x128x256_1x4`, `256x128x128x256_1x4`, gufusion_v3       |


## Correctness


| Metric                    | K=128     | K=256 fixed |
| ------------------------- | --------- | ----------- |
| MBSTAT absmean, seed 1234 | 2.75533e4 | 2.75533e4   |
| MBSTAT std, seed 1234     | 3.46699e4 | 3.46699e4   |
| MBSTAT sum                | 4.5938e7  | 4.5980e7    |
| GSM8K accuracy            | 0.929     | 0.930       |
| GSM8K invalid             | —         | 0.001       |




## Fix


| Field          | Value                                                                                                                        |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| File           | `3rdparty/composable_kernel/include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_gather_direct_load.hpp` |
| Broken swizzle | `(threadIdx.x % 64) / 8`                                                                                                     |
| Fixed swizzle  | `(threadIdx.x / block_slice_lengths.At(I0)) % block_slice_lengths.At(I0)`                                                    |
| K=128 behavior | Preserved                                                                                                                    |
| K=256 behavior | Runtime LDS-write swizzle now matches the descriptor read swizzle                                                            |




## Corrected-kernel performance

Median isolated gemm1 `us_stage1`, three alternating repetitions on GPU2:


| M     | K128 M64 µs | K256 M64 µs | K128 M128 µs | K256 M128 µs | Best K128 µs | Best K256 µs | K256 delta |
| ----- | ----------- | ----------- | ------------ | ------------ | ------------ | ------------ | ---------- |
| 1024  | 177.068     | 376.090     | 199.959      | 217.045      | 177.068      | 217.045      | +22.6%     |
| 4096  | 308.631     | 812.824     | 314.376      | 361.631      | 308.631      | 361.631      | +17.2%     |
| 8192  | 457.508     | 1350.550    | 421.297      | 490.984      | 421.297      | 490.984      | +16.5%     |
| 16384 | 739.723     | 2398.680    | 651.215      | 775.001      | 651.215      | 775.001      | +19.0%     |
| 32768 | 1300.280    | 4540.910    | 1073.100     | 1354.160     | 1073.100     | 1354.160     | +26.2%     |



| Check           | Result                                               |
| --------------- | ---------------------------------------------------- |
| CK-only tuner   | K128 selected over K256 for every shared `block_m`   |
| Direct dispatch | Full `kernelName1` verified for every arm            |
| Stage2 control  | Identical FlyDSL gemm2 within each paired comparison |
| K256 vs K128    | K256 slower at every M                               |
| Correctness     | K256 remains GSM8K/MBSTAT validated                  |




## Same-session K128 vs FlyDSL

Median isolated gemm1 `us_stage1`, four alternating repetitions on GPU2:


| M     | K128 CK µs | FlyDSL µs | CK delta |
| ----- | ---------- | --------- | -------- |
| 1024  | 177.807    | 151.865   | +17.1%   |
| 4096  | 309.852    | 283.250   | +9.4%    |
| 8192  | 419.085    | 425.754   | −1.6%    |
| 16384 | 648.645    | 687.847   | −5.7%    |
| 32768 | 1071.710   | 1173.645  | −8.7%    |




## FlyDSL XCD4 remap A/B

Median isolated gemm1 `us_stage1`, six alternating repetitions on GPU4:


| M     | No XCD µs | XCD4 µs  | XCD4 delta |
| ----- | --------- | -------- | ---------- |
| 1024  | 152.297   | 191.346  | +25.6%     |
| 4096  | 304.593   | 282.966  | −7.1%      |
| 8192  | 421.754   | 420.763  | −0.2%      |
| 16384 | 683.359   | 677.419  | −0.9%      |
| 32768 | 1176.155  | 1169.315 | −0.6%      |




## M=1024 dynamic instruction differential

GPU4, 25 identical dispatches per single-metric pass:


| Metric                    | CK K128 M64 | FlyDSL M64 BK256 w2 | CK delta |
| ------------------------- | ----------- | ------------------- | -------- |
| Grid threads              | 410624      | 409600              | +0.2%    |
| `SQ_INSTS_MFMA`           | 3158016     | 3158016             | 0.0%     |
| `SQ_INSTS_VALU`           | 8869584     | 7383004             | +20.1%   |
| `SQ_INSTS_VALU_FMA_F32`   | 1052672     | 0                   | +1052672 |
| `SQ_INSTS_VALU_ADD_F32`   | 394752      | 187708              | +110.3%  |
| `SQ_INSTS_VALU_CVT`       | 197376      | 0                   | +197376  |
| `SQ_INSTS_VALU_MUL_F32`   | 526336      | 432168              | +21.8%   |
| `SQ_INSTS_VALU_TRANS_F32` | 263168      | 263168              | 0.0%     |
| `SQ_INSTS_VALU_INT32`     | 1110240     | 1070772             | +3.7%    |
| `SQ_INSTS_VALU_INT64`     | 41120       | 127880              | −67.8%   |




## Reciprocal SiLU experiment at M=1024

Median isolated gemm1 `us_stage1`, six alternating repetitions on GPU4:


| Metric                  | Generic SiLU | Reciprocal SiLU | Delta  |
| ----------------------- | ------------ | --------------- | ------ |
| `us_stage1`             | 177.593 µs   | 176.572 µs      | −0.6%  |
| `SQ_INSTS_VALU`         | 8869584      | 7553744         | −14.8% |
| `SQ_INSTS_VALU_FMA_F32` | 1052672      | 263168          | −75.0% |
| `SQ_INSTS_VALU_CVT`     | 197376       | 197376          | 0.0%   |
| Strict accuracy         | pass         | pass            | —      |




## M64 MFMA priority experiment

Median isolated gemm1 `us_stage1`, four alternating repetitions on GPU4:


| M     | Baseline µs | M64 priority µs | Delta |
| ----- | ----------- | --------------- | ----- |
| 1024  | 178.070     | 175.092         | −1.7% |
| 4096  | 309.316     | 309.831         | +0.2% |
| 8192  | 420.379     | 418.845         | −0.4% |
| 16384 | 642.892     | 644.055         | +0.2% |
| 32768 | 1054.610    | 1052.945        | −0.2% |




## Promotion gates


| Gate                                  | Required                        | Result         | Status  |
| ------------------------------------- | ------------------------------- | -------------- | ------- |
| M=1024 versus same-session baseline   | material improvement            | −1.7%          | fail    |
| M=1024 versus 171.2 µs reference      | <171.2 µs                       | 175.092 µs     | fail    |
| Operand-preparation instruction count | reduced                         | unchanged      | fail    |
| Strict accuracy                       | pass                            | pass           | pass    |
| M≥8192 regression                     | within noise                    | −0.4% to +0.2% | pass    |
| Production autotune                   | winning kernel required         | not run        | blocked |
| Production CSV                        | validated winning rows required | unchanged      | blocked |




## Packed register-fragment experiment

M=1024 median isolated gemm1 `us_stage1`, six alternating repetitions on GPU4:


| M    | Baseline µs | M64 priority + packed fragments µs | Delta  |
| ---- | ----------- | ---------------------------------- | ------ |
| 1024 | 177.767     | 173.949                            | −2.15% |


Median isolated gemm1 `us_stage1`, three alternating full-sweep repetitions on GPU4:


| M     | Baseline µs | M64 priority + packed fragments µs | Delta  |
| ----- | ----------- | ---------------------------------- | ------ |
| 1024  | 176.897     | 174.688                            | −1.25% |
| 4096  | 308.722     | 311.394                            | +0.87% |
| 8192  | 416.857     | 409.914                            | −1.67% |
| 16384 | 639.944     | 631.698                            | −1.29% |
| 32768 | 1058.600    | 1042.910                           | −1.48% |


M=1024 trace and single-counter passes:


| Metric                  | Baseline   | M64 priority + packed fragments | Delta     |
| ----------------------- | ---------- | ------------------------------- | --------- |
| Kernel-trace median     | 177.161 µs | 173.121 µs                      | −2.28%    |
| VGPR count              | 96         | 88                              | −8.33%    |
| SGPR count              | 112        | 112                             | 0%        |
| LDS per block           | 32768 B    | 32768 B                         | 0%        |
| `SQ_INSTS_VALU`         | 8869584    | 8869584                         | 0%        |
| `SQ_INSTS_MFMA`         | 3158016    | 3158016                         | 0%        |
| `SQ_INSTS_VALU_FMA_F32` | 1052672    | 1052672                         | 0%        |
| `SQ_INSTS_VALU_CVT`     | 197376     | 197376                          | 0%        |
| `SQ_INSTS_VALU_INT64`   | 41120      | 41120                           | 0%        |
| `SQ_INSTS_VALU_INT32`   | 1110240    | 1110240                         | 0%        |
| OccupancyPercent        | 18.657%    | 18.200%                         | −0.457 pp |
| MfmaUtil                | 11.522%    | 11.579%                         | +0.057 pp |
| VALUUtilization         | 99.210%    | 99.210%                         | 0 pp      |
| Strict accuracy         | pass       | pass                            | —         |


M=8192 M128 trace, M64 priority compile-time disabled:


| Metric              | Baseline   | Packed fragments | Delta  |
| ------------------- | ---------- | ---------------- | ------ |
| Kernel-trace median | 418.722 µs | 397.001 µs       | −5.19% |
| VGPR count          | 128        | 128              | 0%     |
| SGPR count          | 112        | 112              | 0%     |
| LDS per block       | 32768 B    | 32768 B          | 0%     |




## Retained packed-fragment + reciprocal-SiLU candidate

M=1024 median isolated gemm1, four alternating repetitions on GPU4:


| Metric                  | Baseline   | Packed + reciprocal | Delta   |
| ----------------------- | ---------- | ------------------- | ------- |
| `us_stage1`             | 177.587 µs | 172.553 µs          | −2.83%  |
| `SQ_INSTS_VALU`         | 8869584    | 7553744             | −14.84% |
| `SQ_INSTS_MFMA`         | 3158016    | 3158016             | 0%      |
| `SQ_INSTS_VALU_FMA_F32` | 1052672    | 263168              | −75.00% |
| `SQ_INSTS_VALU_CVT`     | 197376     | 197376              | 0%      |
| Strict accuracy         | pass       | pass                | —       |




## Native-exp2 SiLU candidate

Median isolated gemm1 `us_stage1`, three alternating full-sweep repetitions on GPU4:


| M     | Baseline µs | Packed + reciprocal + exp2 µs | Delta  |
| ----- | ----------- | ----------------------------- | ------ |
| 1024  | 177.794     | 172.108                       | −3.20% |
| 4096  | 308.900     | 307.401                       | −0.49% |
| 8192  | 413.738     | 402.673                       | −2.67% |
| 16384 | 644.110     | 620.169                       | −3.72% |
| 32768 | 1052.040    | 1016.530                      | −3.38% |


M=1024 single-counter passes:


| Metric                    | Baseline | Packed + reciprocal + exp2 | Delta    |
| ------------------------- | -------- | -------------------------- | -------- |
| `SQ_INSTS_VALU`           | 8869584  | 6102208                    | −31.20%  |
| `SQ_INSTS_MFMA`           | 3158016  | 3158016                    | 0%       |
| `SQ_INSTS_VALU_FMA_F32`   | 1052672  | 0                          | −100.00% |
| `SQ_INSTS_VALU_CVT`       | 197376   | 65792                      | −66.67%  |
| `SQ_INSTS_VALU_ADD_F32`   | 394752   | 131584                     | −66.67%  |
| `SQ_INSTS_VALU_MUL_F32`   | 526336   | 394752                     | −25.00%  |
| `SQ_INSTS_VALU_TRANS_F32` | 263168   | 263168                     | 0%       |
| Strict accuracy           | pass     | pass                       | —        |



| M=1024 VALU comparison        | Instructions | CK delta |
| ----------------------------- | ------------ | -------- |
| FlyDSL                        | 7383004      | —        |
| CK packed + reciprocal + exp2 | 6102208      | −17.35%  |




## Post-exp2 structural differential at M=1024


| Metric               | CK K128 M64 | FlyDSL BK256 M64 | CK delta  |
| -------------------- | ----------- | ---------------- | --------- |
| Kernel-trace median  | 172.401 µs  | 151.241 µs       | +13.99%   |
| Workgroup size       | 256         | 256              | 0%        |
| Grid X×Y             | 1024×401    | 1024×400         | +0.25%    |
| VGPR count           | 88          | 100              | −12.00%   |
| LDS per block        | 32768 B     | 55808 B          | −41.29%   |
| `SQ_INSTS_VALU`      | 6102208     | 7383004          | −17.35%   |
| `SQ_INSTS_MFMA`      | 3158016     | 3158016          | 0%        |
| `SQ_INSTS_SALU`      | 1265680     | 762296           | +66.04%   |
| `SQ_INSTS_SMEM`      | 76816       | 35648            | +115.48%  |
| `SQ_INSTS_VMEM`      | 1414528     | 1449324          | −2.40%    |
| `SQ_INSTS_LDS`       | 888192      | 947604           | −6.27%    |
| `SQ_INSTS_LDS_LOAD`  | 822400      | 824660           | −0.27%    |
| `SQ_INSTS_LDS_STORE` | 65792       | 66820            | −1.54%    |
| LDSBankConflict      | 0.600%      | 0.270%           | +0.330 pp |
| MemUnitStalled       | 0.096%      | 0.132%           | −0.036 pp |


CK M128 K-granularity differential at M=1024:


| Metric          | K128    | K256    | K256 delta |
| --------------- | ------- | ------- | ---------- |
| `SQ_INSTS_SALU` | 1491344 | 1367984 | −8.27%     |
| `SQ_INSTS_SMEM` | 81584   | 81584   | 0%         |
| `SQ_INSTS_VMEM` | 1858624 | 1858624 | 0%         |
| `SQ_INSTS_LDS`  | 1776384 | 1776384 | 0%         |


Active-kernel static scalar instruction classes:


| Instruction      | CK K128 M64 | FlyDSL BK256 M64 |
| ---------------- | ----------- | ---------------- |
| `s_mul_i32`      | 20          | 4                |
| `s_mul_hi_i32`   | 2           | 1                |
| `s_add_i32`      | 36          | 5                |
| `s_add_u32`      | 19          | 0                |
| `s_addc_u32`     | 19          | 0                |
| `s_lshl_b32`     | 14          | 6                |
| `s_lshr_b32`     | 11          | 2                |
| `s_load_dword`   | 10          | 1                |
| `s_load_dwordx2` | 4           | 2                |
| `s_load_dwordx4` | 2           | 3                |
| `s_load_dwordx8` | 1           | 1                |


## Precomputed coordinate-step probes at M=1024

| Arm | `us_stage1` smoke µs | `SQ_INSTS_SALU` | SALU delta | `SQ_INSTS_SMEM` | Strict accuracy |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 172.108 | 1265680 | — | 76816 | pass |
| Precomputed scale steps | 172.312 | 1272096 | +0.51% | 76816 | pass |
| Precomputed B gate/up step | 172.002 | 1269792 | +0.32% | 76816 | pass |

## Raw scale-offset experiment at M=1024

| Metric | Baseline | Raw scale offsets | Delta |
| --- | ---: | ---: | ---: |
| Median `us_stage1`, 6 repetitions | 173.246 µs | 173.363 µs | +0.07% |
| Kernel-trace median | — | 173.321 µs | — |
| `SQ_INSTS_SALU` | 1265680 | 1189856 | −5.99% |
| `SQ_INSTS_SMEM` | 76816 | 68592 | −10.71% |
| `SQ_INSTS_VALU` | 6102208 | 5917168 | −3.03% |
| `SQ_INSTS_MFMA` | 3158016 | 3158016 | 0% |
| `SQ_INSTS_VMEM` | 1414528 | 1414528 | 0% |
| VGPR count | 88 | 88 | 0% |
| SGPR count | 112 | 112 | 0% |
| LDS per block | 32768 B | 32768 B | 0% |
| Strict accuracy | pass | pass | — |



## MFMA-aligned A-fragment experiment

Median isolated K256 M128 gemm1 `us_stage1`, three alternating repetitions on GPU4:


| M     | Baseline µs | Aligned A fragment µs | Delta |
| ----- | ----------- | --------------------- | ----- |
| 1024  | 215.291     | 215.866               | +0.3% |
| 4096  | 361.226     | 362.786               | +0.4% |
| 8192  | 495.863     | 490.383               | −1.1% |
| 16384 | 765.258     | 764.044               | −0.2% |
| 32768 | 1312.630    | 1315.490              | +0.2% |




## K=256 resource diagnosis at M=16384

Exact rocprof kernel trace and separate single-counter passes:


| Metric                  | K128 M128  | K256 M128  | Delta   |
| ----------------------- | ---------- | ---------- | ------- |
| Kernel-trace median     | 643.085 µs | 795.085 µs | +23.6%  |
| VGPR count              | 128        | 200        | +56.3%  |
| LDS per block           | 32768 B    | 65536 B    | +100.0% |
| Workgroup size          | 256        | 256        | 0%      |
| Grid size               | 1024       | 1024       | 0%      |
| OccupancyPercent median | 22.193%    | 11.561%    | −47.9%  |
| MfmaUtil median         | 50.545%    | 34.198%    | −32.3%  |



| Trace check                    | K128               | K256                |
| ------------------------------ | ------------------ | ------------------- |
| KPerBlock in dispatched symbol | 128                | 256                 |
| A-transfer cluster             | `Sequence<8,32,1>` | `Sequence<16,16,1>` |
| TailNumber                     | 2                  | 1                   |
| Kernel identity                | Correct            | Correct             |




## Invalid historical timings


| M     | K=256 µs | Reason invalid                                                         |
| ----- | -------- | ---------------------------------------------------------------------- |
| 512   | 65.9     | Broken pre-fix K=256 kernel                                            |
| 1024  | 123.2    | Broken pre-fix K=256 kernel                                            |
| 4096  | 242.9    | Broken pre-fix K=256 kernel                                            |
| 8192  | 391.0    | Broken pre-fix K=256 kernel                                            |
| 1024  | 125.5    | Post-fix `--kernel --csv-filter` comparison did not isolate valid arms |
| 16384 | 758.2    | Post-fix `--kernel --csv-filter` comparison did not isolate valid arms |
| 32768 | 1342.9   | Post-fix `--kernel --csv-filter` comparison did not isolate valid arms |




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
HIP_VISIBLE_DEVICES=2 \
TUNE_ONLY=cktile \
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i /tmp/g1_untuned.csv \
  -o /tmp/g1_ck_tuned.csv \
  -o2 /tmp/g1_ck_profile.csv \
  --all --mp 1 --shape_grouped
```



## Verification method


| Check            | Requirement                                                           |
| ---------------- | --------------------------------------------------------------------- |
| Tuner output     | `/tmp/g1_ck_profile.csv`                                              |
| Tuner comparison | Best candidate retained per `block_m`; K128 retained for M64 and M128 |
| Direct timing    | `test_moe_2stage.py --kernel --no-legacy --csv-filter moe_ck2stages`  |
| Repetitions      | Three, alternating K128/K256 arms                                     |
| Timing field     | `us_stage1`                                                           |
| Kernel identity  | Full `kernelName1` must contain the expected KPerBlock value          |
| Backend          | CK Tile only via `TUNE_ONLY=cktile`                                   |
| Control          | Same GPU, shape, seed, sorting `block_m`, and FlyDSL gemm2            |


