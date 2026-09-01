# GLM-5.3 MXFP4 — i8k/o1k REPS=3

Branch: `RM/glm51` (`4c6b53d54d`)  
Image: `rocm/sgl-dev:v0.5.18-rocm720-mi35x-20260830`  
Model: `/data2/hf_home/hub/GLM-5.3-Quark-MXFP4-AttnFP8`  
Server: `bash utilities/run_glm53.sh --triton`  
Benchmark: `REPS=3 PORT=8553 bash utilities/e2e_glm5.sh 8192 1024 0`

| concurrency | TTFT (ms) | ITL (ms) | E2EL (ms) | output tok/s |
|---:|---:|---:|---:|---:|
| 4 | 240.2 | 11.63 | 12161 | 336.8 |
| 8 | 277.1 | 13.92 | 14912 | 549.7 |
| 16 | 292.1 | 16.30 | 17871 | 921.3 |
| 32 | 340.4 | 21.40 | 24068 | 1376.2 |
| 64 | 509.5 | 25.79 | 30437 | 2206.5 |
