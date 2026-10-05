# GLM-5.3-Flash Agentic benchmarks

Imported from [`Jacob0226/SGLang-benchmarks@GLM5_InferenceX`](https://github.com/Jacob0226/SGLang-benchmarks/tree/GLM5_InferenceX/Agentic) at `1661b3c91777bbbbeff79ec8d37776e6cfd93cc5`.

Use `ix_agentx_glm53flash.sh` inside the matching SGLang container. It drives the B200 NVFP4 or MI355X Quark MXFP4 recipe with InferenceX's AgentX corpus, result schema, warmup, metrics, and failure gates.

## RM/glm53 MI355X tuple

- Image: `rocm/sgl-dev:v0.5.21-rocm724-mi35x-20261001`
- Model: `amd/GLM-5.3-Flash-Quark-MXFP4`
- TP4/EP1, BF16 KV, AITER MoE, Triton linear attention
- The DSA sub-backends are intentionally left unset: #41615 makes the existing HIP default select Triton for the validated gfx950 GLM-5.3 geometry, while an explicit TileLang selection remains a true baseline.
- `SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj` enables #38764.
- #39121 uses static shape dispatch and needs no environment flag.
- #41870 lets the target model auto-enable shared-expert fusion after validating its Quark layouts.
- The recipe intentionally does not pass `--enforce-shared-experts-fusion`: until #41258 lands, that flag would bypass the built-in MTP draft's architecture guard. The recipe also does not enable `SGLANG_GLM_NEXTN_MOE_PTPC`.
- INT4 QuickReduce is selected through both ROCm and AITER environment names.

An explicitly empty `SGLANG_OPT_GLM53_KDA_PTPC_MODULES` keeps the selector off for matched A/B runs.

## Usage

```bash
cd Agentic
IX=/path/to/InferenceX/inferencex-e2e \
  ./ix_agentx_glm53flash.sh --platform mi355x --gpus 4,5,6,7 --dry-run
```

Use `--smoke` only for plumbing. Its shortened duration is not valid final benchmark evidence.
