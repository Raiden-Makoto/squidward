#!/usr/bin/env python3
"""Bucket a glm5_prof_csv.py per-kernel CSV into the profile-table sections
(per prefill forward pass, ms). AllReduce is reported but excluded from the
compute total (carried-over clean numbers are used in the md instead)."""
import csv
import collections
import sys

NF = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rows = list(csv.reader(open(sys.argv[1])))
hdr = rows[0]
iB, iS, iK, iT = (hdr.index(x) for x in
                  ("Block", "DecoderSubmodule", "KernelName", "Total_us"))

bucket = collections.defaultdict(float)
gemm_detail = collections.defaultdict(float)

for r in rows[1:]:
    blk, sub, kern, us = r[iB], r[iS], r[iK].strip('"'), float(r[iT])
    ms = us / NF / 1000.0
    kl = kern.lower()
    if any(x in kl for x in
           ("cross_device_reduce", "allreduce", "allgather", "quickreduce")):
        bucket["AllReduce (excluded)"] += ms
        continue
    if "mfma_moe1" in kl:
        bucket["MoE gemm1 (mfma_moe1)"] += ms
    elif "mfma_moe2" in kl:
        bucket["MoE gemm2 (mfma_moe2)"] += ms
    elif "moe_reduction" in kl:
        bucket["MoE combine"] += ms
    elif "dynamic_per_group_scaled_quant" in kl:
        bucket["MoE actquant"] += ms
    elif "shared_experts" in kl or "moe_sort" in kl:
        bucket["MoE other"] += ms
    elif "ck_tile" in kl and "fmha" in kl:
        bucket["Attn FMHA (ck_tile)"] += ms
    elif "set_mla_kv_buffer" in kl or "cached_indirect" in kl:
        bucket["Attn kv-write"] += ms
    elif "elementwise_kernel" in kl and blk == "L" and sub == "Attn(MLA)":
        bucket["Attn kv-write"] += ms  # bf16->fp8 cast + concat for KV write
    elif "indexer_k_quant" in kl or "hadamard" in kl:
        bucket["Attn indexer"] += ms
    elif "layernorm2d" in kl:
        bucket["Attn qk-norm"] += ms
    elif "add_rmsnorm_quant" in kl:
        if blk == "L" and sub == "Attn(MLA)":
            bucket["Attn qk-norm"] += ms
        else:
            bucket["RMSNorm"] += ms
    elif ("cijk" in kl) or ("bf16gemm" in kl) or ("hgemm_bf16" in kl):
        # dense projection / router / dense-MLP GEMMs
        bucket["Dense GEMM"] += ms
        gemm_detail[(blk, sub)] += ms
    else:
        bucket["misc: %s/%s" % (blk, sub)] += ms
        if ms > 0.01:
            sys.stderr.write("MISC %6.2f %s/%s %s\n" % (ms, blk, sub, kern[:70]))

order = ["Attn FMHA (ck_tile)", "Attn kv-write", "Attn qk-norm", "Attn indexer",
         "MoE gemm1 (mfma_moe1)", "MoE gemm2 (mfma_moe2)", "MoE combine",
         "MoE actquant", "MoE other", "Dense GEMM", "RMSNorm",
         "AllReduce (excluded)"]
print("per prefill forward (ms), n_fwd=%d" % NF)
for k in order:
    if k in bucket:
        print("  %-26s %8.2f" % (k, bucket[k]))
for k in sorted(bucket):
    if k not in order:
        print("  %-26s %8.2f" % (k, bucket[k]))

attn = sum(bucket[k] for k in bucket if k.startswith("Attn"))
moe = sum(bucket[k] for k in bucket if k.startswith("MoE"))
comp = sum(v for k, v in bucket.items() if not k.startswith("AllReduce"))
print("-" * 40)
print("  Attention subtotal        %8.2f" % attn)
print("  MoE subtotal              %8.2f" % moe)
print("  Dense GEMM subtotal       %8.2f" % bucket["Dense GEMM"])
print("  RMSNorm subtotal          %8.2f" % bucket["RMSNorm"])
print("  COMPUTE total (excl AR)   %8.2f" % comp)
print("  (AllReduce profiler-inflated, excluded: %.2f)" %
      bucket["AllReduce (excluded)"])
print("--- Dense GEMM by block/submodule ---")
for k in sorted(gemm_detail, key=lambda x: -gemm_detail[x]):
    print("  %-28s %8.2f" % (str(k), gemm_detail[k]))
