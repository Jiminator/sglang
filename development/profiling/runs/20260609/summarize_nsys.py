#!/usr/bin/env python3
"""Roll an nsys cuda_gpu_kern_sum CSV into the deliverable's kernel categories.

Run nsys first:
  nsys stats --report cuda_gpu_kern_sum --format csv --output - trace.nsys-rep > kern_sum.csv
then: summarize_nsys.py kern_sum.csv out.txt

cuda_gpu_kern_sum aggregates the WHOLE capture. With output-len 512 the decode
region (512 steps) dominates over the single prefill, so this is a decode-weighted
proxy; the authoritative clean decode split is the torch --profile-by-stage DECODE
trace. We report both and note the caveat.
"""
import sys, csv, collections

inp, out = sys.argv[1], sys.argv[2]

def classify(name):
    n = name.lower()
    if any(k in n for k in ("nccl", "allreduce", "all_reduce", "reduce_kernel", "custom_all_reduce", "one_shot", "two_shot", "cross_device_reduce", "allreduce_fusion", "lamport")):
        return "all-reduce"
    if any(k in n for k in ("topk", "top_k", "gathertopk", "hadamard", "signature", "logical_score", "logicalscore", "mqa_logits", "fp8_mqa", "index_k", "fused_store_index", "radixsort", "bitonicsort", "computeblockdigit", "computedigit", "computeblockwise", "per_token_group_quant", "scan_by_key", "searchsorted")):
        return "DS-index/scoring"
    if any(k in n for k in ("moe", "grouped_gemm", "group_gemm", "expert", "fused_experts", "silu", "topk_softmax")):
        return "MoE"
    if any(k in n for k in ("flash_fwd", "flashmla", "_mla_", "mla_combine", "attention", "fa3", "paged", "decode_attention", "splitkv")):
        return "attention(MLA)"
    if any(k in n for k in ("gemm", "cutlass", "nvjet", "scaled_mm", "fp8_gemm", "cublas", "tst_")):
        return "GEMM/proj"
    if any(k in n for k in ("memcpy", "memset")):
        return "memcpy/set"
    if any(k in n for k in ("norm", "rope", "rotary", "embed", "elementwise", "act_and_mul", "scatter_gather", "direct_copy", "copy")):
        return "norm/rope/elementwise"
    return "other"

# nsys csv columns vary by version; locate Time(%) / Total Time / Instances / Name.
rows = list(csv.reader(open(inp)))
# find header row
hdr_i = next((i for i, r in enumerate(rows) if any("Name" == c.strip() for c in r)), 0)
hdr = [c.strip() for c in rows[hdr_i]]
def col(*cands):
    for cand in cands:
        for i, c in enumerate(hdr):
            if c.lower().startswith(cand.lower()): return i
    return None
i_tot = col("Total Time", "Total")
i_name = col("Name")
i_inst = col("Instances", "Num Calls", "Count")
data = rows[hdr_i+1:]

kern = {}
for r in data:
    if not r or len(r) <= max(i_name, i_tot): continue
    name = r[i_name].strip().strip('"')
    if not name: continue
    try:
        tot = float(r[i_tot].replace(",", ""))
    except ValueError:
        continue
    inst = 0
    if i_inst is not None and i_inst < len(r):
        try: inst = int(float(r[i_inst].replace(",", "")))
        except ValueError: inst = 0
    kern[name] = (tot, inst)

total = sum(v[0] for v in kern.values()) or 1.0
cat = collections.defaultdict(lambda: [0.0, 0])
for nm, (t, c) in kern.items():
    k = classify(nm); cat[k][0] += t; cat[k][1] += c
catrows = sorted(cat.items(), key=lambda kv: -kv[1][0])
toprows = sorted(kern.items(), key=lambda kv: -kv[1][0])[:30]

with open(out, "w") as fh:
    fh.write(f"source: {inp}\n")
    fh.write(f"total GPU-kernel time units (ns)={total:.0f}  (whole capture; decode-dominated, see caveat)\n\n")
    fh.write("=== CATEGORY ROLLUP (% of GPU-kernel time) ===\n")
    fh.write(f"{'%':>7} {'calls':>10}  category\n")
    for c, (t, n) in catrows:
        fh.write(f"{100*t/total:7.1f} {n:10d}  {c}\n")
    fh.write("\n=== TOP 30 KERNELS ===\n")
    fh.write(f"{'%':>7} {'calls':>10}  kernel\n")
    for nm, (t, n) in toprows:
        fh.write(f"{100*t/total:7.1f} {n:10d}  {nm[:100]}\n")
print(f"wrote {out} ({len(kern)} kernels)")
