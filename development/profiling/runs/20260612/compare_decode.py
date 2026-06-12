#!/usr/bin/env python3
"""Join two torch DECODE traces kernel-by-kernel; emit per-kernel us + delta and a
category rollup. Used for the Case1-vs-Case2 (clean DS overhead) and
Case2-vs-Case3 (DSA batch-efficiency) comparisons.

Usage: compare_decode.py <trace_dirA> <labelA> <trace_dirB> <labelB> <out.txt>
Per-kernel rows are normalized to per-decode-step where possible by dividing the
aggregated dur by the number of decode steps inferred from the layer-periodic
kernel call count (calls / 78 layers); but we report raw window totals (both traces
capture the same --profile-steps window) so A vs B is directly comparable.
"""
import sys, os, gzip, json, collections, re

def classify(name):
    n = name.lower()
    if any(k in n for k in ("nccl", "allreduce", "all_reduce", "reduce_kernel", "custom_all_reduce", "one_shot", "two_shot", "oneshot", "lamport", "cross_device")):
        return "all-reduce"
    if "_logical_score" in n or "logical_score" in n:
        return "DS:logical-score"
    if any(k in n for k in ("mbtopk", "radixsort", "sbtopk", "searchsorted", "scan_by_key", "bitonicsort", "computedigit", "computeblock")):
        return "topk/sort stack"
    if "paged_mqa_logits" in n or "mqa_logits" in n:
        return "DSA:mqa-logits"
    if "topk_transform" in n:
        return "DSA:topk-transform"
    if "hadamard" in n or "signature" in n:
        return "hadamard/sig"
    if "per_token_group_quant" in n or "_act_quant" in n:
        return "fp8-quant(index)"
    if any(k in n for k in ("moe", "expert", "silu", "moe_align", "moe_sum")):
        return "MoE"
    if any(k in n for k in ("flash_fwd", "flashmla", "_mla_", "mla_combine", "mla_absorb", "attention", "splitkv")):
        return "attention(MLA)"
    if any(k in n for k in ("gemm", "cutlass", "nvjet", "scaled_mm", "cublas", "splitkreduce")):
        return "GEMM/proj"
    if any(k in n for k in ("memcpy", "memset")):
        return "memcpy/set"
    if any(k in n for k in ("norm", "rope", "rotary", "embed", "elementwise", "act_and_mul", "scatter_gather", "direct_copy", "copy", "concat", "distribution")):
        return "norm/rope/elementwise"
    return "other"

def load(trace_dir, stage="decode"):
    allf = []
    for root, _, fs in os.walk(trace_dir):
        for fn in fs:
            if fn.endswith(".trace.json.gz") or fn.endswith(".trace.json"):
                allf.append(os.path.join(root, fn))
    staged = [f for f in allf if stage in os.path.basename(f).lower()]
    def rank_of(p):
        m = re.search(r"tp[-_]?(\d+)", os.path.basename(p).lower())
        return int(m.group(1)) if m else 999
    pool = sorted(staged or allf, key=lambda p: (rank_of(p), p))
    f = pool[0]
    op = gzip.open(f, "rt") if f.endswith(".gz") else open(f)
    data = json.load(op)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    kern = collections.defaultdict(float)
    for e in evs:
        if isinstance(e, dict) and e.get("cat") in ("kernel", "Kernel", "gpu_memcpy", "gpu_memset") and "dur" in e:
            kern[e.get("name", "?")] += float(e["dur"])
    return f, kern

dirA, labA, dirB, labB, out = sys.argv[1:6]
fA, kA = load(dirA); fB, kB = load(dirB)
totA = sum(kA.values()) or 1.0; totB = sum(kB.values()) or 1.0
names = sorted(set(kA) | set(kB), key=lambda n: -(kA.get(n,0)+kB.get(n,0)))

# category rollup
catA = collections.defaultdict(float); catB = collections.defaultdict(float)
for n in names:
    c = classify(n); catA[c]+=kA.get(n,0); catB[c]+=kB.get(n,0)
cats = sorted(set(catA)|set(catB), key=lambda c: -(catA[c]+catB[c]))

with open(out, "w") as fh:
    fh.write(f"A = {labA}  ({os.path.basename(fA)})  total_us={totA:.0f}\n")
    fh.write(f"B = {labB}  ({os.path.basename(fB)})  total_us={totB:.0f}\n")
    fh.write(f"A/B total ratio = {totA/totB:.2f}x\n\n")
    fh.write("=== CATEGORY ROLLUP (us, and % of own total) ===\n")
    fh.write(f"{'category':24} {'A_us':>10} {'A%':>6} {'B_us':>10} {'B%':>6} {'A-B us':>10}\n")
    for c in cats:
        a=catA[c]; b=catB[c]
        fh.write(f"{c:24} {a:10.0f} {100*a/totA:6.1f} {b:10.0f} {100*b/totB:6.1f} {a-b:10.0f}\n")
    fh.write("\n=== PER-KERNEL (us): A vs B, sorted by |A-B| ===\n")
    fh.write(f"{'A_us':>10} {'B_us':>10} {'A-B':>10}  kernel\n")
    for n in sorted(names, key=lambda n: -abs(kA.get(n,0)-kB.get(n,0)))[:40]:
        a=kA.get(n,0); b=kB.get(n,0)
        fh.write(f"{a:10.0f} {b:10.0f} {a-b:10.0f}  {n[:88]}\n")
print(f"wrote {out}")
