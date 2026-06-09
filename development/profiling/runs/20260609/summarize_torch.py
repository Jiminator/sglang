#!/usr/bin/env python3
"""Summarize a torch-profiler trace dir into a decode-stage GPU-kernel breakdown.

Adapts the aggregation block from development/profile_ds.sh: walk the trace dir,
sum chrome-trace events with cat in {kernel, gpu_memcpy, gpu_memset} by name, take
the top-N by total dur. Handles --profile-by-stage (separate prefill/decode files)
by picking the DECODE trace, and picks rank TP-0 as representative.

Usage: summarize_torch.py <trace_dir> <out.txt> [stage=decode]
"""
import sys, os, gzip, json, collections, re

trace_dir, out = sys.argv[1], sys.argv[2]
want_stage = sys.argv[3] if len(sys.argv) > 3 else "decode"

# Buckets for classifying kernels into the deliverable's categories.
def classify(name):
    n = name.lower()
    if any(k in n for k in ("nccl", "allreduce", "all_reduce", "reduce_kernel", "custom_all_reduce", "one_shot", "two_shot", "cross_device_reduce")):
        return "all-reduce"
    # DS-specific index/scoring stack
    if any(k in n for k in ("topk", "top_k", "gathertopk", "hadamard", "signature", "logical_score", "logicalscore", "sparse_mla", "sparsemla", "mqa_logits", "fp8_mqa", "index_k", "fused_store_index")):
        return "DS-index/scoring"
    if any(k in n for k in ("moe", "grouped_gemm", "group_gemm", "expert", "fused_experts", "silu", "topk_softmax")):
        return "MoE"
    if any(k in n for k in ("attn", "attention", "flashmla", "mla", "flash_fwd", "fa3", "paged", "decode_attention")):
        return "attention"
    if any(k in n for k in ("gemm", "cutlass", "matmul", "scaled_mm", "fp8", "linear", "cublas")):
        return "GEMM/proj"
    if any(k in n for k in ("memcpy", "memset", "copy")):
        return "memcpy/set"
    if any(k in n for k in ("norm", "rmsnorm", "layernorm", "rope", "rotary", "embed", "elementwise", "add", "act")):
        return "norm/rope/elementwise"
    return "other"

# Collect candidate trace files, partition by stage hint in the path/name.
allf = []
for root, _, fs in os.walk(trace_dir):
    for fn in fs:
        if fn.endswith(".trace.json.gz") or fn.endswith(".trace.json") or fn.endswith(".json"):
            allf.append(os.path.join(root, fn))
if not allf:
    open(out, "w").write(f"NO TRACE FILES under {trace_dir}\n"); print("FAIL: no trace files", file=sys.stderr); sys.exit(2)

def stage_of(p):
    pl = p.lower()
    if "decode" in pl: return "decode"
    if "prefill" in pl: return "prefill"
    return "unknown"

def rank_of(p):
    m = re.search(r"tp[-_]?(\d+)", os.path.basename(p).lower())
    return int(m.group(1)) if m else 999

staged = [f for f in allf if stage_of(f) == want_stage]
pool = staged if staged else allf  # fall back to all if no stage tag
pool.sort(key=lambda p: (rank_of(p), p))
f = pool[0]  # TP-0 (or lowest rank), representative

op = gzip.open(f, "rt") if f.endswith(".gz") else open(f)
data = json.load(op)
evs = data.get("traceEvents", data) if isinstance(data, dict) else data

kern = collections.defaultdict(lambda: [0.0, 0])
for e in evs:
    if not isinstance(e, dict): continue
    if e.get("cat") in ("kernel", "Kernel", "gpu_memcpy", "gpu_memset") and "dur" in e:
        nm = e.get("name", "?")
        kern[nm][0] += float(e["dur"]); kern[nm][1] += 1
if not kern:
    open(out, "w").write(f"NO GPU-KERNEL ROWS in {os.path.basename(f)} ({len(evs)} events)\n")
    print("FAIL: zero GPU-kernel events", file=sys.stderr); sys.exit(3)

tot = sum(v[0] for v in kern.values()) or 1.0
# category rollup
cat = collections.defaultdict(lambda: [0.0, 0])
for nm, (d, n) in kern.items():
    c = classify(nm); cat[c][0] += d; cat[c][1] += n
rows = sorted(kern.items(), key=lambda kv: -kv[1][0])[:30]
catrows = sorted(cat.items(), key=lambda kv: -kv[1][0])

with open(out, "w") as fh:
    fh.write(f"trace file : {f}\n")
    fh.write(f"stage      : {want_stage}  (matched {len(staged)} staged files; {len(allf)} total)\n")
    fh.write(f"events={len(evs)}  total GPU-kernel us={tot:.0f}\n\n")
    fh.write("=== CATEGORY ROLLUP (% of GPU-kernel time) ===\n")
    fh.write(f"{'tot_us':>14} {'%':>6} {'calls':>9}  category\n")
    for c, (d, n) in catrows:
        fh.write(f"{d:14.0f} {100*d/tot:6.1f} {n:9d}  {c}\n")
    fh.write("\n=== TOP 30 KERNELS ===\n")
    fh.write(f"{'tot_us':>14} {'%':>6} {'calls':>9}  kernel\n")
    for nm, (d, n) in rows:
        fh.write(f"{d:14.0f} {100*d/tot:6.1f} {n:9d}  {nm[:100]}\n")
print(f"wrote {out} from {os.path.basename(f)} ({len(kern)} kernels, stage={want_stage})")
