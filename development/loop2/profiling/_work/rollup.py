import gzip, json, sys, re
from collections import defaultdict

path = sys.argv[1]
op = gzip.open(path, 'rt') if path.endswith('.gz') else open(path)
data = json.load(op)
events = data.get("traceEvents", data) if isinstance(data, dict) else data

# kernel events: have 'dur' and 'cat' in {'kernel'} on a GPU stream; chrome trace
# uses ph 'X'. We bucket GPU-side kernel ops by name into loop2 categories.
def category(name):
    n = name.lower()
    if 'fused_moe' in n or 'moe_sum_reduce' in n or 'moe_align' in n or 'grouped_topk' in n.replace('group','grouped'):
        return 'MoE'
    if 'flashattn' in n or 'flash::' in n or 'fmha' in n or 'attnfwd' in n:
        return 'Attn(MLA/DSA)'
    if 'paged_mqa_logits' in n or 'index' in n and 'cache' in n:
        return 'DSA-indexer'
    if 'topk' in n or 'gathertopk' in n or 'bitonicsort' in n:
        return 'topk/indexer'
    if 'allreduce' in n or 'all_reduce' in n or 'reduce_scatter' in n or 'all_gather' in n or 'nccl' in n:
        return 'Comms'
    if 'quant' in n:
        return 'Quantize'
    if 'gemm' in n or 'cutlass' in n or 'nvjet' in n or 'matmul' in n:
        return 'GEMM(dense/other)'
    if 'elementwise' in n or 'rmsnorm' in n or 'layernorm' in n or 'rotary' in n or 'silu' in n or 'add' in n or 'cast' in n or 'copy' in n:
        return 'elementwise/norm'
    return 'other'

cat_time = defaultdict(float)
cat_cnt = defaultdict(int)
total = 0.0
for e in events:
    if not isinstance(e, dict): continue
    if e.get('ph') != 'X': continue
    if e.get('cat') != 'kernel': continue
    dur = e.get('dur', 0) or 0
    nm = e.get('name','')
    c = category(nm)
    cat_time[c] += dur
    cat_cnt[c] += 1
    total += dur

print(f"total_kernel_us={total:.0f}  ({total/1e3:.1f} ms)")
print(f"{'category':22s} {'ms':>10s} {'share%':>8s} {'launches':>10s}")
for c in sorted(cat_time, key=lambda k:-cat_time[k]):
    print(f"{c:22s} {cat_time[c]/1e3:>10.2f} {100*cat_time[c]/total:>7.1f}% {cat_cnt[c]:>10d}")
