"""
Microbenchmark: compare sgl_kernel.rmsnorm vs jit_kernel.rmsnorm_hf vs sgl_kernel.rmsnorm_hf vs load_inline ext.
Run from the repo root: python exps/bench_rmsnorm_fp16.py
"""
import sys
import time

import torch

sys.path.insert(0, "python")
import sglang.srt.layers.layernorm as _ln
from sglang.srt.layers.layernorm import RMSNorm

if not torch.cuda.is_available():
    print("CUDA not available, skipping benchmark")
    sys.exit(0)

HIDDEN = 4096
EPS = 1e-5
WARMUP = 50
ITERS = 5000

# Direct jit_kernel import for isolated benchmarking
from sglang.jit_kernel.rmsnorm_hf import rmsnorm_hf as jit_rmsnorm_hf

print(f"{'M':>5}  {'variant':<35}  {'ms/call':>10}  {'ratio':>8}")
print("-" * 65)

for M in [1, 4, 16, 32, 64]:
    x = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda")

    norm_base = RMSNorm(HIDDEN, eps=EPS, weight_dtype=torch.bfloat16).cuda()

    results = {}

    def bench(label, fn):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / ITERS
        results[label] = ms

    # 1. sgl_kernel.rmsnorm (baseline — FlashInfer, fp32 weight mul)
    bench("sgl_kernel (baseline)", lambda: norm_base(x))

    # 2. jit_kernel.rmsnorm_hf (direct call — HF semantics)
    bench("jit_kernel.rmsnorm_hf", lambda: jit_rmsnorm_hf(x, w, EPS))

    # 3. sgl_kernel.rmsnorm_hf (the sgl-kernel op from Plan B)
    from sgl_kernel import rmsnorm_hf as _sgl_rmsnorm_hf
    bench("sgl_kernel.rmsnorm_hf", lambda: _sgl_rmsnorm_hf(x, w, EPS))

    # 4. load_inline CUDA ext
    ext = _ln._rmsnorm_fp16w_ext
    if ext is not None:
        bench("load_inline CUDA ext", lambda: ext.sglang_rmsnorm_fp16w(x, w, EPS))

    base_ms = results["sgl_kernel (baseline)"]
    for label, ms in results.items():
        ratio = ms / base_ms
        print(f"{M:>5}  {label:<35}  {ms:>10.4f}  {ratio:>8.3f}x")
    print()
