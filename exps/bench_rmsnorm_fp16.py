"""
Microbenchmark: compare sgl_kernel.rmsnorm vs B-CUDA ext vs torch.compile vs sgl_kernel.rmsnorm_hf.
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

print(f"{'M':>5}  {'variant':<35}  {'ms/call':>10}  {'ratio':>8}")
print("-" * 65)

for M in [1, 4, 16, 32, 64]:
    x = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device="cuda")

    norm_base = RMSNorm(HIDDEN, eps=EPS, weight_dtype=torch.bfloat16).cuda()

    # B-CUDA: load_inline ext active (disable rmsnorm_hf path by monkey-patching)
    norm_cuda_ext = RMSNorm(
        HIDDEN, eps=EPS, cast_x_before_out_mul=True, weight_dtype=torch.bfloat16
    ).cuda()
    norm_cuda_ext.weight.data.copy_(norm_base.weight.data)

    # torch.compile: disable both rmsnorm_hf and CUDA ext
    norm_compiled = RMSNorm(
        HIDDEN, eps=EPS, cast_x_before_out_mul=True, weight_dtype=torch.bfloat16
    ).cuda()
    norm_compiled.weight.data.copy_(norm_base.weight.data)

    # sgl_kernel.rmsnorm_hf: natural dispatch (rmsnorm_hf is primary)
    norm_hf = RMSNorm(
        HIDDEN, eps=EPS, cast_x_before_out_mul=True, weight_dtype=torch.bfloat16
    ).cuda()
    norm_hf.weight.data.copy_(norm_base.weight.data)

    results = {}

    def bench(label, norm, patch_rmsnorm_hf=None, patch_ext=None):
        saved_hf = _ln.__dict__.get("rmsnorm_hf", None)
        saved_ext = _ln._rmsnorm_fp16w_ext

        # Monkey-patch to control which path is taken
        if patch_rmsnorm_hf is not None:
            _ln.rmsnorm_hf = patch_rmsnorm_hf
            # Also patch the name in the module globals used by forward_cuda
            import sglang.srt.layers.layernorm as ln_mod
            ln_mod.rmsnorm_hf = patch_rmsnorm_hf
        if patch_ext is not None:
            _ln._rmsnorm_fp16w_ext = patch_ext

        for _ in range(WARMUP):
            norm(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            norm(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / ITERS
        results[label] = ms

        # Restore
        if patch_rmsnorm_hf is not None:
            import sglang.srt.layers.layernorm as ln_mod
            ln_mod.rmsnorm_hf = saved_hf
        if patch_ext is not None:
            _ln._rmsnorm_fp16w_ext = saved_ext

    # Import the real rmsnorm_hf to patch around
    from sgl_kernel import rmsnorm_hf as _real_rmsnorm_hf

    def _raise(*a, **kw):
        raise RuntimeError("disabled")

    bench("sgl_kernel (baseline)", norm_base)
    bench("sgl_kernel.rmsnorm_hf", norm_hf)  # natural dispatch
    bench("B-CUDA load_inline ext", norm_cuda_ext, patch_rmsnorm_hf=_raise)
    bench("torch.compile (no ext)", norm_compiled, patch_rmsnorm_hf=_raise, patch_ext=None)

    base_ms = results["sgl_kernel (baseline)"]
    for label, ms in results.items():
        ratio = ms / base_ms
        print(f"{M:>5}  {label:<35}  {ms:>10.4f}  {ratio:>8.3f}x")
    print()
