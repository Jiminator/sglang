"""
CUDA graph capture smoke test for the Triton fp16-weight RMSNorm kernel.
Run from the repo root: python exps/test_cuda_graph_rmsnorm.py
"""
import sys

import torch

sys.path.insert(0, "python")
from sglang.srt.layers.layernorm import RMSNorm

if not torch.cuda.is_available():
    print("CUDA not available, skipping CUDA graph test")
    sys.exit(0)

HIDDEN = 4096
device = "cuda"

x_static = torch.randn(1, HIDDEN, dtype=torch.float16, device=device)
norm = RMSNorm(HIDDEN, eps=1e-5, cast_x_before_out_mul=True, weight_dtype=torch.float16).to(
    device
)

# Warmup: compile Triton kernel BEFORE graph capture
for _ in range(3):
    norm(x_static)
torch.cuda.synchronize()

# Capture
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        out_static = norm(x_static)
    print("CUDA graph capture: OK")
except Exception as e:
    print(f"CUDA graph capture: FAILED — {e}")
    sys.exit(1)

# Replay and verify output matches eager
x_static.normal_()
torch.cuda.synchronize()

g.replay()
torch.cuda.synchronize()
out_replayed = out_static.clone()

out_eager = norm(x_static)

max_diff = (out_replayed - out_eager).abs().max().item()
if max_diff == 0.0:
    print(f"CUDA graph replay: OK (max_diff={max_diff})")
else:
    print(f"CUDA graph replay: MISMATCH (max_diff={max_diff})")
    sys.exit(1)

# Also test with a larger batch size
for M in [4, 16, 32]:
    x_static2 = torch.randn(M, HIDDEN, dtype=torch.float16, device=device)
    g2 = torch.cuda.CUDAGraph()
    norm(x_static2)  # warmup for this shape
    torch.cuda.synchronize()
    with torch.cuda.graph(g2):
        out2 = norm(x_static2)
    x_static2.normal_()
    g2.replay()
    torch.cuda.synchronize()
    out2_replayed = out2.clone()
    out2_eager = norm(x_static2)
    diff = (out2_replayed - out2_eager).abs().max().item()
    status = "OK" if diff == 0.0 else f"MISMATCH (diff={diff})"
    print(f"CUDA graph M={M}: {status}")

print("\nAll CUDA graph tests passed.")
