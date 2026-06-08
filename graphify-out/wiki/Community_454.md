# Community 454

> 13 nodes

## Key Concepts

- **triton_kernel_fused_experts_with_bias()** (8 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- **triton_kernel_fused_experts()** (7 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- **triton_kernels_moe.py** (5 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- **triton_kernel_moe_forward()** (5 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- **triton_kernel_moe_with_bias_forward()** (5 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- **routing()** (5 connections) — `python/sglang/srt/layers/moe/topk.py`
- **Tensor** (4 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- **RoutingData** (3 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- **GatherIndx** (3 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- **ScatterIndx** (3 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- **quantize()** (2 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- **TopKOutput** (2 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- **MoeRunnerConfig** (2 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`

## Relationships

- [[MoE Dispatch/Combine (Cutlass)]] (2 shared connections)
- [[Community 213]] (1 shared connections)
- [[Community 396]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- `python/sglang/srt/layers/moe/topk.py`

## Audit Trail

- EXTRACTED: 46 (85%)
- INFERRED: 8 (15%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*