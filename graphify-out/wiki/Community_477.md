# Community 477

> 12 nodes

## Key Concepts

- **fused_topk()** (6 connections) — `python/sglang/srt/layers/moe/topk.py`
- **topk.py** (5 connections) — `python/sglang/srt/hardware_backend/musa/kernels/topk.py`
- **topk_sigmoid()** (5 connections) — `python/sglang/srt/hardware_backend/musa/kernels/topk.py`
- **topk_softmax()** (4 connections) — `python/sglang/srt/hardware_backend/musa/kernels/topk.py`
- **topk_softmax_triton_kernel()** (3 connections) — `python/sglang/srt/hardware_backend/musa/kernels/topk.py`
- **.forward_torch()** (3 connections) — `python/sglang/srt/layers/moe/router.py`
- **tanh()** (2 connections) — `python/sglang/srt/hardware_backend/musa/kernels/topk.py`
- **constexpr** (2 connections) — `python/sglang/srt/hardware_backend/musa/kernels/topk.py`
- **Tensor** (2 connections) — `python/sglang/srt/hardware_backend/musa/kernels/topk.py`
- **topk_sigmoid_triton_kernel()** (2 connections) — `python/sglang/srt/hardware_backend/musa/kernels/topk.py`
- **Compute top-k softmax for MoE routing.      Args:         topk_weights: Output t** (1 connections) — `python/sglang/srt/hardware_backend/musa/kernels/topk.py`
- **Compute top-k sigmoid for MoE routing.      Args:         topk_weights: Output t** (1 connections) — `python/sglang/srt/hardware_backend/musa/kernels/topk.py`

## Relationships

- [[Community 213]] (3 shared connections)
- [[Community 461]] (2 shared connections)
- [[Community 396]] (1 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/musa/kernels/topk.py`
- `python/sglang/srt/layers/moe/router.py`
- `python/sglang/srt/layers/moe/topk.py`

## Audit Trail

- EXTRACTED: 29 (81%)
- INFERRED: 7 (19%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*