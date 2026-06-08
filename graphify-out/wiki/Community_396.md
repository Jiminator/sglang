# Community 396

> 16 nodes

## Key Concepts

- **select_experts()** (19 connections) — `python/sglang/srt/layers/moe/topk.py`
- **.forward_cuda()** (14 connections) — `python/sglang/srt/layers/moe/topk.py`
- **TopKOutput** (12 connections) — `python/sglang/srt/layers/moe/topk.py`
- **ExpertLocationDispatchInfo** (10 connections) — `python/sglang/srt/layers/moe/topk.py`
- **._apply_deepep_waterfill()** (8 connections) — `python/sglang/srt/layers/moe/topk.py`
- **fused_topk_npu()** (7 connections) — `python/sglang/srt/hardware_backend/npu/moe/topk.py`
- **.forward_native()** (6 connections) — `python/sglang/srt/layers/moe/topk.py`
- **.forward_cpu()** (6 connections) — `python/sglang/srt/layers/moe/topk.py`
- **_make_round_robin_expert_ids()** (5 connections) — `python/sglang/srt/layers/moe/topk.py`
- **.forward_npu()** (5 connections) — `python/sglang/srt/layers/moe/topk.py`
- **.forward_xpu()** (5 connections) — `python/sglang/srt/layers/moe/topk.py`
- **device** (3 connections) — `python/sglang/srt/layers/moe/topk.py`
- **Tensor** (2 connections) — `python/sglang/srt/hardware_backend/npu/moe/topk.py`
- **dtype** (2 connections) — `python/sglang/srt/layers/moe/topk.py`
- **topk.py** (1 connections) — `python/sglang/srt/hardware_backend/npu/moe/topk.py`
- **Protocol for top-k outputs in different formats.** (1 connections) — `python/sglang/srt/layers/moe/topk.py`

## Relationships

- [[Community 213]] (16 shared connections)
- [[DeepSeek MLA Attention & MoE]] (7 shared connections)
- [[Context-Parallel Attention]] (7 shared connections)
- [[NCCL Symmetric Memory]] (6 shared connections)
- [[Community 395]] (6 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (2 shared connections)
- [[Community 526]] (1 shared connections)
- [[Hybrid Attention Backend]] (1 shared connections)
- [[Batch-Overlap Operations]] (1 shared connections)
- [[Community 47]] (1 shared connections)
- [[Community 833]] (1 shared connections)
- [[Community 454]] (1 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/npu/moe/topk.py`
- `python/sglang/srt/layers/moe/topk.py`

## Audit Trail

- EXTRACTED: 85 (80%)
- INFERRED: 21 (20%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*