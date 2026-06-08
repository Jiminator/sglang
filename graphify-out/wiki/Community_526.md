# Community 526

> 10 nodes

## Key Concepts

- **.forward()** (10 connections) — `python/sglang/srt/layers/moe/hash_topk.py`
- **topk_ids_logical_to_physical()** (8 connections) — `python/sglang/srt/eplb/expert_location_dispatch.py`
- **expert_location_dispatch.py** (5 connections) — `python/sglang/srt/eplb/expert_location_dispatch.py`
- **Tensor** (4 connections) — `python/sglang/srt/eplb/expert_location_dispatch.py`
- **_topk_ids_logical_to_physical_static()** (4 connections) — `python/sglang/srt/eplb/expert_location_dispatch.py`
- **_topk_ids_logical_to_physical_dynamic()** (4 connections) — `python/sglang/srt/eplb/expert_location_dispatch.py`
- **transform_select_experts_inputs()** (3 connections) — `python/sglang/srt/eplb/expert_location_dispatch.py`
- **._forward_torch()** (3 connections) — `python/sglang/srt/layers/moe/hash_topk.py`
- **Tensor** (3 connections) — `python/sglang/srt/layers/moe/hash_topk.py`
- **ExpertLocationDispatchInfo** (2 connections) — `python/sglang/srt/layers/moe/hash_topk.py`

## Relationships

- [[DeepSeek MLA Attention & MoE]] (8 shared connections)
- [[Community 213]] (3 shared connections)
- [[Hybrid Attention Backend]] (2 shared connections)
- [[Community 47]] (2 shared connections)
- [[Community 396]] (1 shared connections)

## Source Files

- `python/sglang/srt/eplb/expert_location_dispatch.py`
- `python/sglang/srt/layers/moe/hash_topk.py`

## Audit Trail

- EXTRACTED: 37 (80%)
- INFERRED: 9 (20%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*