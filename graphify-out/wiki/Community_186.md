# Community 186

> 37 nodes

## Key Concepts

- **Tensor** (10 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **._build_dispatch_plan()** (10 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **._materialize_dispatch()** (9 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **.expand_topk()** (9 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **deepep_waterfill.py** (7 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **WaterfillDispatchPlan** (7 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **_empty_expanded()** (6 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **._all_reduce_dynamic_rank_load()** (6 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **._with_expanded_topk()** (6 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **materialize_waterfill_dispatch_fused()** (5 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **expand_topk_with_shared_expert()** (5 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **._is_low_batch()** (5 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **._build_static_dispatch_plan()** (5 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **._build_dynamic_dispatch_plan()** (5 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **._expand_local_shared()** (5 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **.count_local_routed()** (4 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **._can_skip_dispatch_plan_for_low_batch()** (4 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **_count_routed_per_rank_kernel()** (3 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **_waterfill_expand_kernel()** (3 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **StandardTopKOutput** (3 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **constexpr** (2 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **Inputs needed by the fused DeepEP Waterfill expansion path.** (1 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **Return empty expanded tensors for zero-token batches.** (1 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **Count routed tokens per rank using block-level histogram.** (1 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- **Fused waterfill + expand. ID remap: old_id -> old_id + old_id // old_epr.** (1 connections) — `python/sglang/srt/layers/moe/deepep_waterfill.py`
- *... and 12 more nodes in this community*

## Relationships

- [[Hybrid Attention Backend]] (12 shared connections)
- [[Qwen3 / Kimi Model Configs]] (2 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/moe/deepep_waterfill.py`

## Audit Trail

- EXTRACTED: 133 (99%)
- INFERRED: 2 (1%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*