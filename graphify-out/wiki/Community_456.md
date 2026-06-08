# Community 456

> 13 nodes

## Key Concepts

- **build_verify_plan_radix_sweep()** (7 connections) — `python/sglang/srt/kv_canary/sweep_plan_builder.py`
- **walk_radix_cache_for_canary()** (6 connections) — `python/sglang/srt/kv_canary/radix_cache_walker.py`
- **radix_cache_walker.py** (5 connections) — `python/sglang/srt/kv_canary/radix_cache_walker.py`
- **_walk_radix_subtree()** (4 connections) — `python/sglang/srt/kv_canary/radix_cache_walker.py`
- **.maybe_run_sweep()** (3 connections) — `python/sglang/srt/kv_canary/runner/sweep.py`
- **Tensor** (3 connections) — `python/sglang/srt/kv_canary/sweep_plan_builder.py`
- **_swa_translate()** (3 connections) — `python/sglang/srt/kv_canary/sweep_plan_builder.py`
- **_node_is_unlocked_for_canary()** (2 connections) — `python/sglang/srt/kv_canary/radix_cache_walker.py`
- **_node_is_swa_resident_for_canary()** (2 connections) — `python/sglang/srt/kv_canary/radix_cache_walker.py`
- **sweep_plan_builder.py** (2 connections) — `python/sglang/srt/kv_canary/sweep_plan_builder.py`
- **VerifyPlan** (2 connections) — `python/sglang/srt/kv_canary/sweep_plan_builder.py`
- **Walk the radix tree and emit flat (slot_indices, positions, prev_slot_indices) t** (1 connections) — `python/sglang/srt/kv_canary/radix_cache_walker.py`
- **Build a sweep VerifyPlan directly from the radix-cache walker.      The walker c** (1 connections) — `python/sglang/srt/kv_canary/sweep_plan_builder.py`

## Relationships

- [[HiCache Controller & Radix Tree]] (4 shared connections)
- [[Community 122]] (1 shared connections)
- [[Community 91]] (1 shared connections)
- [[Community 56]] (1 shared connections)

## Source Files

- `python/sglang/srt/kv_canary/radix_cache_walker.py`
- `python/sglang/srt/kv_canary/runner/sweep.py`
- `python/sglang/srt/kv_canary/sweep_plan_builder.py`

## Audit Trail

- EXTRACTED: 33 (80%)
- INFERRED: 8 (20%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*