# Community 122

> 54 nodes

## Key Concepts

- **WarmupGate** (10 connections) — `python/sglang/srt/kv_canary/perturb/utils.py`
- **run()** (9 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_post_forward.py`
- **run()** (9 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_unused_cache.py`
- **run()** (9 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_used.py`
- **ReqToTokenEntry** (8 connections) — `python/sglang/srt/kv_canary/perturb/slot_picker.py`
- **TargetGroupKind** (7 connections) — `python/sglang/srt/kv_canary/perturb/config.py`
- **CanaryBufferGroup** (7 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_used.py`
- **pick_target_group()** (7 connections) — `python/sglang/srt/kv_canary/perturb/utils.py`
- **flip_first_byte_in_source()** (7 connections) — `python/sglang/srt/kv_canary/perturb/utils.py`
- **config.py** (6 connections) — `python/sglang/srt/kv_canary/perturb/config.py`
- **CanaryBufferGroup** (6 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_unused_cache.py`
- **PerturbConfig** (6 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_used.py`
- **WarmupGate** (6 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_used.py`
- **ReqToTokenEntry** (6 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_used.py`
- **utils.py** (6 connections) — `python/sglang/srt/kv_canary/perturb/utils.py`
- **should_run_perturbation()** (6 connections) — `python/sglang/srt/kv_canary/perturb/utils.py`
- **CanaryBufferGroup** (6 connections) — `python/sglang/srt/kv_canary/perturb/utils.py`
- **require_target_group_kind()** (5 connections) — `python/sglang/srt/kv_canary/perturb/config.py`
- **PerturbConfig** (5 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_unused_cache.py`
- **WarmupGate** (5 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_unused_cache.py`
- **_pick_sweep_slot_for_group()** (5 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_unused_cache.py`
- **Tensor** (5 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_unused_cache.py`
- **_pick_active_slot_for_group()** (5 connections) — `python/sglang/srt/kv_canary/perturb/real_kv_used.py`
- **flip_random_source_byte_and_log()** (5 connections) — `python/sglang/srt/kv_canary/perturb/utils.py`
- **_parse_target_group_kind_from_env()** (4 connections) — `python/sglang/srt/kv_canary/perturb/config.py`
- *... and 29 more nodes in this community*

## Relationships

- [[Vision-Language Model Configs]] (17 shared connections)
- [[Community 91]] (16 shared connections)
- [[Community 153]] (15 shared connections)
- [[Disaggregation Bootstrap & Decode]] (5 shared connections)
- [[HiCache Controller & Radix Tree]] (4 shared connections)
- [[Community 309]] (2 shared connections)
- [[Community 107]] (1 shared connections)
- [[Community 419]] (1 shared connections)
- [[Community 456]] (1 shared connections)

## Source Files

- `python/sglang/srt/kv_canary/perturb/config.py`
- `python/sglang/srt/kv_canary/perturb/real_kv_post_forward.py`
- `python/sglang/srt/kv_canary/perturb/real_kv_unused_cache.py`
- `python/sglang/srt/kv_canary/perturb/real_kv_used.py`
- `python/sglang/srt/kv_canary/perturb/slot_picker.py`
- `python/sglang/srt/kv_canary/perturb/utils.py`

## Audit Trail

- EXTRACTED: 136 (59%)
- INFERRED: 94 (41%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*