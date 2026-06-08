# Community 95

> 66 nodes

## Key Concepts

- **HiCacheController** (90 connections) — `python/sglang/srt/managers/cache_controller.py`
- **Tensor** (20 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.attach_storage_backend()** (12 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.prefetch_thread_func()** (9 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.start_loading()** (8 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.put()** (6 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.get()** (6 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.clear()** (6 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.detach_storage_backend()** (6 connections) — `python/sglang/srt/managers/cache_controller.py`
- **._stop_storage_threads()** (5 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.prefetch()** (5 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.append_host_mem_release()** (5 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.prefetch_io_aux_func()** (5 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.write_storage()** (5 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.backup_thread_func()** (5 connections) — `python/sglang/srt/managers/cache_controller.py`
- **._start_storage_threads()** (4 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.load()** (4 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.move_indices()** (4 connections) — `python/sglang/srt/managers/cache_controller.py`
- **._maybe_register_draft_with_storage()** (4 connections) — `python/sglang/srt/managers/cache_controller.py`
- **._page_transfer()** (4 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.__init__()** (3 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.increment()** (3 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.mark_terminate()** (3 connections) — `python/sglang/srt/managers/cache_controller.py`
- **.get_attn_cp_rank_and_size()** (3 connections) — `python/sglang/srt/managers/cache_controller.py`
- **._create_prefetch_sync_groups()** (3 connections) — `python/sglang/srt/managers/cache_controller.py`
- *... and 41 more nodes in this community*

## Relationships

- [[Aibrix KV Cache Storage]] (46 shared connections)
- [[HiCache Controller & Radix Tree]] (22 shared connections)
- [[Disaggregation Utils & Cache Tests]] (14 shared connections)
- [[Context-Parallel Attention]] (3 shared connections)
- [[Community 47]] (3 shared connections)
- [[Community 81]] (2 shared connections)
- [[Grammar Manager & HiCache Clear]] (1 shared connections)
- [[Breakable CUDA Graph (TBO)]] (1 shared connections)

## Source Files

- `python/sglang/srt/managers/cache_controller.py`

## Audit Trail

- EXTRACTED: 232 (79%)
- INFERRED: 62 (21%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*