# HiCache Controller & Radix Tree

> 473 nodes

## Key Concepts

- **Req** (601 connections) — `python/sglang/srt/managers/schedule_batch.py`
- **EvictParams** (340 connections) — `python/sglang/srt/mem_cache/base_prefix_cache.py`
- **BasePrefixCache** (310 connections) — `python/sglang/srt/mem_cache/base_prefix_cache.py`
- **MatchPrefixParams** (299 connections) — `python/sglang/srt/mem_cache/base_prefix_cache.py`
- **MatchResult** (255 connections) — `python/sglang/srt/mem_cache/base_prefix_cache.py`
- **CacheInitParams** (247 connections) — `python/sglang/srt/mem_cache/cache_init_params.py`
- **IncLockRefResult** (238 connections) — `python/sglang/srt/mem_cache/base_prefix_cache.py`
- **DecLockRefParams** (238 connections) — `python/sglang/srt/mem_cache/base_prefix_cache.py`
- **RadixKey** (212 connections) — `python/sglang/srt/mem_cache/radix_cache.py`
- **InsertResult** (199 connections) — `python/sglang/srt/mem_cache/base_prefix_cache.py`
- **InsertParams** (191 connections) — `python/sglang/srt/mem_cache/base_prefix_cache.py`
- **EvictResult** (182 connections) — `python/sglang/srt/mem_cache/base_prefix_cache.py`
- **DecLockRefResult** (165 connections) — `python/sglang/srt/mem_cache/base_prefix_cache.py`
- **HybridReqToTokenPool** (126 connections) — `python/sglang/srt/mem_cache/memory_pool.py`
- **InitLoadBackParams** (125 connections) — `python/sglang/srt/mem_cache/base_prefix_cache.py`
- **HybridCacheController** (102 connections) — `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`
- **RadixCache** (93 connections) — `python/sglang/srt/mem_cache/radix_cache.py`
- **KVCacheEventMixin** (90 connections) — `python/sglang/srt/mem_cache/events.py`
- **TreeNode** (90 connections) — `python/sglang/srt/mem_cache/radix_cache.py`
- **StorageMetricsCollector** (77 connections) — `python/sglang/srt/observability/metrics_collector.py`
- **StorageMedium** (70 connections) — `python/sglang/srt/disaggregation/kv_events.py`
- **PrefetchOperation** (65 connections) — `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`
- **LRUList** (51 connections) — `python/sglang/srt/mem_cache/mamba_radix_cache.py`
- **PrefetchTimeoutConfig** (39 connections) — `python/sglang/srt/mem_cache/hicache_storage.py`
- **UnifiedLRUList** (39 connections) — `python/sglang/srt/mem_cache/unified_radix_cache.py`
- *... and 448 more nodes in this community*

## Relationships

- [[Aibrix KV Cache Storage]] (809 shared connections)
- [[Disaggregation Utils & Cache Tests]] (258 shared connections)
- [[Disaggregation Bootstrap & Decode]] (192 shared connections)
- [[Community 44]] (151 shared connections)
- [[Grammar Manager & HiCache Clear]] (137 shared connections)
- [[CLI Arg Parsing & Deprecation]] (122 shared connections)
- [[Community 52]] (117 shared connections)
- [[Community 61]] (107 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (105 shared connections)
- [[Community 70]] (104 shared connections)
- [[Community 75]] (71 shared connections)
- [[Community 81]] (64 shared connections)

## Source Files

- `python/sglang/srt/disaggregation/decode_hicache_mixin.py`
- `python/sglang/srt/disaggregation/kv_events.py`
- `python/sglang/srt/kv_canary/radix_cache_walker.py`
- `python/sglang/srt/managers/cache_controller.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/mem_cache/base_prefix_cache.py`
- `python/sglang/srt/mem_cache/cache_init_params.py`
- `python/sglang/srt/mem_cache/chunk_cache.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/mem_cache/cpp_radix_tree/radix_tree.py`
- `python/sglang/srt/mem_cache/events.py`
- `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py`
- `python/sglang/srt/mem_cache/hicache_storage.py`
- `python/sglang/srt/mem_cache/hiradix_cache.py`
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`
- `python/sglang/srt/mem_cache/mamba_radix_cache.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/mem_cache/radix_cache_cpp.py`

## Audit Trail

- EXTRACTED: 1408 (16%)
- INFERRED: 7317 (84%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*