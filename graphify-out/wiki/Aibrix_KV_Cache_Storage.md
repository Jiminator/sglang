# Aibrix KV Cache Storage

> 449 nodes

## Key Concepts

- **PoolName** (220 connections) — `python/sglang/srt/mem_cache/hicache_storage.py`
- **PoolTransfer** (201 connections) — `python/sglang/srt/mem_cache/hicache_storage.py`
- **UnifiedTreeNode** (145 connections) — `python/sglang/srt/mem_cache/unified_radix_cache.py`
- **PoolHitPolicy** (136 connections) — `python/sglang/srt/mem_cache/hicache_storage.py`
- **ABC** (130 connections)
- **PoolTransferResult** (120 connections) — `python/sglang/srt/mem_cache/hicache_storage.py`
- **HostKVCache** (104 connections) — `python/sglang/srt/mem_cache/memory_pool_host.py`
- **TreeComponent** (96 connections) — `python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
- **HiCacheStorageExtraInfo** (86 connections) — `python/sglang/srt/mem_cache/hicache_storage.py`
- **ComponentType** (83 connections) — `python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
- **HiCacheStorage** (77 connections) — `python/sglang/srt/mem_cache/hicache_storage.py`
- **HiCacheStorageConfig** (76 connections) — `python/sglang/srt/mem_cache/hicache_storage.py`
- **LayerDoneCounter** (70 connections) — `python/sglang/srt/managers/cache_controller.py`
- **EvictLayer** (70 connections) — `python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
- **CacheTransferPhase** (70 connections) — `python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
- **SWAComponent** (47 connections) — `python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- **MambaComponent** (45 connections) — `python/sglang/srt/mem_cache/unified_cache_components/mamba_component.py`
- **HiCacheHF3FS** (43 connections) — `python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py`
- **LRURefreshPhase** (36 connections) — `python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
- **StorageMetrics** (34 connections) — `python/sglang/srt/observability/metrics_collector.py`
- **UnifiedTreeNode** (33 connections) — `python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- **Hf3fsMetadataInterface** (30 connections) — `python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py`
- **FullComponent** (29 connections) — `python/sglang/srt/mem_cache/unified_cache_components/full_component.py`
- **CacheOperation** (28 connections) — `python/sglang/srt/managers/cache_controller.py`
- **Hf3fsLocalMetadataClient** (28 connections) — `python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py`
- *... and 424 more nodes in this community*

## Relationships

- [[HiCache Controller & Radix Tree]] (809 shared connections)
- [[Disaggregation Utils & Cache Tests]] (168 shared connections)
- [[Community 52]] (79 shared connections)
- [[Community 108]] (56 shared connections)
- [[Community 95]] (46 shared connections)
- [[Community 63]] (29 shared connections)
- [[Community 83]] (29 shared connections)
- [[Community 178]] (24 shared connections)
- [[Community 162]] (23 shared connections)
- [[Grammar Manager & HiCache Clear]] (21 shared connections)
- [[Community 218]] (20 shared connections)
- [[Community 70]] (16 shared connections)

## Source Files

- `python/sglang/srt/debug_utils/schedule_simulator/schedulers/base.py`
- `python/sglang/srt/entrypoints/EngineBase.py`
- `python/sglang/srt/entrypoints/openai/serving_base.py`
- `python/sglang/srt/managers/cache_controller.py`
- `python/sglang/srt/mem_cache/hicache_storage.py`
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`
- `python/sglang/srt/mem_cache/memory_pool_host.py`
- `python/sglang/srt/mem_cache/storage/aibrix_kvcache/aibrix_kvcache_storage.py`
- `python/sglang/srt/mem_cache/storage/eic/eic_storage.py`
- `python/sglang/srt/mem_cache/storage/hf3fs/hf3fs_client.py`
- `python/sglang/srt/mem_cache/storage/hf3fs/hf3fs_usrbio_client.py`
- `python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py`
- `python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py`
- `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`
- `python/sglang/srt/mem_cache/storage/simm/hicache_simm.py`
- `python/sglang/srt/mem_cache/unified_cache_components/full_component.py`
- `python/sglang/srt/mem_cache/unified_cache_components/mamba_component.py`
- `python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- `python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`

## Audit Trail

- EXTRACTED: 1668 (35%)
- INFERRED: 3098 (65%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*