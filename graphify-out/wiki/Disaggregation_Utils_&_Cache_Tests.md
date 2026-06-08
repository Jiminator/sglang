# Disaggregation Utils & Cache Tests

> 531 nodes

## Key Concepts

- **DeepSeekV4TokenToKVPool** (230 connections) — `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- **KVCache** (141 connections) — `python/sglang/srt/mem_cache/memory_pool.py`
- **MLATokenToKVPool** (136 connections) — `python/sglang/srt/mem_cache/memory_pool.py`
- **DSATokenToKVPool** (133 connections) — `python/sglang/srt/mem_cache/memory_pool.py`
- **HybridLinearKVPool** (113 connections) — `python/sglang/srt/mem_cache/memory_pool.py`
- **MLATokenToKVPoolHost** (93 connections) — `python/sglang/srt/mem_cache/memory_pool_host.py`
- **MHATokenToKVPool** (88 connections) — `python/sglang/srt/mem_cache/memory_pool.py`
- **MHATokenToKVPoolHost** (75 connections) — `python/sglang/srt/mem_cache/memory_pool_host.py`
- **NPUMLATokenToKVPool** (64 connections) — `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py`
- **MambaPool** (62 connections) — `python/sglang/srt/mem_cache/memory_pool.py`
- **DeepSeekV4PagedHostPool** (58 connections) — `python/sglang/srt/mem_cache/memory_pool_host.py`
- **SidecarPoolSpec** (51 connections) — `python/sglang/srt/mem_cache/hicache_storage.py`
- **Tensor** (50 connections) — `python/sglang/srt/mem_cache/memory_pool_host.py`
- **MambaPoolHost** (50 connections) — `python/sglang/srt/mem_cache/memory_pool_host.py`
- **BaseSWAKVPool** (48 connections) — `python/sglang/srt/mem_cache/base_swa_memory_pool.py`
- **DeepSeekV4StateHostPool** (47 connections) — `python/sglang/srt/mem_cache/memory_pool_host.py`
- **HostPoolGroup** (46 connections) — `python/sglang/srt/mem_cache/memory_pool_host.py`
- **DSAIndexerPoolHost** (43 connections) — `python/sglang/srt/mem_cache/memory_pool_host.py`
- **LogicalHostPool** (41 connections) — `python/sglang/srt/mem_cache/memory_pool_host.py`
- **PoolEntry** (41 connections) — `python/sglang/srt/mem_cache/memory_pool_host.py`
- **Tensor** (38 connections) — `python/sglang/srt/mem_cache/memory_pool.py`
- **Any** (36 connections) — `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`
- **NPUMHATokenToKVPool** (35 connections) — `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py`
- **ServerArgs** (35 connections) — `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`
- **CacheInitParams** (35 connections) — `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`
- *... and 506 more nodes in this community*

## Relationships

- [[HiCache Controller & Radix Tree]] (258 shared connections)
- [[Aibrix KV Cache Storage]] (168 shared connections)
- [[Disaggregation Bootstrap & Decode]] (146 shared connections)
- [[Community 38]] (60 shared connections)
- [[CLI Arg Parsing & Deprecation]] (54 shared connections)
- [[Community 84]] (49 shared connections)
- [[Aiter Attention Backend]] (36 shared connections)
- [[Context-Parallel Attention]] (33 shared connections)
- [[Community 81]] (32 shared connections)
- [[DeepSeek MLA Attention & MoE]] (30 shared connections)
- [[Hybrid Attention Backend]] (26 shared connections)
- [[Community 115]] (26 shared connections)

## Source Files

- `python/sglang/srt/configs/mamba_utils.py`
- `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- `python/sglang/srt/disaggregation/kv_events.py`
- `python/sglang/srt/disaggregation/utils.py`
- `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py`
- `python/sglang/srt/kv_canary/pool_patcher/api.py`
- `python/sglang/srt/mem_cache/base_swa_memory_pool.py`
- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `python/sglang/srt/mem_cache/hicache_storage.py`
- `python/sglang/srt/mem_cache/hiradix_cache.py`
- `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`
- `python/sglang/srt/mem_cache/kv_cache_builder.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/memory_pool_host.py`
- `python/sglang/srt/mem_cache/storage/aibrix_kvcache/unit_test.py`
- `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`
- `python/sglang/srt/mem_cache/swa_memory_pool.py`
- `python/sglang/srt/mem_cache/utils.py`
- `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py`

## Audit Trail

- EXTRACTED: 1812 (42%)
- INFERRED: 2519 (58%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*