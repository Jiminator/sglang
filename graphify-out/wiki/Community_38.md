# Community 38

> 155 nodes

## Key Concepts

- **DeepSeekV4HiSparseTokenToKVPoolAllocator** (70 connections) — `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- **HiSparseTokenToKVPoolAllocator** (35 connections) — `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- **Tensor** (30 connections) — `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- **SWATokenToKVPoolAllocator** (29 connections) — `python/sglang/srt/mem_cache/allocator/swa.py`
- **HiSparseDSATokenToKVPool** (29 connections) — `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- **PagedTokenToKVPoolAllocator** (20 connections) — `python/sglang/srt/mem_cache/allocator/paged.py`
- **NPUPagedTokenToKVPoolAllocator** (15 connections) — `python/sglang/srt/hardware_backend/npu/allocator_npu.py`
- **Req** (15 connections) — `python/sglang/srt/managers/hisparse_coordinator.py`
- **Tensor** (12 connections) — `python/sglang/srt/managers/hisparse_coordinator.py`
- **get_num_new_pages()** (12 connections) — `python/sglang/srt/utils/common.py`
- **HiSparseAct** (10 connections) — `python/sglang/srt/managers/hisparse_coordinator.py`
- **HiSparseTokenStats** (10 connections) — `python/sglang/srt/managers/hisparse_coordinator.py`
- **Tensor** (10 connections) — `python/sglang/srt/mem_cache/allocator/swa.py`
- **.alloc_extend()** (9 connections) — `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- **ReqToTokenPool** (8 connections) — `python/sglang/srt/managers/hisparse_coordinator.py`
- **HiSparseTokenToKVPoolAllocator** (8 connections) — `python/sglang/srt/managers/hisparse_coordinator.py`
- **DeepSeekV4HiSparseTokenToKVPoolAllocator** (8 connections) — `python/sglang/srt/managers/hisparse_coordinator.py`
- **.available_size()** (8 connections) — `python/sglang/srt/mem_cache/allocator/swa.py`
- **.available_size()** (8 connections) — `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- **transfer_kv_all_layer_mla()** (7 connections) — `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- **RadixAttention** (7 connections) — `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- **.free()** (7 connections) — `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- **.__init__()** (6 connections) — `python/sglang/srt/managers/hisparse_coordinator.py`
- **.admit_request_direct()** (6 connections) — `python/sglang/srt/managers/hisparse_coordinator.py`
- **.alloc_extend()** (6 connections) — `python/sglang/srt/mem_cache/allocator/swa.py`
- *... and 130 more nodes in this community*

## Relationships

- [[Disaggregation Utils & Cache Tests]] (60 shared connections)
- [[HiCache Controller & Radix Tree]] (32 shared connections)
- [[Grammar Manager & HiCache Clear]] (24 shared connections)
- [[Disaggregation Bootstrap & Decode]] (10 shared connections)
- [[DeepSeek MLA Attention & MoE]] (5 shared connections)
- [[Community 84]] (5 shared connections)
- [[Hybrid Attention Backend]] (4 shared connections)
- [[Community 47]] (3 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (2 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (2 shared connections)
- [[Community 45]] (1 shared connections)
- [[Community 107]] (1 shared connections)

## Source Files

- `python/sglang/srt/disaggregation/decode.py`
- `python/sglang/srt/hardware_backend/npu/allocator_npu.py`
- `python/sglang/srt/managers/hisparse_coordinator.py`
- `python/sglang/srt/mem_cache/allocator/paged.py`
- `python/sglang/srt/mem_cache/allocator/swa.py`
- `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
- `python/sglang/srt/mem_cache/memory_pool_host.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 523 (73%)
- INFERRED: 197 (27%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*