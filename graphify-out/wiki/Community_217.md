# Community 217

> 31 nodes

## Key Concepts

- **OrderedDict** (11 connections) — `python/sglang/srt/layers/moe/token_dispatcher/base.py`
- **EvictionPolicy** (9 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **LRUEvictionPolicy** (7 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **FIFOEvictionPolicy** (7 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **eviction_policy.py** (5 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **get_eviction_policy()** (4 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **.__init__()** (3 connections) — `python/sglang/srt/multimodal/processors/mimo_v2.py`
- **.__init__()** (2 connections) — `python/sglang/srt/layers/moe/token_dispatcher/base.py`
- **.mark_used()** (2 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **.select_victim()** (2 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **.remove()** (2 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **.__init__()** (2 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **.select_victim()** (2 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **.__init__()** (2 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **.mark_used()** (2 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **.select_victim()** (2 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **.__init__()** (2 connections) — `python/sglang/srt/mem_cache/multimodal_cache.py`
- **.__init__()** (2 connections) — `python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py`
- **.mark_used()** (1 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **.remove()** (1 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **.remove()** (1 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **Abstract base class for LoRA adapter eviction policies.** (1 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **Marks an adapter as used.** (1 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **Selects an adapter to evict from candidates.** (1 connections) — `python/sglang/srt/lora/eviction_policy.py`
- **Removes an adapter from the policy's tracking.** (1 connections) — `python/sglang/srt/lora/eviction_policy.py`
- *... and 6 more nodes in this community*

## Relationships

- [[Batch-Overlap Operations]] (2 shared connections)
- [[Community 83]] (2 shared connections)
- [[Aibrix KV Cache Storage]] (2 shared connections)
- [[Community 71]] (2 shared connections)
- [[Disaggregation Bootstrap & Decode]] (1 shared connections)
- [[Community 230]] (1 shared connections)
- [[Community 39]] (1 shared connections)
- [[Community 161]] (1 shared connections)
- [[Model Config & Encode Server]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/moe/token_dispatcher/base.py`
- `python/sglang/srt/lora/eviction_policy.py`
- `python/sglang/srt/mem_cache/multimodal_cache.py`
- `python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py`
- `python/sglang/srt/multimodal/processors/mimo_v2.py`

## Audit Trail

- EXTRACTED: 66 (81%)
- INFERRED: 15 (19%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*