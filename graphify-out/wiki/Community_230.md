# Community 230

> 29 nodes

## Key Concepts

- **LoRARegistry** (15 connections) — `python/sglang/srt/lora/lora_registry.py`
- **LoRARef** (10 connections) — `python/sglang/srt/lora/lora_registry.py`
- **._register_adapter()** (6 connections) — `python/sglang/srt/lora/lora_registry.py`
- **.__init__()** (5 connections) — `python/sglang/srt/lora/lora_registry.py`
- **.register()** (4 connections) — `python/sglang/srt/lora/lora_registry.py`
- **.get_all_adapters()** (3 connections) — `python/sglang/srt/lora/lora_registry.py`
- **lora_registry.py** (2 connections) — `python/sglang/srt/lora/lora_registry.py`
- **.deterministic_id()** (2 connections) — `python/sglang/srt/lora/lora_registry.py`
- **.unregister()** (2 connections) — `python/sglang/srt/lora/lora_registry.py`
- **.acquire()** (2 connections) — `python/sglang/srt/lora/lora_registry.py`
- **.release()** (2 connections) — `python/sglang/srt/lora/lora_registry.py`
- **.wait_for_unload()** (2 connections) — `python/sglang/srt/lora/lora_registry.py`
- **.get_unregistered_loras()** (2 connections) — `python/sglang/srt/lora/lora_registry.py`
- **.lru_lora_name()** (2 connections) — `python/sglang/srt/lora/lora_registry.py`
- **.num_registered_loras()** (2 connections) — `python/sglang/srt/lora/lora_registry.py`
- **.__post_init__()** (1 connections) — `python/sglang/srt/lora/lora_registry.py`
- **Reference record for a LoRA model.      This object guarantees a unique ``lora_i** (1 connections) — `python/sglang/srt/lora/lora_registry.py`
- **Stable ``lora_id`` for ``--lora-paths`` adapters.          Each node in a multi-** (1 connections) — `python/sglang/srt/lora/lora_registry.py`
- **The central registry to keep track of available LoRA adapters and ongoing LoRA r** (1 connections) — `python/sglang/srt/lora/lora_registry.py`
- **Register a new LoRARef object in the registry.          Args:             lora_r** (1 connections) — `python/sglang/srt/lora/lora_registry.py`
- **Unregister a LoRARef object from the registry and returns the removed LoRA ID.** (1 connections) — `python/sglang/srt/lora/lora_registry.py`
- **Queries registry for LoRA IDs based on LoRA names and start tracking the usage o** (1 connections) — `python/sglang/srt/lora/lora_registry.py`
- **Decrements the usage counter for a LoRA adapter, indicating that it is no longer** (1 connections) — `python/sglang/srt/lora/lora_registry.py`
- **Waits until the usage counter for a LoRA adapter reaches zero, indicating that i** (1 connections) — `python/sglang/srt/lora/lora_registry.py`
- **Returns all LoRA adapters in lora_name that are not found in self._registry.** (1 connections) — `python/sglang/srt/lora/lora_registry.py`
- *... and 4 more nodes in this community*

## Relationships

- [[Anthropic/OpenAI API Entrypoints]] (4 shared connections)
- [[Community 382]] (1 shared connections)
- [[Community 217]] (1 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/lora_registry.py`

## Audit Trail

- EXTRACTED: 69 (92%)
- INFERRED: 6 (8%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*