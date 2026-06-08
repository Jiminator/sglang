# Community 202

> 34 nodes

## Key Concepts

- **LoRARef** (22 connections) — `python/sglang/srt/lora/lora_manager.py`
- **LoRAUpdateOutput** (17 connections) — `python/sglang/srt/lora/lora_manager.py`
- **LoRAConfig** (15 connections) — `python/sglang/srt/lora/lora_manager.py`
- **Tensor** (15 connections) — `python/sglang/srt/lora/lora_manager.py`
- **.init_state()** (10 connections) — `python/sglang/srt/lora/lora_manager.py`
- **.load_lora_adapter()** (9 connections) — `python/sglang/srt/lora/lora_manager.py`
- **.load_lora_adapter_from_tensors()** (8 connections) — `python/sglang/srt/lora/lora_manager.py`
- **.validate_new_adapter()** (6 connections) — `python/sglang/srt/lora/lora_manager.py`
- **.create_lora_update_result()** (5 connections) — `python/sglang/srt/lora/lora_manager.py`
- **.unload_lora_adapter()** (5 connections) — `python/sglang/srt/lora/lora_manager.py`
- **.init_lora_adapters()** (5 connections) — `python/sglang/srt/lora/lora_manager.py`
- **.init_lora_shapes()** (5 connections) — `python/sglang/srt/lora/lora_manager.py`
- **.load_lora_weights_from_tensors()** (5 connections) — `python/sglang/srt/lora/lora_manager.py`
- **.init_memory_pool()** (5 connections) — `python/sglang/srt/lora/lora_manager.py`
- **get_normalized_target_modules()** (5 connections) — `python/sglang/srt/lora/utils.py`
- **.update_lora_info()** (4 connections) — `python/sglang/srt/lora/lora_manager.py`
- **.load_lora_weights()** (4 connections) — `python/sglang/srt/lora/lora_manager.py`
- **.init_lora_modules()** (4 connections) — `python/sglang/srt/lora/lora_manager.py`
- **get_target_module_name()** (4 connections) — `python/sglang/srt/lora/utils.py`
- **auto_detect_lora_target_modules()** (4 connections) — `python/sglang/srt/lora/utils.py`
- **.fetch_new_loras()** (3 connections) — `python/sglang/srt/lora/lora_manager.py`
- **Load a single LoRA adapter from the specified path.          Args:             l** (1 connections) — `python/sglang/srt/lora/lora_manager.py`
- **Validate if an adapter can be loaded into the current LoRA memory pool and gener** (1 connections) — `python/sglang/srt/lora/lora_manager.py`
- **Unload LoRA adapters by their names. This will remove the adapters from the memo** (1 connections) — `python/sglang/srt/lora/lora_manager.py`
- **Update all LoRA modules to associate them with the latest memory buffer.** (1 connections) — `python/sglang/srt/lora/lora_manager.py`
- *... and 9 more nodes in this community*

## Relationships

- [[Hybrid Attention Backend]] (18 shared connections)
- [[Community 116]] (17 shared connections)
- [[Community 80]] (9 shared connections)
- [[DeepSeek MLA Attention & MoE]] (8 shared connections)
- [[Community 161]] (5 shared connections)
- [[Community 111]] (4 shared connections)
- [[Grammar Manager & HiCache Clear]] (4 shared connections)
- [[Vision-Language Model Configs]] (4 shared connections)
- [[CLI Arg Parsing & Deprecation]] (4 shared connections)
- [[Community 389]] (3 shared connections)
- [[Community 47]] (2 shared connections)
- [[Qwen3 / Kimi Model Configs]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/lora_manager.py`
- `python/sglang/srt/lora/utils.py`

## Audit Trail

- EXTRACTED: 110 (64%)
- INFERRED: 63 (36%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*