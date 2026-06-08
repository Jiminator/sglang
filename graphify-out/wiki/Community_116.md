# Community 116

> 54 nodes

## Key Concepts

- **BaseLayerWithLoRA** (46 connections) — `python/sglang/srt/lora/layers.py`
- **LoRAAdapter** (41 connections) — `python/sglang/srt/lora/lora.py`
- **LoRAConfig** (35 connections) — `python/sglang/srt/lora/lora_config.py`
- **Module** (14 connections) — `python/sglang/srt/lora/lora_manager.py`
- **AutoConfig** (14 connections) — `python/sglang/srt/lora/lora_manager.py`
- **LoadConfig** (14 connections) — `python/sglang/srt/lora/lora_manager.py`
- **dtype** (14 connections) — `python/sglang/srt/lora/lora_manager.py`
- **ServerArgs** (14 connections) — `python/sglang/srt/lora/lora_manager.py`
- **ForwardBatch** (14 connections) — `python/sglang/srt/lora/lora_manager.py`
- **Tensor** (12 connections) — `python/sglang/srt/lora/lora.py`
- **._normalize_weights()** (9 connections) — `python/sglang/srt/lora/lora.py`
- **.__init__()** (9 connections) — `python/sglang/srt/lora/lora_manager.py`
- **Tensor** (9 connections) — `python/sglang/srt/lora/mem_pool.py`
- **.load_lora_weight_to_buffer()** (9 connections) — `python/sglang/srt/lora/mem_pool.py`
- **LoRAAdapter** (7 connections) — `python/sglang/srt/lora/mem_pool.py`
- **BaseLayerWithLoRA** (7 connections) — `python/sglang/srt/lora/mem_pool.py`
- **LoRAType** (7 connections) — `python/sglang/srt/lora/mem_pool.py`
- **._process_weight()** (6 connections) — `python/sglang/srt/lora/lora.py`
- **AutoConfig** (6 connections) — `python/sglang/srt/lora/mem_pool.py`
- **LoRAConfig** (6 connections) — `python/sglang/srt/lora/mem_pool.py`
- **LoRARef** (6 connections) — `python/sglang/srt/lora/mem_pool.py`
- **.prepare_lora_batch()** (5 connections) — `python/sglang/srt/lora/mem_pool.py`
- **._is_non_gated_moe_weight()** (4 connections) — `python/sglang/srt/lora/lora.py`
- **.initialize_weights()** (4 connections) — `python/sglang/srt/lora/lora.py`
- **.initialize_weights_from_tensors()** (4 connections) — `python/sglang/srt/lora/lora.py`
- *... and 29 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (28 shared connections)
- [[Community 80]] (28 shared connections)
- [[Community 161]] (28 shared connections)
- [[Community 111]] (18 shared connections)
- [[Community 202]] (17 shared connections)
- [[Hybrid Attention Backend]] (16 shared connections)
- [[Vision-Language Model Configs]] (7 shared connections)
- [[Grammar Manager & HiCache Clear]] (6 shared connections)
- [[CLI Arg Parsing & Deprecation]] (6 shared connections)
- [[Qwen3 / Kimi Model Configs]] (2 shared connections)
- [[Model Configs & Pooler]] (1 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/layers.py`
- `python/sglang/srt/lora/lora.py`
- `python/sglang/srt/lora/lora_config.py`
- `python/sglang/srt/lora/lora_manager.py`
- `python/sglang/srt/lora/mem_pool.py`

## Audit Trail

- EXTRACTED: 168 (44%)
- INFERRED: 213 (56%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*