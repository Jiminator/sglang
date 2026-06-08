# Community 161

> 41 nodes

## Key Concepts

- **LoRAMemoryPool** (40 connections) — `python/sglang/srt/lora/mem_pool.py`
- **Module** (15 connections) — `python/sglang/srt/lora/mem_pool.py`
- **.__init__()** (10 connections) — `python/sglang/srt/lora/mem_pool.py`
- **EmptySlot** (9 connections) — `python/sglang/srt/lora/mem_pool.py`
- **._get_num_local_experts()** (7 connections) — `python/sglang/srt/lora/mem_pool.py`
- **mem_pool.py** (6 connections) — `python/sglang/srt/lora/mem_pool.py`
- **dtype** (6 connections) — `python/sglang/srt/lora/mem_pool.py`
- **._iter_local_expert_weights()** (6 connections) — `python/sglang/srt/lora/mem_pool.py`
- **.get_lora_A_shape()** (6 connections) — `python/sglang/srt/lora/mem_pool.py`
- **.get_lora_B_shape()** (6 connections) — `python/sglang/srt/lora/mem_pool.py`
- **_get_moe_ep_context()** (5 connections) — `python/sglang/srt/lora/mem_pool.py`
- **_get_moe_tp_context()** (5 connections) — `python/sglang/srt/lora/mem_pool.py`
- **_moe_runner_keeps_global_expert_ids()** (4 connections) — `python/sglang/srt/lora/mem_pool.py`
- **.is_moe_module()** (4 connections) — `python/sglang/srt/lora/mem_pool.py`
- **._get_num_experts()** (4 connections) — `python/sglang/srt/lora/mem_pool.py`
- **._get_standard_shape()** (4 connections) — `python/sglang/srt/lora/mem_pool.py`
- **._column_parallel_lora_b_per_rank_dim()** (4 connections) — `python/sglang/srt/lora/mem_pool.py`
- **.init_buffers()** (4 connections) — `python/sglang/srt/lora/mem_pool.py`
- **get_lm_head_lora_b_shard_size()** (4 connections) — `python/sglang/srt/lora/utils.py`
- **append_cache_key_suffix()** (3 connections) — `python/sglang/srt/lora/mem_pool.py`
- **._global_to_local_expert_id()** (3 connections) — `python/sglang/srt/lora/mem_pool.py`
- **.get_embedding_lora_B_shape()** (3 connections) — `python/sglang/srt/lora/mem_pool.py`
- **._has_moe_module()** (2 connections) — `python/sglang/srt/lora/mem_pool.py`
- **.get_embedding_lora_A_shape()** (2 connections) — `python/sglang/srt/lora/mem_pool.py`
- **.__repr__()** (1 connections) — `python/sglang/srt/lora/mem_pool.py`
- *... and 16 more nodes in this community*

## Relationships

- [[Community 116]] (28 shared connections)
- [[DeepSeek MLA Attention & MoE]] (5 shared connections)
- [[Community 202]] (5 shared connections)
- [[Community 80]] (4 shared connections)
- [[Linear Layer Parameters]] (3 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (2 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)
- [[Community 165]] (1 shared connections)
- [[NCCL Symmetric Memory]] (1 shared connections)
- [[Hybrid Attention Backend]] (1 shared connections)
- [[Community 217]] (1 shared connections)
- [[Community 389]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/mem_pool.py`
- `python/sglang/srt/lora/utils.py`

## Audit Trail

- EXTRACTED: 135 (75%)
- INFERRED: 44 (25%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*