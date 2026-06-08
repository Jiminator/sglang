# Community 409

> 15 nodes

## Key Concepts

- **adjust_config_with_unaligned_cpu_tp()** (12 connections) — `python/sglang/srt/configs/update_config.py`
- **update_config.py** (8 connections) — `python/sglang/srt/configs/update_config.py`
- **pad_vocab_size()** (6 connections) — `python/sglang/srt/layers/vocab_parallel_embedding.py`
- **adjust_tp_num_heads_if_necessary()** (5 connections) — `python/sglang/srt/configs/update_config.py`
- **update_config()** (5 connections) — `python/sglang/srt/configs/update_config.py`
- **update_intermediate_size()** (4 connections) — `python/sglang/srt/configs/update_config.py`
- **log_debug_on_rank0()** (4 connections) — `python/sglang/srt/utils/common.py`
- **may_get_weight_block_size()** (3 connections) — `python/sglang/srt/configs/update_config.py`
- **ModelConfig** (3 connections) — `python/sglang/srt/configs/update_config.py`
- **LoadConfig** (3 connections) — `python/sglang/srt/configs/update_config.py`
- **get_moe_padding_size()** (2 connections) — `python/sglang/srt/configs/update_config.py`
- **get_num_heads_padding_size()** (2 connections) — `python/sglang/srt/configs/update_config.py`
- **resolve_head_dim()** (2 connections) — `python/sglang/srt/configs/update_config.py`
- **Pad the vocab size to the given value.** (1 connections) — `python/sglang/srt/layers/vocab_parallel_embedding.py`
- **Log a debug message only on tensor model parallel rank 0.     Falls back to logg** (1 connections) — `python/sglang/srt/utils/common.py`

## Relationships

- [[Hybrid Attention Backend]] (3 shared connections)
- [[Model Config & Encode Server]] (2 shared connections)
- [[Community 48]] (2 shared connections)
- [[Community 35]] (1 shared connections)
- [[Community 408]] (1 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)
- [[Community 42]] (1 shared connections)

## Source Files

- `python/sglang/srt/configs/update_config.py`
- `python/sglang/srt/layers/vocab_parallel_embedding.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 45 (74%)
- INFERRED: 16 (26%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*