# Community 329

> 20 nodes

## Key Concepts

- **MlxAttentionKVPool** (15 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **array** (5 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **.set_kv()** (4 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **.set_kv_all_layers()** (4 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **.get_kv()** (3 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **.get_kv_all_layers()** (3 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **.all_buffers()** (3 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **.__init__()** (2 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **Dtype** (2 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **attention_kv_pool.py** (2 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **.__init__()** (2 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **Dtype** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **.clear()** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **Flat attention KV pool for the MLX backend.  Each layer buffer has shape ``(pool** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **Pre-allocated attention KV pool indexed by integer slot IDs.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **Scatter K/V into *slots* for one layer.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **Gather K/V from *slots* for one layer.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **Gather K/V from *slots* across all layers.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **Scatter K/V into *slots* across all layers.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`
- **Return all buffer arrays (for ``mx.eval``).** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`

## Relationships

- [[Community 281]] (3 shared connections)
- [[Community 133]] (2 shared connections)
- [[Community 1641]] (1 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_pool.py`

## Audit Trail

- EXTRACTED: 47 (87%)
- INFERRED: 7 (13%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*