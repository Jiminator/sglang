# Community 281

> 23 nodes

## Key Concepts

- **PoolBackedAttentionKVCache** (11 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **array** (9 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **.update_and_fetch()** (5 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **.update_and_fetch()** (5 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **._allocate()** (4 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **.get_kv()** (4 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **.to_contiguous()** (4 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **._grow()** (3 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **.write_token()** (3 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **.__init__()** (3 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **MlxAttentionKVPool** (2 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **.keys()** (2 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **.values()** (2 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **.state()** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **.make_mask()** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **Allocate buffers matching the first key tensor's shape.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **Double the buffer until it can hold *required* tokens.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **Append K/V and return all valid K/V up to current offset.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **Write one token. k, v shape: (1, n_kv_heads, 1, head_dim).** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **Return valid K/V: (1, n_kv_heads, offset, head_dim).** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **Lazily gathers cached attention KV from the shared pool during forward.      Eac** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **Gather cached prefix from pool, concatenate with new K/V.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`
- **Convert to contiguous attention KV reusing forward-pass arrays.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`

## Relationships

- [[Community 133]] (6 shared connections)
- [[Community 329]] (3 shared connections)
- [[Community 1641]] (1 shared connections)
- [[Community 87]] (1 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py`

## Audit Trail

- EXTRACTED: 63 (94%)
- INFERRED: 4 (6%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*