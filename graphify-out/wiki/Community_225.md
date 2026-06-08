# Community 225

> 29 nodes

## Key Concepts

- **patch_buf_info_method()** (7 connections) — `python/sglang/srt/kv_canary/pool_patcher/buf_info_splice.py`
- **attach_dsv4()** (6 connections) — `python/sglang/srt/kv_canary/pool_patcher/adapters/dsv4.py`
- **attach_mha()** (6 connections) — `python/sglang/srt/kv_canary/pool_patcher/adapters/mha.py`
- **splice_kv_buf_info()** (6 connections) — `python/sglang/srt/kv_canary/pool_patcher/buf_info_splice.py`
- **alloc_canary_buf()** (6 connections) — `python/sglang/srt/kv_canary/pool_patcher/buffer_alloc.py`
- **make_row_source()** (6 connections) — `python/sglang/srt/kv_canary/pool_patcher/buffer_alloc.py`
- **buffer_alloc.py** (5 connections) — `python/sglang/srt/kv_canary/pool_patcher/buffer_alloc.py`
- **buf_info_splice.py** (4 connections) — `python/sglang/srt/kv_canary/pool_patcher/buf_info_splice.py`
- **_clip_read_bytes_aligned()** (4 connections) — `python/sglang/srt/kv_canary/pool_patcher/buffer_alloc.py`
- **make_packed_source()** (4 connections) — `python/sglang/srt/kv_canary/pool_patcher/buffer_alloc.py`
- **device** (3 connections) — `python/sglang/srt/kv_canary/pool_patcher/adapters/dsv4.py`
- **CanaryBufferGroup** (3 connections) — `python/sglang/srt/kv_canary/pool_patcher/adapters/dsv4.py`
- **device** (3 connections) — `python/sglang/srt/kv_canary/pool_patcher/adapters/mha.py`
- **CanaryBufferGroup** (3 connections) — `python/sglang/srt/kv_canary/pool_patcher/adapters/mha.py`
- **CanaryBufferGroup** (3 connections) — `python/sglang/srt/kv_canary/pool_patcher/buf_info_splice.py`
- **BufInfoTriple** (3 connections) — `python/sglang/srt/kv_canary/pool_patcher/buf_info_splice.py`
- **_entry_triple()** (3 connections) — `python/sglang/srt/kv_canary/pool_patcher/buf_info_splice.py`
- **_untranspose_entries()** (3 connections) — `python/sglang/srt/kv_canary/pool_patcher/buf_info_splice.py`
- **resolve_real_kv_read_bytes()** (3 connections) — `python/sglang/srt/kv_canary/pool_patcher/buffer_alloc.py`
- **Tensor** (3 connections) — `python/sglang/srt/kv_canary/pool_patcher/buffer_alloc.py`
- **Tensor** (2 connections) — `python/sglang/srt/kv_canary/pool_patcher/buf_info_splice.py`
- **RealKvSource** (2 connections) — `python/sglang/srt/kv_canary/pool_patcher/buffer_alloc.py`
- **dsv4.py** (1 connections) — `python/sglang/srt/kv_canary/pool_patcher/adapters/dsv4.py`
- **Attach canary buffers to a DeepSeekV4TokenToKVPool.      TODO: only the swa_kv_p** (1 connections) — `python/sglang/srt/kv_canary/pool_patcher/adapters/dsv4.py`
- **mha.py** (1 connections) — `python/sglang/srt/kv_canary/pool_patcher/adapters/mha.py`
- *... and 4 more nodes in this community*

## Relationships

- [[Community 153]] (7 shared connections)
- [[Community 91]] (7 shared connections)
- [[Community 519]] (1 shared connections)
- [[Community 47]] (1 shared connections)
- [[Disaggregation Utils & Cache Tests]] (1 shared connections)

## Source Files

- `python/sglang/srt/kv_canary/pool_patcher/adapters/dsv4.py`
- `python/sglang/srt/kv_canary/pool_patcher/adapters/mha.py`
- `python/sglang/srt/kv_canary/pool_patcher/buf_info_splice.py`
- `python/sglang/srt/kv_canary/pool_patcher/buffer_alloc.py`

## Audit Trail

- EXTRACTED: 68 (72%)
- INFERRED: 27 (28%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*