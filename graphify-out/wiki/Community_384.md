# Community 384

> 16 nodes

## Key Concepts

- **.check_offload_progress()** (8 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **.finalize_release_on_finish()** (6 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **.offload_kv_cache()** (5 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **._release_finished_req()** (5 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **._trigger_backup()** (4 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **._has_inflight_offload()** (3 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **._check_backup_progress()** (3 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **._compute_prefix_hash()** (3 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **._mark_offload_started()** (2 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **._mark_offload_finished()** (2 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **Offload incremental KV cache for decode side.** (1 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **Check the progress of offload from device to host and backup from host to storag** (1 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **Check the progress of offload from device to host.** (1 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **Check the progress of backup from host to storage.** (1 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **Trigger async backup from host to storage.** (1 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- **Free any remaining tail KV that was not offloaded due to non-aligned length.** (1 connections) — `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`

## Relationships

- [[Grammar Manager & HiCache Clear]] (10 shared connections)
- [[Disaggregation Utils & Cache Tests]] (4 shared connections)
- [[Community 48]] (1 shared connections)

## Source Files

- `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`

## Audit Trail

- EXTRACTED: 44 (94%)
- INFERRED: 3 (6%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*