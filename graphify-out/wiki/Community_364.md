# Community 364

> 17 nodes

## Key Concepts

- **CommonKVManager** (21 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **.try_ensure_parallel_info()** (5 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **._handle_node_failure()** (5 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **._resolve_rank_mapping()** (4 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **.check_status()** (3 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **._mla_slice_ptrs_for_pp()** (3 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **.get_mla_kv_ptrs_with_pp()** (2 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **._start_heartbeat_checker_thread()** (2 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **._on_heartbeat_success()** (2 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **._connect()** (1 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **.get_mha_kv_ptrs_with_pp()** (1 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **Single non-blocking attempt to fetch and cache prefill parallel info.         Re** (1 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **Compute TP/CP/PP rank mapping and store on the PrefillServerInfo object.** (1 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **Produce aligned (src, dst) pointer lists for compressed-MLA         pools (e.g.** (1 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **Start the heartbeat checker thread for Decode worker.** (1 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **Hook called on successful heartbeat. Override for backend-specific cleanup.** (1 connections) — `python/sglang/srt/disaggregation/common/conn.py`
- **Handle failure of a prefill node.** (1 connections) — `python/sglang/srt/disaggregation/common/conn.py`

## Relationships

- [[Community 356]] (6 shared connections)
- [[CLI Arg Parsing & Deprecation]] (4 shared connections)
- [[Disaggregation Bootstrap & Decode]] (3 shared connections)
- [[Community 93]] (1 shared connections)
- [[Breakable CUDA Graph (TBO)]] (1 shared connections)
- [[Community 357]] (1 shared connections)
- [[Community 47]] (1 shared connections)

## Source Files

- `python/sglang/srt/disaggregation/common/conn.py`

## Audit Trail

- EXTRACTED: 52 (95%)
- INFERRED: 3 (5%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*