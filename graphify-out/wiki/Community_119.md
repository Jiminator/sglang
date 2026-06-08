# Community 119

> 54 nodes

## Key Concepts

- **MooncakeKVManager** (51 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **.__init__()** (16 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **.transfer_worker()** (14 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **.maybe_send_extra()** (9 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **init_staging_buffers()** (6 connections) — `python/sglang/srt/disaggregation/common/staging_handler.py`
- **init_staging_allocator()** (6 connections) — `python/sglang/srt/disaggregation/common/staging_handler.py`
- **._do_staging_transfer()** (6 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **._send_mamba_state_slice()** (6 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **.__init__()** (6 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **init_mooncake_custom_mem_pool()** (6 connections) — `python/sglang/srt/disaggregation/mooncake/utils.py`
- **_get_custom_mem_pool()** (5 connections) — `python/sglang/srt/disaggregation/common/staging_handler.py`
- **prefetch_staging_reqs()** (5 connections) — `python/sglang/srt/disaggregation/common/staging_handler.py`
- **._transfer_data()** (5 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **.send_aux()** (5 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **._send_mamba_state()** (5 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **is_watermark_ready()** (4 connections) — `python/sglang/srt/disaggregation/common/staging_handler.py`
- **._send_chunk_ready()** (4 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **.send_aux_tcp()** (4 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **check_mooncake_custom_mem_pool_enabled()** (4 connections) — `python/sglang/srt/disaggregation/mooncake/utils.py`
- **._init_staging_buffers()** (3 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **._init_staging_allocator()** (3 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **._try_create_staging_strategy()** (3 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **.send_aux_data_to_endpoint()** (3 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **.sync_status_to_decode_endpoint()** (3 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **._run_one_probe_pass()** (3 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- *... and 29 more nodes in this community*

## Relationships

- [[Community 68]] (45 shared connections)
- [[Disaggregation Bootstrap & Decode]] (6 shared connections)
- [[Community 151]] (4 shared connections)
- [[Community 78]] (4 shared connections)
- [[Community 47]] (3 shared connections)
- [[Community 199]] (2 shared connections)
- [[Community 339]] (1 shared connections)
- [[Community 76]] (1 shared connections)
- [[CLI Arg Parsing & Deprecation]] (1 shared connections)
- [[Community 32]] (1 shared connections)
- [[Disaggregation Utils & Cache Tests]] (1 shared connections)
- [[Community 256]] (1 shared connections)

## Source Files

- `python/sglang/srt/disaggregation/common/staging_handler.py`
- `python/sglang/srt/disaggregation/mooncake/conn.py`
- `python/sglang/srt/disaggregation/mooncake/utils.py`
- `python/sglang/srt/disaggregation/nixl/conn.py`
- `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`

## Audit Trail

- EXTRACTED: 182 (80%)
- INFERRED: 46 (20%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*