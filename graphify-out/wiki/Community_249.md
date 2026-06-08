# Community 249

> 25 nodes

## Key Concepts

- **elastic_ep.py** (10 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **ElasticEPState** (10 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **try_recover_ranks()** (8 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **EPBuffer** (8 connections) — `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py`
- **.init()** (6 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **device** (6 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **join_process_groups()** (6 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **._build_state()** (5 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **.healthy_rank_state()** (5 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **ServerArgs** (4 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **Tensor** (4 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **_iter_live_parallel_groups()** (4 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **GroupCoordinator** (4 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **.sync_active_to_cpu()** (3 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **.snapshot_active_to_last()** (3 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **.reset()** (3 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **._select_device()** (3 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **_get_process_group_backend()** (3 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **_maybe_create_message_queue()** (3 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **_refresh_ep_members()** (3 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **.init_mha_chunk_metadata()** (3 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **.instance()** (2 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **_map_global_to_group_local_ranks()** (2 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **_wait_for_peer_state()** (2 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **.is_active_equal_last()** (1 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`

## Relationships

- [[Hybrid Attention Backend]] (8 shared connections)
- [[Community 313]] (5 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (5 shared connections)
- [[Community 210]] (2 shared connections)
- [[Breakable CUDA Graph (TBO)]] (1 shared connections)
- [[Community 86]] (1 shared connections)
- [[Mamba2 / Hybrid Linear Attention]] (1 shared connections)

## Source Files

- `python/sglang/srt/elastic_ep/elastic_ep.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py`

## Audit Trail

- EXTRACTED: 86 (77%)
- INFERRED: 25 (23%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*