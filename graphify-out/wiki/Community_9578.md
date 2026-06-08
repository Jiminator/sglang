# Community 9578

> 6 nodes

## Key Concepts

- **ServerArgs** (10 connections) — `python/sglang/srt/mem_cache/common.py`
- **get_alloc_reserve_per_decode()** (7 connections) — `python/sglang/srt/mem_cache/common.py`
- **get_req_to_token_extra_context_len()** (5 connections) — `python/sglang/srt/mem_cache/common.py`
- **get_alloc_len_per_decode()** (4 connections) — `python/sglang/srt/mem_cache/common.py`
- **KV length reserved per request at each decode step.      The 2x is a double-buff** (1 connections) — `python/sglang/srt/mem_cache/common.py`
- **req_to_token row headroom beyond the model context length.      Sized to hold th** (1 connections) — `python/sglang/srt/mem_cache/common.py`

## Relationships

- [[Community 47]] (4 shared connections)
- [[HiCache Controller & Radix Tree]] (4 shared connections)
- [[CLI Arg Parsing & Deprecation]] (3 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)
- [[Disaggregation Bootstrap & Decode]] (1 shared connections)
- [[Disaggregation Utils & Cache Tests]] (1 shared connections)

## Source Files

- `python/sglang/srt/mem_cache/common.py`

## Audit Trail

- EXTRACTED: 17 (61%)
- INFERRED: 11 (39%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*