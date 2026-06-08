# Community 151

> 43 nodes

## Key Concepts

- **StagingBuffer** (17 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **staging_buffer.py** (16 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **.send_kvcache_staged()** (10 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **gather_all_layers_to_staging()** (7 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **compute_head_slice_params()** (7 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **_scatter_staging_to_kv_torch()** (6 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **scatter_staging_to_kv()** (6 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **compute_staging_layout()** (6 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **resolve_total_kv_heads()** (6 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **handle_staging_req()** (6 connections) — `python/sglang/srt/disaggregation/common/staging_handler.py`
- **Tensor** (5 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **_gather_all_layers_torch()** (5 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **_scatter_staging_to_kv_triton()** (5 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **gather_kv_head_slices()** (4 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **scatter_kv_head_slices()** (4 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **_gather_all_layers_triton()** (4 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **.create()** (4 connections) — `python/sglang/srt/disaggregation/common/staging_handler.py`
- **_fused_gather_to_staging_kernel()** (2 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **constexpr** (2 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **_fused_scatter_from_staging_kernel()** (2 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **.__init__()** (2 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **._handle_staging_req()** (2 connections) — `python/sglang/srt/disaggregation/mooncake/conn.py`
- **._handle_staging_req()** (2 connections) — `python/sglang/srt/disaggregation/nixl/conn.py`
- **.__init__()** (1 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- **.get_ptr()** (1 connections) — `python/sglang/srt/disaggregation/common/staging_buffer.py`
- *... and 18 more nodes in this community*

## Relationships

- [[Community 68]] (10 shared connections)
- [[Community 78]] (5 shared connections)
- [[Community 119]] (4 shared connections)
- [[Disaggregation Bootstrap & Decode]] (3 shared connections)
- [[Community 47]] (2 shared connections)

## Source Files

- `python/sglang/srt/disaggregation/common/staging_buffer.py`
- `python/sglang/srt/disaggregation/common/staging_handler.py`
- `python/sglang/srt/disaggregation/mooncake/conn.py`
- `python/sglang/srt/disaggregation/nixl/conn.py`

## Audit Trail

- EXTRACTED: 118 (79%)
- INFERRED: 32 (21%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*