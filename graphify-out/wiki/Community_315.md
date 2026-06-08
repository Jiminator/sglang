# Community 315

> 21 nodes

## Key Concepts

- **SparsePrefillChunkCache** (20 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **sparse_prefill_utils.py** (7 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **combine_topk_swa_indices()** (7 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **Tensor** (6 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **combined_topk_width()** (5 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **.combine_c4_layer()** (5 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **build_swa_token_ids()** (4 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **.build()** (4 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **.ensure_c128()** (4 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **.ensure_c4()** (3 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **_combine_topk_swa_indices_kernel()** (2 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **_build_swa_token_ids_kernel()** (1 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **constexpr** (1 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **Per-query sparse-index combiner for the FlashMLA sparse prefill path.  Adapts vl** (1 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **Width of the padded combined_indices last dim that     ``combine_topk_swa_indice** (1 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **Combine topk + SWA indices into a single ``flash_mla_sparse_fwd`` row.      Args** (1 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **Build a flat list of physical SWA-cache token IDs covering each     request's po** (1 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **Chunk-invariant scaffolding for ``_forward_prefill_sparse``.      The fields her** (1 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **Populate c128-side fields from per-query c128 page indices.          ``c128_page** (1 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **Populate c4-side fields from the per-query page table.          ``page_table`` i** (1 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- **Per-layer combine for c4. ``c4_sparse_raw_indices`` is the topk         kernel's** (1 connections) — `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`

## Relationships

- [[Community 110]] (9 shared connections)
- [[Community 115]] (5 shared connections)
- [[Community 48]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`

## Audit Trail

- EXTRACTED: 62 (81%)
- INFERRED: 15 (19%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*