# Community 312

> 21 nodes

## Key Concepts

- **Tensor** (23 connections) — `python/sglang/srt/managers/mm_utils.py`
- **DataEmbeddingFunc** (14 connections) — `python/sglang/srt/managers/mm_utils.py`
- **_get_chunked_embedding_full()** (11 connections) — `python/sglang/srt/managers/mm_utils.py`
- **device** (10 connections) — `python/sglang/srt/managers/mm_utils.py`
- **get_embedding_and_mask()** (10 connections) — `python/sglang/srt/managers/mm_utils.py`
- **_get_chunked_embedding_by_item()** (9 connections) — `python/sglang/srt/managers/mm_utils.py`
- **_get_chunked_prefill_embedding()** (8 connections) — `python/sglang/srt/managers/mm_utils.py`
- **_get_precomputed_embedding()** (6 connections) — `python/sglang/srt/managers/mm_utils.py`
- **_move_items_to_device()** (6 connections) — `python/sglang/srt/managers/mm_utils.py`
- **get_embedding_chunk()** (5 connections) — `python/sglang/srt/managers/mm_utils.py`
- **_adjust_embedding_length()** (5 connections) — `python/sglang/srt/managers/mm_utils.py`
- **_can_skip_pre_embed_feature_move()** (4 connections) — `python/sglang/srt/managers/mm_utils.py`
- **_get_multimodal_mask()** (3 connections) — `python/sglang/srt/managers/mm_utils.py`
- **Extract a chunk of embeddings based on the specified prefix length, sequence len** (1 connections) — `python/sglang/srt/managers/mm_utils.py`
- **If all items have precomputed_embeddings, return their concatenation.     If som** (1 connections) — `python/sglang/srt/managers/mm_utils.py`
- **qwen-vl visual forward already moves batched features to the target device.** (1 connections) — `python/sglang/srt/managers/mm_utils.py`
- **Move item features to the target device (in-place, non-blocking).** (1 connections) — `python/sglang/srt/managers/mm_utils.py`
- **Fallback: encode all items at once, cache combined result, extract chunk.     Us** (1 connections) — `python/sglang/srt/managers/mm_utils.py`
- **Per-image chunk-aware encoding: only encode images overlapping with the     curr** (1 connections) — `python/sglang/srt/managers/mm_utils.py`
- **Chunked prefill embedding: encode per-request items and extract the chunk.     I** (1 connections) — `python/sglang/srt/managers/mm_utils.py`
- **Generate multimodal embeddings and create a mask for identifying their positions** (1 connections) — `python/sglang/srt/managers/mm_utils.py`

## Relationships

- [[Vision-Language Model Configs]] (26 shared connections)
- [[Community 318]] (12 shared connections)
- [[Model Config & Encode Server]] (9 shared connections)
- [[Community 369]] (2 shared connections)
- [[Community 479]] (1 shared connections)
- [[Community 47]] (1 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)

## Source Files

- `python/sglang/srt/managers/mm_utils.py`

## Audit Trail

- EXTRACTED: 100 (82%)
- INFERRED: 22 (18%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*