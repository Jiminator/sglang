# Community 488

> 11 nodes

## Key Concepts

- **fused_qk_norm_rope_store.py** (6 connections) — `python/sglang/srt/layers/fused_qk_norm_rope_store.py`
- **fused_qk_norm_rope_swa_store()** (6 connections) — `python/sglang/srt/layers/fused_qk_norm_rope_store.py`
- **_batched_rope()** (4 connections) — `python/sglang/srt/layers/fused_qk_norm_rope_store.py`
- **_fused_qk_norm_rope_store_kernel()** (4 connections) — `python/sglang/srt/layers/fused_qk_norm_rope_store.py`
- **_gptj_rotate()** (3 connections) — `python/sglang/srt/layers/fused_qk_norm_rope_store.py`
- **constexpr** (3 connections) — `python/sglang/srt/layers/fused_qk_norm_rope_store.py`
- **_batched_rmsnorm()** (2 connections) — `python/sglang/srt/layers/fused_qk_norm_rope_store.py`
- **Tensor** (1 connections) — `python/sglang/srt/layers/fused_qk_norm_rope_store.py`
- **dtype** (1 connections) — `python/sglang/srt/layers/fused_qk_norm_rope_store.py`
- **Fused Q per-head RMSNorm + KV RMSNorm + RoPE + FP8 nope quant + paged SWA store.** (1 connections) — `python/sglang/srt/layers/fused_qk_norm_rope_store.py`
- **Fused Q norm + KV norm + RoPE + optional FP8 paged SWA store.      Args:** (1 connections) — `python/sglang/srt/layers/fused_qk_norm_rope_store.py`

## Relationships

- [[Context-Parallel Attention]] (2 shared connections)

## Source Files

- `python/sglang/srt/layers/fused_qk_norm_rope_store.py`

## Audit Trail

- EXTRACTED: 30 (94%)
- INFERRED: 2 (6%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*