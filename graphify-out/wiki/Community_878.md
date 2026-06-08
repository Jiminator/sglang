# Community 878

> 9 nodes

## Key Concepts

- **cache_ops.py** (4 connections) — `python/sglang/srt/layers/attention/triton_ops/cache_ops.py`
- **concat_and_cast_mha_k_triton()** (4 connections) — `python/sglang/srt/layers/attention/triton_ops/cache_ops.py`
- **reshape_and_cache_flash()** (3 connections) — `python/sglang/srt/layers/attention/triton_ops/cache_ops.py`
- **concat_and_cast_mha_k_kernel()** (2 connections) — `python/sglang/srt/layers/attention/triton_ops/cache_ops.py`
- **constexpr** (2 connections) — `python/sglang/srt/layers/attention/triton_ops/cache_ops.py`
- **launch_reshape_and_cache_flash()** (2 connections) — `python/sglang/srt/layers/attention/triton_ops/cache_ops.py`
- **Tensor** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/cache_ops.py`
- **Triton kernel for reshaping per-token K/V tensors into paged KV cache layout.** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/cache_ops.py`
- **Launch wrapper for reshape_and_cache_flash Triton kernel.      This wrapper prep** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/cache_ops.py`

## Relationships

- [[Community 354]] (1 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/triton_ops/cache_ops.py`

## Audit Trail

- EXTRACTED: 18 (90%)
- INFERRED: 2 (10%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*