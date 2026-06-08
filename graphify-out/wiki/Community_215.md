# Community 215

> 31 nodes

## Key Concepts

- **quant_k_cache.py** (8 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- **mla_buffer.py** (8 connections) — `python/sglang/srt/mem_cache/triton_ops/mla_buffer.py`
- **.set_mla_kv_buffer()** (7 connections) — `python/sglang/srt/mem_cache/memory_pool.py`
- **quantize_k_cache_separate()** (5 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- **set_mla_kv_buffer_triton()** (5 connections) — `python/sglang/srt/mem_cache/triton_ops/mla_buffer.py`
- **set_mla_kv_buffer_triton_fp8_quant()** (5 connections) — `python/sglang/srt/mem_cache/triton_ops/mla_buffer.py`
- **quantize_k_cache()** (4 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- **_quantize_k_cache_fast_wrapped()** (4 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- **constexpr** (4 connections) — `python/sglang/srt/mem_cache/triton_ops/mla_buffer.py`
- **Tensor** (4 connections) — `python/sglang/srt/mem_cache/triton_ops/mla_buffer.py`
- **Tensor** (3 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- **_quantize_k_cache_ref()** (3 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- **_quantize_k_cache_fast()** (3 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- **_quantize_k_cache_fast_separate()** (3 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- **set_mla_kv_buffer_fp8_quant_kernel()** (3 connections) — `python/sglang/srt/mem_cache/triton_ops/mla_buffer.py`
- **set_mla_kv_scale_buffer_triton()** (3 connections) — `python/sglang/srt/mem_cache/triton_ops/mla_buffer.py`
- **get_mla_kv_buffer_triton()** (3 connections) — `python/sglang/srt/mem_cache/triton_ops/mla_buffer.py`
- **_quantize_k_cache_fast_kernel()** (2 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- **set_mla_kv_buffer_kernel()** (2 connections) — `python/sglang/srt/mem_cache/triton_ops/mla_buffer.py`
- **set_mla_kv_scale_buffer_kernel()** (2 connections) — `python/sglang/srt/mem_cache/triton_ops/mla_buffer.py`
- **get_mla_kv_buffer_kernel()** (2 connections) — `python/sglang/srt/mem_cache/triton_ops/mla_buffer.py`
- **constexpr** (1 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- **run_ans()** (1 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- **Quantize k_nope and k_rope separately without concat, returns two tensors.** (1 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- **Quantize the k-cache     Return a tensor with shape (num_blocks, block_size, h_k** (1 connections) — `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- *... and 6 more nodes in this community*

## Relationships

- [[Disaggregation Utils & Cache Tests]] (7 shared connections)
- [[Community 43]] (1 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/dsa/quant_k_cache.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/triton_ops/mla_buffer.py`

## Audit Trail

- EXTRACTED: 81 (87%)
- INFERRED: 12 (13%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*