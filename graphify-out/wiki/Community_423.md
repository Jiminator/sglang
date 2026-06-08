# Community 423

> 15 nodes

## Key Concepts

- **FP4KVCacheQuantMethod** (15 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **fp4_kv_cache_quant_method.py** (5 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **get_fp4_kv_cache_quant_method()** (3 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **.needs_dequant_workspace()** (2 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **.needs_global_scale()** (2 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **.create_buffers()** (2 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **.compute_cell_size()** (2 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **.load_scales_from_model()** (2 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **Abstract base for FP4 KV cache quantization strategies.      Owns the quantize/d** (1 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **Whether the pool should allocate dq_k_buffer / dq_v_buffer for prefill.** (1 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **Whether this method uses a per-layer global FP32 scale.** (1 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **Allocate and return a buffer dict:         {             "k_buffer": list[Tensor** (1 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **Per-token memory footprint in bytes (for capacity estimation).** (1 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **Load per-layer global scales from model weights (no-op by default).** (1 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **Instantiate a FP4KVCacheQuantMethod by recipe name.** (1 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`

## Relationships

- [[Aibrix KV Cache Storage]] (2 shared connections)
- [[Community 9586]] (2 shared connections)
- [[Community 1654]] (2 shared connections)
- [[Community 879]] (2 shared connections)
- [[Community 401]] (2 shared connections)

## Source Files

- `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`

## Audit Trail

- EXTRACTED: 38 (95%)
- INFERRED: 2 (5%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*