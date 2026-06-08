# Community 879

> 9 nodes

## Key Concepts

- **Tensor** (7 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **.quantize_and_store()** (3 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **.dequantize_prev_kv()** (3 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **.dequantize_prev_kv()** (3 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **.quantize_and_store()** (2 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **.dequantize_prev_kv()** (2 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **Quantize cache_k / cache_v and write into buffers at loc.** (1 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **Dequantize stored FP4 KV (selected token indices already applied).          Retu** (1 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- **Dequantize FP4 KV (indexed tokens) → FP8 E4M3.** (1 connections) — `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`

## Relationships

- [[Community 423]] (2 shared connections)
- [[Community 401]] (2 shared connections)
- [[Community 1654]] (2 shared connections)
- [[Community 9586]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`

## Audit Trail

- EXTRACTED: 21 (91%)
- INFERRED: 2 (9%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*