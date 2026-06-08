# Community 401

> 16 nodes

## Key Concepts

- **BlockFP4KVQuantizeUtil** (8 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **NVFP4KVQuantizeUtil** (8 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **kvfp4_tensor.py** (4 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **Tensor** (4 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **.batched_dequantize()** (4 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **.dequantize()** (4 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **.batched_quantize()** (3 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **.quantize()** (3 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **FP4KVCacheRecipe** (2 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **dtype** (2 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **Block-wise FP4 (E2M1) quantization for KV cache.      Similar to MXFP4 but uses** (1 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **Quantize tensor to KVFP4 format         Args:             tensor: Input tensor o** (1 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **Dequantize KVFP4 tensor         Args:             quant_tensor: Quantized tensor** (1 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **Utility class for NVFP4 quantization and dequantization with two-level scaling** (1 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **Quantize BF16/FP16 tensor to NVFP4 format.          Requires SM90+.  Uses ``nvfp** (1 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **Dequantize NVFP4 tensor to BF16/FP16.          Uses ``nvfp4_kv_dequantize`` on S** (1 connections) — `python/sglang/srt/layers/quantization/kvfp4_tensor.py`

## Relationships

- [[Community 107]] (2 shared connections)
- [[Community 423]] (2 shared connections)
- [[Community 879]] (2 shared connections)
- [[Community 1654]] (2 shared connections)
- [[Community 9586]] (2 shared connections)

## Source Files

- `python/sglang/srt/layers/quantization/kvfp4_tensor.py`

## Audit Trail

- EXTRACTED: 40 (83%)
- INFERRED: 8 (17%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*