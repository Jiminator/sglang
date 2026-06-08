# Community 314

> 21 nodes

## Key Concepts

- **Tensor** (16 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **ForwardBatch** (15 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **.forward_compress()** (12 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **Compressor** (11 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **CompressorDecodePlan** (11 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **CompressorPrefillPlan** (10 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **DeepSeekV4TokenToKVPool** (10 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **compressor.py** (7 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **.compute_kv_score()** (7 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **.forward()** (7 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **quant_to_nope_fp8_rope_bf16_pack_triton()** (7 connections) — `python/sglang/srt/layers/attention/dsv4/quant_k_cache.py`
- **create_paged_compressor_data()** (6 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **.forward_core_compressor()** (5 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **.forward_indexer_compressor()** (5 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **make_compressor_plan()** (5 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **is_overlap_compress()** (4 connections) — `python/sglang/srt/layers/attention/dsv4/compressor.py`
- **quant_k_cache.py** (2 connections) — `python/sglang/srt/layers/attention/dsv4/quant_k_cache.py`
- **_quant_k_cache_fused_kernel()** (2 connections) — `python/sglang/srt/layers/attention/dsv4/quant_k_cache.py`
- **constexpr** (1 connections) — `python/sglang/srt/layers/attention/dsv4/quant_k_cache.py`
- **Tensor** (1 connections) — `python/sglang/srt/layers/attention/dsv4/quant_k_cache.py`
- **NopeFp8RopeBf16Pack** (1 connections) — `python/sglang/srt/layers/attention/dsv4/quant_k_cache.py`

## Relationships

- [[Community 90]] (12 shared connections)
- [[DeepSeek MLA Attention & MoE]] (12 shared connections)
- [[Community 115]] (10 shared connections)
- [[Context-Parallel Attention]] (10 shared connections)
- [[Vision-Language Model Configs]] (8 shared connections)
- [[Community 110]] (7 shared connections)
- [[Aiter Attention Backend]] (6 shared connections)
- [[Disaggregation Utils & Cache Tests]] (6 shared connections)
- [[Community 266]] (3 shared connections)
- [[Community 2463]] (1 shared connections)
- [[Community 49]] (1 shared connections)
- [[NCCL Symmetric Memory]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/dsv4/compressor.py`
- `python/sglang/srt/layers/attention/dsv4/quant_k_cache.py`

## Audit Trail

- EXTRACTED: 79 (54%)
- INFERRED: 66 (46%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*