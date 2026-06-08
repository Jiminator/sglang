# Community 448

> 13 nodes

## Key Concepts

- **Lfm2VlConfig** (9 connections) — `python/sglang/srt/configs/lfm2_vl.py`
- **.mamba2_cache_params()** (4 connections) — `python/sglang/srt/configs/lfm2_vl.py`
- **Mamba2CacheParams** (3 connections) — `python/sglang/srt/configs/lfm2_vl.py`
- **.full_attention_layer_ids()** (2 connections) — `python/sglang/srt/configs/lfm2_vl.py`
- **.linear_layer_ids()** (2 connections) — `python/sglang/srt/configs/lfm2_vl.py`
- **.mamba_chunk_size()** (2 connections) — `python/sglang/srt/configs/lfm2_vl.py`
- **lfm2_vl.py** (1 connections) — `python/sglang/srt/configs/lfm2_vl.py`
- **HFLfm2VlConfig** (1 connections)
- **SGLang configuration for LFM2-VL models.      Extends HuggingFace's Lfm2VlConfig** (1 connections) — `python/sglang/srt/configs/lfm2_vl.py`
- **Return indices of attention layers for KV cache (from text_config).** (1 connections) — `python/sglang/srt/configs/lfm2_vl.py`
- **Return indices of conv layers for conv state cache (from text_config).** (1 connections) — `python/sglang/srt/configs/lfm2_vl.py`
- **Return chunk size for Mamba2 backend. LFM2 doesn't use chunking, return 1.** (1 connections) — `python/sglang/srt/configs/lfm2_vl.py`
- **Get cache params for HybridReqToTokenPool initialization.          LFM2 uses Sho** (1 connections) — `python/sglang/srt/configs/lfm2_vl.py`

## Relationships

- [[Disaggregation Bootstrap & Decode]] (2 shared connections)
- [[Community 342]] (2 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)

## Source Files

- `python/sglang/srt/configs/lfm2_vl.py`

## Audit Trail

- EXTRACTED: 24 (83%)
- INFERRED: 5 (17%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*