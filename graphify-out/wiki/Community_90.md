# Community 90

> 70 nodes

## Key Concepts

- **CompressorHip** (41 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **CompressStatePool** (38 connections) — `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
- **KVAndScore** (26 connections) — `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
- **Tensor** (17 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **ForwardBatch** (14 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **AttentionBackend** (14 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **.compress_decode_paged()** (14 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **DeepseekRefRMSNorm** (11 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **KVAndScore** (11 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **.compress_extend_paged()** (11 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **constexpr** (8 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **CompressStatePool** (8 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **Any** (8 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **deepseek_v4_rope.py** (8 connections) — `python/sglang/srt/layers/deepseek_v4_rope.py`
- **.compress_dispatch()** (7 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **Tensor** (7 connections) — `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
- **._get_state_pool()** (6 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **.compress_fused()** (6 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **fused_norm_rope_inplace_triton()** (6 connections) — `python/sglang/srt/layers/deepseek_v4_rope.py`
- **.forward()** (5 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **fused_softmax_pool_triton()** (5 connections) — `python/sglang/srt/layers/deepseek_v4_rope.py`
- **.__init__()** (5 connections) — `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
- **compress_hip.py** (4 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **._get_states()** (4 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- **.overlap_transform()** (4 connections) — `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- *... and 45 more nodes in this community*

## Relationships

- [[Community 115]] (14 shared connections)
- [[Vision-Language Model Configs]] (13 shared connections)
- [[Disaggregation Utils & Cache Tests]] (12 shared connections)
- [[Community 314]] (12 shared connections)
- [[Aiter Attention Backend]] (9 shared connections)
- [[Community 123]] (9 shared connections)
- [[Context-Parallel Attention]] (7 shared connections)
- [[Community 84]] (7 shared connections)
- [[Disaggregation Bootstrap & Decode]] (4 shared connections)
- [[Community 49]] (2 shared connections)
- [[Community 266]] (1 shared connections)
- [[Community 168]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/dsv4/compress_hip.py`
- `python/sglang/srt/layers/deepseek_v4_rope.py`
- `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`

## Audit Trail

- EXTRACTED: 240 (63%)
- INFERRED: 139 (37%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*