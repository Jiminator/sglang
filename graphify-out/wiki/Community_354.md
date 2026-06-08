# Community 354

> 18 nodes

## Key Concepts

- **ForwardBatch** (15 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **DeepseekV2AttentionMLA** (15 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **.forward_normal_prepare()** (14 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **Tensor** (13 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **._get_mla_kv_buffer()** (8 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **._chunked_prefix_attn_mha()** (7 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **._concat_and_cast_mha_k()** (7 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **BumpAllocator** (6 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **.forward_normal_chunked_kv_prepare()** (6 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **.forward_normal_one_shot_prepare()** (6 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **._set_mla_kv_buffer()** (6 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **._get_mla_kv_buffer_from_fp8_for_dsa()** (6 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **.forward_normal_core()** (5 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **.forward_normal_chunked_kv_core()** (5 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **.forward_normal_one_shot_core()** (5 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **_resolve_attn_backend()** (4 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **dtype** (4 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **Dequantize FP8 KV cache to BF16 for MLA attention (DSA-specific format).** (1 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`

## Relationships

- [[DeepSeek MLA Attention & MoE]] (17 shared connections)
- [[Hybrid Attention Backend]] (5 shared connections)
- [[Vision-Language Model Configs]] (5 shared connections)
- [[Community 49]] (4 shared connections)
- [[Context-Parallel Attention]] (2 shared connections)
- [[Community 914]] (1 shared connections)
- [[Community 878]] (1 shared connections)

## Source Files

- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`

## Audit Trail

- EXTRACTED: 111 (83%)
- INFERRED: 22 (17%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*