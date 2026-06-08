# Community 363

> 17 nodes

## Key Concepts

- **.forward_absorb_prepare()** (18 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **DeepseekMLAForwardMixin** (10 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **.forward_absorb_core()** (10 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **DeepseekV2AttentionMLA** (8 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **._fuse_rope_for_trtllm_mla()** (7 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **._skip_rope_for_dsa_tilelang_fused()** (6 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **forward_mla.py** (5 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **Tensor** (4 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **bmm_fp8()** (4 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **ForwardBatch** (4 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **._skip_rope_for_aiter_fused_mla()** (4 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **_bmm_fp8_op()** (3 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **.init_mla_forward()** (3 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **fused_qk_rmsnorm_bf16()** (2 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **Check if we should skip rope and do fused rope+quantize for TRTLLM MLA decode in** (1 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **Check if we should skip rope and use fused rope+cache path for TileLang DSA on g** (1 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- **Skip rope in prepare and let the fused kernel in forward_absorb_core handle it,** (1 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`

## Relationships

- [[Vision-Language Model Configs]] (5 shared connections)
- [[DeepSeek MLA Attention & MoE]] (5 shared connections)
- [[Context-Parallel Attention]] (4 shared connections)
- [[Community 333]] (4 shared connections)
- [[Community 49]] (3 shared connections)
- [[Community 914]] (2 shared connections)
- [[Community 517]] (2 shared connections)
- [[Community 37]] (1 shared connections)
- [[Qwen3 / Kimi Model Configs]] (1 shared connections)

## Source Files

- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`

## Audit Trail

- EXTRACTED: 68 (75%)
- INFERRED: 23 (25%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*