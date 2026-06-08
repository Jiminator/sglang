# Community 914

> 8 nodes

## Key Concepts

- **__init__.py** (10 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py`
- **AttnForwardMethod** (4 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py`
- **forward_mha.py** (4 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **forward_methods.py** (3 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py`
- **forward_mla_fused_rope_rocm.py** (3 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`
- **forward_mla_fused_rope_cpu.py** (2 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py`
- **# TODO: Design a finer way to determine the threshold** (1 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- **# NOTE: hidden_states can be a tuple for some quantization paths.** (1 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`

## Relationships

- [[DeepSeek MLA Attention & MoE]] (5 shared connections)
- [[Community 363]] (2 shared connections)
- [[Community 2464]] (2 shared connections)
- [[Community 107]] (1 shared connections)
- [[Community 419]] (1 shared connections)
- [[Community 354]] (1 shared connections)

## Source Files

- `python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`

## Audit Trail

- EXTRACTED: 27 (96%)
- INFERRED: 1 (4%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*