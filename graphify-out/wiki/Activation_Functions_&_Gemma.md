# Activation Functions & Gemma

> 200 nodes

## Key Concepts

- **GeluAndMul** (111 connections) — `python/sglang/srt/layers/activation.py`
- **Gemma3RMSNorm** (42 connections) — `python/sglang/srt/layers/layernorm.py`
- **Gemma3ForCausalLM** (39 connections) — `python/sglang/srt/models/gemma3_causal.py`
- **Tensor** (28 connections) — `python/sglang/srt/layers/activation.py`
- **Gemma2ForCausalLM** (23 connections) — `python/sglang/srt/models/gemma2.py`
- **Tensor** (23 connections) — `python/sglang/srt/models/gemma3_causal.py`
- **Tensor** (23 connections) — `python/sglang/srt/models/grok.py`
- **get_act_fn()** (22 connections) — `python/sglang/srt/layers/activation.py`
- **Gemma2Model** (22 connections) — `python/sglang/srt/models/gemma2.py`
- **ScalingRotaryEmbedding** (21 connections) — `python/sglang/srt/models/grok.py`
- **Grok1ForCausalLM** (21 connections) — `python/sglang/srt/models/grok.py`
- **QuantizationConfig** (20 connections) — `python/sglang/srt/models/grok.py`
- **PretrainedConfig** (20 connections) — `python/sglang/srt/models/grok.py`
- **ForwardBatch** (19 connections) — `python/sglang/srt/models/gemma3_causal.py`
- **Grok1DecoderLayer** (19 connections) — `python/sglang/srt/models/grok.py`
- **Tensor** (18 connections) — `python/sglang/srt/models/gemma2.py`
- **Gemma3Attention** (18 connections) — `python/sglang/srt/models/gemma3_causal.py`
- **Gemma3RotaryEmbedding** (18 connections) — `python/sglang/srt/models/gemma3_causal.py`
- **Grok1MLP** (18 connections) — `python/sglang/srt/models/grok.py`
- **Grok1MoE** (18 connections) — `python/sglang/srt/models/grok.py`
- **Grok1Attention** (18 connections) — `python/sglang/srt/models/grok.py`
- **Grok1Model** (18 connections) — `python/sglang/srt/models/grok.py`
- **Tensor** (17 connections) — `python/sglang/srt/models/gemma.py`
- **QuantizationConfig** (17 connections) — `python/sglang/srt/models/gemma3_causal.py`
- **Gemma3TextModel** (17 connections) — `python/sglang/srt/models/gemma3_causal.py`
- *... and 175 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (342 shared connections)
- [[Vision-Language Model Configs]] (146 shared connections)
- [[Model Configs & Pooler]] (80 shared connections)
- [[Context-Parallel Attention]] (44 shared connections)
- [[Qwen3 / Kimi Model Configs]] (16 shared connections)
- [[Community 34]] (12 shared connections)
- [[Compressed-Tensors Quant Linear]] (12 shared connections)
- [[Hybrid Attention Backend]] (11 shared connections)
- [[Community 64]] (10 shared connections)
- [[Community 51]] (7 shared connections)
- [[Community 126]] (5 shared connections)
- [[Community 117]] (5 shared connections)

## Source Files

- `python/sglang/srt/layers/activation.py`
- `python/sglang/srt/layers/elementwise.py`
- `python/sglang/srt/layers/layernorm.py`
- `python/sglang/srt/layers/moe/fused_moe_native.py`
- `python/sglang/srt/models/gemma.py`
- `python/sglang/srt/models/gemma2.py`
- `python/sglang/srt/models/gemma2_reward.py`
- `python/sglang/srt/models/gemma3_causal.py`
- `python/sglang/srt/models/gemma3_mm.py`
- `python/sglang/srt/models/grok.py`

## Audit Trail

- EXTRACTED: 745 (48%)
- INFERRED: 815 (52%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*