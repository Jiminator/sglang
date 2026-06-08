# Mamba2 / Hybrid Linear Attention

> 199 nodes

## Key Concepts

- **HybridLinearAttnBackend** (92 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **MambaMixer2** (67 connections) — `python/sglang/srt/layers/attention/mamba/mamba.py`
- **Mamba2AttnBackend** (60 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **ReLU2** (42 connections) — `python/sglang/srt/layers/activation.py`
- **GraniteMoeMoE** (34 connections) — `python/sglang/srt/models/granitemoe.py`
- **Tensor** (30 connections) — `python/sglang/srt/models/nemotron_h.py`
- **NemotronHForCausalLM** (30 connections) — `python/sglang/srt/models/nemotron_h.py`
- **GraniteMoeHybridForCausalLM** (27 connections) — `python/sglang/srt/models/granitemoehybrid.py`
- **Tensor** (26 connections) — `python/sglang/srt/models/granitemoehybrid.py`
- **QuantizationConfig** (25 connections) — `python/sglang/srt/models/granitemoehybrid.py`
- **NemotronHConfig** (25 connections) — `python/sglang/srt/models/nemotron_h.py`
- **QuantizationConfig** (25 connections) — `python/sglang/srt/models/nemotron_h.py`
- **GraniteMoeSharedMLP** (24 connections) — `python/sglang/srt/models/granitemoehybrid.py`
- **GraniteMoeHybridConfig** (24 connections) — `python/sglang/srt/models/granitemoehybrid.py`
- **ForwardBatch** (24 connections) — `python/sglang/srt/models/granitemoehybrid.py`
- **GraniteMoeHybridModel** (24 connections) — `python/sglang/srt/models/granitemoehybrid.py`
- **GraniteMoeHybridConfig** (23 connections) — `python/sglang/srt/configs/granitemoehybrid.py`
- **FalconH1ForCausalLM** (23 connections) — `python/sglang/srt/models/falcon_h1.py`
- **GraniteMoeHybridAttention** (23 connections) — `python/sglang/srt/models/granitemoehybrid.py`
- **ForwardBatch** (23 connections) — `python/sglang/srt/models/nemotron_h.py`
- **FalconH1HybridAttentionDecoderLayer** (22 connections) — `python/sglang/srt/models/falcon_h1.py`
- **Tensor** (22 connections) — `python/sglang/srt/models/falcon_h1.py`
- **GraniteMoeHybridMambaDecoderLayer** (22 connections) — `python/sglang/srt/models/granitemoehybrid.py`
- **GraniteMoeHybridAttentionDecoderLayer** (22 connections) — `python/sglang/srt/models/granitemoehybrid.py`
- **NemotronHMoE** (22 connections) — `python/sglang/srt/models/nemotron_h.py`
- *... and 174 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (410 shared connections)
- [[Vision-Language Model Configs]] (118 shared connections)
- [[Model Configs & Pooler]] (92 shared connections)
- [[Context-Parallel Attention]] (36 shared connections)
- [[Community 86]] (32 shared connections)
- [[Aiter Attention Backend]] (19 shared connections)
- [[Community 534]] (9 shared connections)
- [[Qwen3 / Kimi Model Configs]] (5 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (4 shared connections)
- [[Disaggregation Bootstrap & Decode]] (3 shared connections)
- [[Activation Functions & Gemma]] (3 shared connections)
- [[Community 342]] (2 shared connections)

## Source Files

- `python/sglang/srt/configs/falcon_h1.py`
- `python/sglang/srt/configs/granitemoehybrid.py`
- `python/sglang/srt/configs/parakeet.py`
- `python/sglang/srt/layers/activation.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `python/sglang/srt/layers/attention/mamba/mamba.py`
- `python/sglang/srt/layers/radix_attention.py`
- `python/sglang/srt/model_executor/breakable_cuda_graph/context.py`
- `python/sglang/srt/models/falcon_h1.py`
- `python/sglang/srt/models/granitemoe.py`
- `python/sglang/srt/models/granitemoehybrid.py`
- `python/sglang/srt/models/nemotron_h.py`
- `python/sglang/srt/models/nemotron_h_mtp.py`
- `python/sglang/srt/models/parakeet.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 724 (40%)
- INFERRED: 1098 (60%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*