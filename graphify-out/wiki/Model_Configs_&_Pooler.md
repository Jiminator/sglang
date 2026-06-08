# Model Configs & Pooler

> 763 nodes

## Key Concepts

- **RowParallelLinear** (1569 connections) — `python/sglang/srt/layers/linear.py`
- **LogitsProcessorOutput** (580 connections) — `python/sglang/srt/layers/logits_processor.py`
- **Pooler** (470 connections) — `python/sglang/srt/layers/pooler.py`
- **PoolingType** (433 connections) — `python/sglang/srt/layers/pooler.py`
- **EmbeddingPoolerOutput** (165 connections) — `python/sglang/srt/layers/pooler.py`
- **AttentionType** (163 connections) — `python/sglang/srt/layers/radix_attention.py`
- **Qwen2Model** (101 connections) — `python/sglang/srt/models/qwen2.py`
- **LlamaModel** (57 connections) — `python/sglang/srt/models/llama.py`
- **TransformersBase** (56 connections) — `python/sglang/srt/models/transformers.py`
- **LlamaAttention** (49 connections) — `python/sglang/srt/models/llama.py`
- **Qwen2MLP** (45 connections) — `python/sglang/srt/models/qwen2.py`
- **MultiModalMixin** (41 connections) — `python/sglang/srt/models/transformers.py`
- **Tensor** (40 connections) — `python/sglang/srt/models/transformers.py`
- **CrossEncodingPooler** (33 connections) — `python/sglang/srt/layers/pooler.py`
- **MoEMixin** (33 connections) — `python/sglang/srt/models/transformers.py`
- **ApertusForCausalLM** (32 connections) — `python/sglang/srt/models/apertus.py`
- **Exaone4ForCausalLM** (31 connections) — `python/sglang/srt/models/exaone4.py`
- **Tensor** (31 connections) — `python/sglang/srt/models/jet_nemotron.py`
- **transformers.py** (31 connections) — `python/sglang/srt/models/transformers.py`
- **TransformersFusedMoE** (31 connections) — `python/sglang/srt/models/transformers.py`
- **EmbeddingMixin** (31 connections) — `python/sglang/srt/models/transformers.py`
- **Module** (30 connections) — `python/sglang/srt/models/transformers.py`
- **OPTForCausalLM** (29 connections) — `python/sglang/srt/models/opt.py`
- **DynamicShortConvolution** (28 connections) — `python/sglang/srt/models/jet_nemotron.py`
- **QuantizationConfig** (27 connections) — `python/sglang/srt/models/jet_nemotron.py`
- *... and 738 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (2120 shared connections)
- [[Vision-Language Model Configs]] (1359 shared connections)
- [[Context-Parallel Attention]] (244 shared connections)
- [[Llama / GPT-OSS Model Layers]] (143 shared connections)
- [[Qwen3 / Kimi Model Configs]] (111 shared connections)
- [[Mamba2 / Hybrid Linear Attention]] (92 shared connections)
- [[Activation Functions & Gemma]] (80 shared connections)
- [[Community 59]] (80 shared connections)
- [[Hybrid Attention Backend]] (79 shared connections)
- [[Community 46]] (58 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (53 shared connections)
- [[Aiter Attention Backend]] (38 shared connections)

## Source Files

- `python/sglang/srt/configs/jet_nemotron.py`
- `python/sglang/srt/debug_utils/dumper.py`
- `python/sglang/srt/dllm/algorithm/joint_threshold.py`
- `python/sglang/srt/dllm/algorithm/low_confidence.py`
- `python/sglang/srt/layers/activation.py`
- `python/sglang/srt/layers/linear.py`
- `python/sglang/srt/layers/logits_processor.py`
- `python/sglang/srt/layers/pooler.py`
- `python/sglang/srt/layers/radix_attention.py`
- `python/sglang/srt/models/apertus.py`
- `python/sglang/srt/models/arcee.py`
- `python/sglang/srt/models/bert.py`
- `python/sglang/srt/models/clip.py`
- `python/sglang/srt/models/dflash.py`
- `python/sglang/srt/models/exaone4.py`
- `python/sglang/srt/models/granite.py`
- `python/sglang/srt/models/granitemoe.py`
- `python/sglang/srt/models/internlm2.py`
- `python/sglang/srt/models/internlm2_reward.py`
- `python/sglang/srt/models/jet_nemotron.py`

## Audit Trail

- EXTRACTED: 2752 (27%)
- INFERRED: 7468 (73%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*