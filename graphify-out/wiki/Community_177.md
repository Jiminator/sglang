# Community 177

> 38 nodes

## Key Concepts

- **Tensor** (16 connections) — `python/sglang/srt/models/gpt_j.py`
- **GPTJConfig** (14 connections) — `python/sglang/srt/models/gpt_j.py`
- **QuantizationConfig** (14 connections) — `python/sglang/srt/models/gpt_j.py`
- **GPTJModel** (14 connections) — `python/sglang/srt/models/gpt_j.py`
- **GPTJAttention** (13 connections) — `python/sglang/srt/models/gpt_j.py`
- **ForwardBatch** (13 connections) — `python/sglang/srt/models/gpt_j.py`
- **GPTJMLP** (13 connections) — `python/sglang/srt/models/gpt_j.py`
- **GPTJBlock** (13 connections) — `python/sglang/srt/models/gpt_j.py`
- **GPTJForCausalLM** (13 connections) — `python/sglang/srt/models/gpt_j.py`
- **.do_load_weights()** (11 connections) — `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- **.__init__()** (9 connections) — `python/sglang/srt/models/gpt_j.py`
- **._maybe_quant_weights_to_fp8_ue8m0()** (7 connections) — `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- **.__init__()** (7 connections) — `python/sglang/srt/models/gpt_j.py`
- **.__init__()** (7 connections) — `python/sglang/srt/models/gpt_j.py`
- **should_async_load()** (6 connections) — `python/sglang/srt/model_loader/utils.py`
- **._initialize_nextn_conf()** (6 connections) — `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- **.__init__()** (6 connections) — `python/sglang/srt/models/gpt_j.py`
- **.__init__()** (6 connections) — `python/sglang/srt/models/gpt_j.py`
- **deepseek_weight_loader.py** (5 connections) — `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- **Tensor** (5 connections) — `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- **NextNEnabledConfig** (5 connections) — `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- **NextNDisabledConfig** (5 connections) — `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- **NextNConfig** (5 connections) — `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- **gpt_j.py** (5 connections) — `python/sglang/srt/models/gpt_j.py`
- **.forward()** (4 connections) — `python/sglang/srt/models/gpt_j.py`
- *... and 13 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (55 shared connections)
- [[Vision-Language Model Configs]] (31 shared connections)
- [[Context-Parallel Attention]] (9 shared connections)
- [[Model Configs & Pooler]] (9 shared connections)
- [[Breakable CUDA Graph (TBO)]] (4 shared connections)
- [[Model Config & Encode Server]] (2 shared connections)
- [[Qwen3 / Kimi Model Configs]] (2 shared connections)
- [[Community 45]] (2 shared connections)
- [[Community 459]] (1 shared connections)
- [[Weight Loading & EPLB]] (1 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (1 shared connections)
- [[Activation Functions & Gemma]] (1 shared connections)

## Source Files

- `python/sglang/srt/model_loader/utils.py`
- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- `python/sglang/srt/models/gpt_j.py`

## Audit Trail

- EXTRACTED: 139 (56%)
- INFERRED: 111 (44%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*