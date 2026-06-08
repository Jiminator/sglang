# Community 192

> 36 nodes

## Key Concepts

- **Llama4ForConditionalGeneration** (38 connections) — `python/sglang/srt/models/mllama4.py`
- **.load_weights()** (9 connections) — `python/sglang/srt/models/mllama4.py`
- **._handle_expert_weights()** (7 connections) — `python/sglang/srt/models/mllama4.py`
- **._handle_expert_scale_params()** (5 connections) — `python/sglang/srt/models/mllama4.py`
- **._handle_expert_weight_params()** (5 connections) — `python/sglang/srt/models/mllama4.py`
- **._has_vision_weights()** (4 connections) — `python/sglang/srt/models/mllama4.py`
- **.permute_qk_weight_for_rotary()** (4 connections) — `python/sglang/srt/models/mllama4.py`
- **._handle_scale_remapping()** (4 connections) — `python/sglang/srt/models/mllama4.py`
- **._handle_stacked_params()** (4 connections) — `python/sglang/srt/models/mllama4.py`
- **._handle_other_expert_params()** (4 connections) — `python/sglang/srt/models/mllama4.py`
- **._transform_expert_name()** (4 connections) — `python/sglang/srt/models/mllama4.py`
- **._handle_default_weight()** (4 connections) — `python/sglang/srt/models/mllama4.py`
- **._check_vision_weights_in_index()** (3 connections) — `python/sglang/srt/models/mllama4.py`
- **._should_skip_weight()** (3 connections) — `python/sglang/srt/models/mllama4.py`
- **._transform_weight_name()** (3 connections) — `python/sglang/srt/models/mllama4.py`
- **.pad_input_ids()** (2 connections) — `python/sglang/srt/models/mllama4.py`
- **.should_apply_lora()** (2 connections) — `python/sglang/srt/models/mllama4.py`
- **.get_embed_and_head()** (2 connections) — `python/sglang/srt/models/mllama4.py`
- **.set_embed_and_head()** (2 connections) — `python/sglang/srt/models/mllama4.py`
- **.get_embed()** (2 connections) — `python/sglang/srt/models/mllama4.py`
- **.set_embed()** (2 connections) — `python/sglang/srt/models/mllama4.py`
- **.set_eagle3_layers_to_capture()** (1 connections) — `python/sglang/srt/models/mllama4.py`
- **.get_hidden_dim()** (1 connections) — `python/sglang/srt/models/mllama4.py`
- **Check if the model has vision components by examining the checkpoint.** (1 connections) — `python/sglang/srt/models/mllama4.py`
- **Check if the model.safetensors.index.json contains vision weights.** (1 connections) — `python/sglang/srt/models/mllama4.py`
- *... and 11 more nodes in this community*

## Relationships

- [[Vision-Language Model Configs]] (23 shared connections)
- [[DeepSeek MLA Attention & MoE]] (2 shared connections)
- [[Model Configs & Pooler]] (1 shared connections)
- [[Community 54]] (1 shared connections)
- [[Qwen3 / Kimi Model Configs]] (1 shared connections)

## Source Files

- `python/sglang/srt/models/mllama4.py`

## Audit Trail

- EXTRACTED: 114 (89%)
- INFERRED: 14 (11%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*