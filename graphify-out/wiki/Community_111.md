# Community 111

> 56 nodes

## Key Concepts

- **BaseLoRABackend** (78 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **LoRABackendLmHeadMixing** (13 connections) — `python/sglang/srt/lora/backend/lmhead_mixing.py`
- **Tensor** (9 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **.__init__()** (9 connections) — `python/sglang/srt/lora/lora.py`
- **LoRALayer** (7 connections) — `python/sglang/srt/lora/lora.py`
- **LoRAConfig** (6 connections) — `python/sglang/srt/lora/lora.py`
- **AutoConfig** (6 connections) — `python/sglang/srt/lora/lora.py`
- **Module** (6 connections) — `python/sglang/srt/lora/lora.py`
- **._add_moe_lora_info()** (5 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **LoadConfig** (5 connections) — `python/sglang/srt/lora/lora.py`
- **BaseLoRABackend** (5 connections) — `python/sglang/srt/lora/lora.py`
- **._build_moe_gated_map()** (5 connections) — `python/sglang/srt/lora/lora.py`
- **ForwardBatch** (4 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **._prepare_lm_head_batch_info()** (4 connections) — `python/sglang/srt/lora/backend/lmhead_mixing.py`
- **.__init__()** (4 connections) — `python/sglang/srt/lora/lora.py`
- **base_backend.py** (3 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **device** (3 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **.run_lora_a_embedding()** (3 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **.run_extra_token_embedding()** (3 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **.run_lora_a_sgemm()** (3 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **.run_lora_b_sgemm()** (3 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **.run_qkv_lora()** (3 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **.run_gate_up_lora()** (3 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **.init_cuda_graph_moe_buffers()** (3 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- **dtype** (3 connections) — `python/sglang/srt/lora/backend/base_backend.py`
- *... and 31 more nodes in this community*

## Relationships

- [[Community 116]] (18 shared connections)
- [[DeepSeek MLA Attention & MoE]] (14 shared connections)
- [[Hybrid Attention Backend]] (13 shared connections)
- [[Vision-Language Model Configs]] (10 shared connections)
- [[Community 80]] (7 shared connections)
- [[Community 129]] (6 shared connections)
- [[Community 250]] (6 shared connections)
- [[Community 251]] (5 shared connections)
- [[Community 445]] (4 shared connections)
- [[Community 202]] (4 shared connections)
- [[Community 389]] (2 shared connections)
- [[Qwen3 / Kimi Model Configs]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/backend/base_backend.py`
- `python/sglang/srt/lora/backend/lmhead_mixing.py`
- `python/sglang/srt/lora/lora.py`
- `python/sglang/srt/lora/utils.py`

## Audit Trail

- EXTRACTED: 134 (54%)
- INFERRED: 114 (46%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*