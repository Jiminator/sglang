# Qwen3 / Kimi Model Configs

> 320 nodes

## Key Concepts

- **GemmaRMSNorm** (133 connections) — `python/sglang/srt/layers/layernorm.py`
- **Qwen2MoeMLP** (74 connections) — `python/sglang/srt/models/qwen2_moe.py`
- **RadixLinearAttention** (70 connections) — `python/sglang/srt/layers/radix_linear_attention.py`
- **RMSNorm** (69 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **Qwen2MoeSparseMoeBlock** (65 connections) — `python/sglang/srt/models/qwen2_moe.py`
- **get_is_capture_mode()** (39 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **Qwen3_5ForCausalLM** (36 connections) — `python/sglang/srt/models/qwen3_5.py`
- **get_layer_id()** (35 connections) — `python/sglang/srt/layers/utils/common.py`
- **Qwen3_5MoeForConditionalGeneration** (35 connections) — `python/sglang/srt/models/qwen3_5.py`
- **Qwen3_5GatedDeltaNet** (34 connections) — `python/sglang/srt/models/qwen3_5.py`
- **Tensor** (34 connections) — `python/sglang/srt/models/qwen3_5.py`
- **Qwen3NextForCausalLM** (34 connections) — `python/sglang/srt/models/qwen3_next.py`
- **FusedRMSNormGated** (32 connections) — `python/sglang/srt/layers/attention/fla/fused_norm_gate.py`
- **Qwen3_5ForConditionalGeneration** (32 connections) — `python/sglang/srt/models/qwen3_5.py`
- **Qwen3NextConfig** (31 connections) — `python/sglang/srt/configs/qwen3_next.py`
- **QuantizationConfig** (31 connections) — `python/sglang/srt/models/qwen3_5.py`
- **Tensor** (31 connections) — `python/sglang/srt/models/qwen3_next.py`
- **Qwen3_5AttentionDecoderLayer** (30 connections) — `python/sglang/srt/models/qwen3_5.py`
- **Qwen3GatedDeltaNet** (29 connections) — `python/sglang/srt/models/qwen3_next.py`
- **maybe_remap_kv_scale_name()** (28 connections) — `python/sglang/srt/model_loader/weight_utils.py`
- **Tensor** (28 connections) — `python/sglang/srt/models/kimi_linear.py`
- **ColumnParallelBatchedLinear** (27 connections) — `python/sglang/srt/layers/linear.py`
- **Qwen3_5TextConfig** (27 connections) — `python/sglang/srt/models/qwen3_5.py`
- **Qwen3HybridAttentionDecoderLayer** (27 connections) — `python/sglang/srt/models/qwen3_next.py`
- **Qwen3NextModel** (27 connections) — `python/sglang/srt/models/qwen3_next.py`
- *... and 295 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (583 shared connections)
- [[Vision-Language Model Configs]] (178 shared connections)
- [[Model Configs & Pooler]] (111 shared connections)
- [[Context-Parallel Attention]] (74 shared connections)
- [[Linear Layer Parameters]] (26 shared connections)
- [[Community 36]] (22 shared connections)
- [[Compressed-Tensors Quant Linear]] (21 shared connections)
- [[Weight Loading & EPLB]] (20 shared connections)
- [[Activation Functions & Gemma]] (16 shared connections)
- [[Community 31]] (15 shared connections)
- [[Community 46]] (13 shared connections)
- [[Community 117]] (11 shared connections)

## Source Files

- `python/sglang/srt/configs/interns2preview.py`
- `python/sglang/srt/configs/kimi_linear.py`
- `python/sglang/srt/configs/qwen3_5.py`
- `python/sglang/srt/configs/qwen3_next.py`
- `python/sglang/srt/distributed/communication_op.py`
- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/layers/attention/fla/fused_norm_gate.py`
- `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- `python/sglang/srt/layers/attention/mamba/mamba.py`
- `python/sglang/srt/layers/layernorm.py`
- `python/sglang/srt/layers/linear.py`
- `python/sglang/srt/layers/moe/mega_moe.py`
- `python/sglang/srt/layers/moe/utils.py`
- `python/sglang/srt/layers/radix_linear_attention.py`
- `python/sglang/srt/layers/utils/common.py`
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
- `python/sglang/srt/model_loader/weight_utils.py`
- `python/sglang/srt/models/apertus.py`
- `python/sglang/srt/models/bailing_moe.py`
- `python/sglang/srt/models/gemma2.py`

## Audit Trail

- EXTRACTED: 1044 (36%)
- INFERRED: 1846 (64%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*