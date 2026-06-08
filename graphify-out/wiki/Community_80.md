# Community 80

> 76 nodes

## Key Concepts

- **FusedMoEWithLoRA** (68 connections) — `python/sglang/srt/lora/layers.py`
- **Tensor** (58 connections) — `python/sglang/srt/lora/layers.py`
- **ReplicatedLinearWithLoRA** (34 connections) — `python/sglang/srt/lora/layers.py`
- **ColumnParallelLinearWithLoRA** (24 connections) — `python/sglang/srt/lora/layers.py`
- **MergedColumnParallelLinearWithLoRA** (22 connections) — `python/sglang/srt/lora/layers.py`
- **RowParallelLinearWithLoRA** (22 connections) — `python/sglang/srt/lora/layers.py`
- **QKVParallelLinearWithLoRA** (21 connections) — `python/sglang/srt/lora/layers.py`
- **layers.py** (10 connections) — `python/sglang/srt/lora/layers.py`
- **.forward()** (8 connections) — `python/sglang/srt/lora/layers.py`
- **split_tensor_along_last_dim()** (7 connections) — `python/sglang/srt/distributed/utils.py`
- **.apply_lora()** (7 connections) — `python/sglang/srt/lora/layers.py`
- **.forward()** (6 connections) — `python/sglang/srt/lora/layers.py`
- **.forward()** (6 connections) — `python/sglang/srt/lora/layers.py`
- **._forward_with_lora()** (6 connections) — `python/sglang/srt/lora/layers.py`
- **device** (6 connections) — `python/sglang/srt/lora/trtllm_lora_temp/__init__.py`
- **get_lora_layer()** (5 connections) — `python/sglang/srt/lora/layers.py`
- **.set_lora_module()** (5 connections) — `python/sglang/srt/lora/lora_manager.py`
- **init_lora_two_stream_resources()** (5 connections) — `python/sglang/srt/lora/trtllm_lora_temp/__init__.py`
- **.extra_token_embedding()** (4 connections) — `python/sglang/srt/lora/layers.py`
- **.forward()** (4 connections) — `python/sglang/srt/lora/layers.py`
- **._get_lora_n_slices()** (4 connections) — `python/sglang/srt/lora/layers.py`
- **._get_lora_info()** (4 connections) — `python/sglang/srt/lora/layers.py`
- **.slice_moe_lora_a_weights()** (4 connections) — `python/sglang/srt/lora/layers.py`
- **.slice_moe_lora_b_weights()** (4 connections) — `python/sglang/srt/lora/layers.py`
- **.forward()** (3 connections) — `python/sglang/srt/lora/layers.py`
- *... and 51 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (67 shared connections)
- [[Community 116]] (28 shared connections)
- [[Hybrid Attention Backend]] (25 shared connections)
- [[NCCL Symmetric Memory]] (15 shared connections)
- [[Community 202]] (9 shared connections)
- [[Vision-Language Model Configs]] (7 shared connections)
- [[Model Configs & Pooler]] (7 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (7 shared connections)
- [[Community 53]] (7 shared connections)
- [[Community 111]] (7 shared connections)
- [[Community 161]] (4 shared connections)
- [[Community 227]] (1 shared connections)

## Source Files

- `python/sglang/srt/distributed/utils.py`
- `python/sglang/srt/lora/layers.py`
- `python/sglang/srt/lora/lora_manager.py`
- `python/sglang/srt/lora/trtllm_lora_temp/__init__.py`

## Audit Trail

- EXTRACTED: 276 (63%)
- INFERRED: 165 (37%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*