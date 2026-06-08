# NCCL Symmetric Memory

> 287 nodes

## Key Concepts

- **get_tp_group()** (62 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **dp_attention.py** (54 connections) — `python/sglang/srt/layers/dp_attention.py`
- **use_symmetric_memory()** (36 connections) — `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`
- **is_allocation_symmetric()** (32 connections) — `python/sglang/srt/layers/dp_attention.py`
- **get_attention_tp_group()** (28 connections) — `python/sglang/srt/layers/dp_attention.py`
- **Tensor** (23 connections) — `python/sglang/srt/layers/communicator.py`
- **ForwardBatch** (22 connections) — `python/sglang/srt/layers/communicator.py`
- **Tensor** (22 connections) — `python/sglang/srt/layers/dp_attention.py`
- **._gather_hidden_states_and_residual()** (21 connections) — `python/sglang/srt/layers/communicator.py`
- **flashinfer_trtllm.py** (20 connections) — `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- **cp_all_gather_rerange_output()** (18 connections) — `python/sglang/srt/layers/utils/cp_utils.py`
- **_DpGatheredBufferWrapper** (17 connections) — `python/sglang/srt/layers/dp_attention.py`
- **fused_experts_none_to_flashinfer_trtllm_fp4()** (17 connections) — `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- **fused_experts_none_to_flashinfer_trtllm_fp8()** (16 connections) — `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- **communication_op.py** (15 connections) — `python/sglang/srt/distributed/communication_op.py`
- **tensor_model_parallel_all_gather()** (15 connections) — `python/sglang/srt/distributed/communication_op.py`
- **fused_experts_fp8_sgl()** (14 connections) — `python/sglang/srt/lora/trtllm_lora_temp/sgl_fp8_moe.py`
- **__init__.py** (13 connections) — `python/sglang/srt/lora/trtllm_lora_temp/__init__.py`
- **AttnTpContext** (12 connections) — `python/sglang/srt/layers/communicator.py`
- **.prepare_attn()** (12 connections) — `python/sglang/srt/layers/communicator.py`
- **._scatter_hidden_states_moe()** (12 connections) — `python/sglang/srt/layers/communicator.py`
- **fused_experts_none_to_flashinfer_trtllm_bf16()** (12 connections) — `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- **.apply()** (12 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **get_lora_side_stream()** (12 connections) — `python/sglang/srt/lora/trtllm_lora_temp/__init__.py`
- **._scatter_hidden_states()** (11 connections) — `python/sglang/srt/layers/communicator.py`
- *... and 262 more nodes in this community*

## Relationships

- [[Context-Parallel Attention]] (53 shared connections)
- [[DeepSeek MLA Attention & MoE]] (39 shared connections)
- [[Weight Loading & EPLB]] (24 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (19 shared connections)
- [[Batch-Overlap Operations]] (18 shared connections)
- [[Breakable CUDA Graph (TBO)]] (17 shared connections)
- [[Community 80]] (15 shared connections)
- [[Compressed-Tensors Quant Linear]] (15 shared connections)
- [[Community 37]] (15 shared connections)
- [[Model Config & Encode Server]] (14 shared connections)
- [[Vision-Language Model Configs]] (14 shared connections)
- [[CLI Arg Parsing & Deprecation]] (13 shared connections)

## Source Files

- `python/sglang/srt/distributed/communication_op.py`
- `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`
- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/layers/communicator.py`
- `python/sglang/srt/layers/dp_attention.py`
- `python/sglang/srt/layers/linear.py`
- `python/sglang/srt/layers/logits_processor.py`
- `python/sglang/srt/layers/moe/flashinfer_trtllm_moe.py`
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_mxfp4.py`
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
- `python/sglang/srt/layers/moe/topk.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py`
- `python/sglang/srt/layers/quantization/modelopt_quant.py`
- `python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py`
- `python/sglang/srt/layers/sampler.py`
- `python/sglang/srt/layers/utils/cp_utils.py`
- `python/sglang/srt/layers/vocab_parallel_embedding.py`
- `python/sglang/srt/lora/trtllm_lora_temp/__init__.py`
- `python/sglang/srt/lora/trtllm_lora_temp/attention.py`

## Audit Trail

- EXTRACTED: 971 (65%)
- INFERRED: 528 (35%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*