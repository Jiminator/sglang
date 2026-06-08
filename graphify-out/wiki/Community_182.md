# Community 182

> 38 nodes

## Key Concepts

- **virtual_experts.py** (15 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **Tensor** (11 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **_merged_experts_fused_moe_lora_add_impl()** (9 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **_invoke_moe_lora_shrink_splitk()** (7 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **_get_moe_lora_shrink_split_k()** (6 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **_align_block_size_large()** (5 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **_invoke_moe_lora_expand_add()** (4 connections) — `python/sglang/srt/lora/trtllm_lora_temp/specialized_expand.py`
- **_fused_virtual_topk_ids()** (4 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **fused_sanitize_expert_ids()** (4 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **_align_block_size_jit()** (4 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **_align_block_size_torch()** (4 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **merged_experts_fused_moe_lora_add()** (4 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **specialized_expand.py** (3 connections) — `python/sglang/srt/lora/trtllm_lora_temp/specialized_expand.py`
- **_moe_lora_expand_add_kernel()** (3 connections) — `python/sglang/srt/lora/trtllm_lora_temp/specialized_expand.py`
- **_fused_virtual_topk_ids_kernel()** (3 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **constexpr** (3 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **_moe_lora_shrink_splitk_kernel()** (3 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **_merged_experts_fused_moe_lora_add_op()** (3 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **_fused_sanitize_expert_ids_kernel()** (2 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **Any** (2 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **_merged_experts_fused_moe_lora_add_fake()** (2 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`
- **constexpr** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/specialized_expand.py`
- **Tensor** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/specialized_expand.py`
- **Rank-specialized LoRA-B expand for virtual-expert LoRA.  The kernel here was ori** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/specialized_expand.py`
- **Rank-specialized LoRA-B expand for virtual-expert LoRA.      ``GATED_A_HALF`` >** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/specialized_expand.py`
- *... and 13 more nodes in this community*

## Relationships

- [[Community 205]] (3 shared connections)
- [[Community 96]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/trtllm_lora_temp/specialized_expand.py`
- `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py`

## Audit Trail

- EXTRACTED: 112 (95%)
- INFERRED: 6 (5%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*