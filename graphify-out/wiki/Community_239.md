# Community 239

> 28 nodes

## Key Concepts

- **virtual_experts.py** (14 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **Tensor** (10 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **_merged_experts_fused_moe_lora_add_impl()** (7 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **_invoke_moe_lora_shrink_splitk()** (5 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **_align_block_size_large()** (5 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **_align_block_size_jit()** (4 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **_align_block_size_torch()** (4 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **merged_experts_fused_moe_lora_add()** (4 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **_fused_virtual_topk_ids_kernel()** (3 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **constexpr** (3 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **_fused_virtual_topk_ids()** (3 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **fused_sanitize_expert_ids()** (3 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **_moe_lora_shrink_splitk_kernel()** (3 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **_merged_experts_fused_moe_lora_add_op()** (3 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **_fused_sanitize_expert_ids_kernel()** (2 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **_merged_experts_fused_moe_lora_add_fake()** (2 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **Any** (1 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **LoRA Virtual Experts Triton Ops.** (1 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **Fuses _get_virtual_topk_ids: comparison + clamp + arithmetic into one kernel.** (1 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **Returns virtual topk_ids, token_lora_mask, and virtual_num_experts.** (1 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **Sanitize expert_ids by replacing values >= num_virtual_experts with -1.      Ret** (1 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **Split-K grouped GEMM for the LoRA A (shrink) stage with few virtual experts.** (1 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **Launch split-K shrink kernel for LoRA A with few virtual experts.** (1 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **CUDA JIT align_block_size for num_experts > 1024 (up to 8191).      Uses the v2** (1 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- **Pure-PyTorch align_block_size for num_experts > 1024, compiled via torch.compile** (1 connections) — `python/sglang/srt/lora/triton_ops/virtual_experts.py`
- *... and 3 more nodes in this community*

## Relationships

- [[Community 96]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/triton_ops/virtual_experts.py`

## Audit Trail

- EXTRACTED: 86 (99%)
- INFERRED: 1 (1%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*