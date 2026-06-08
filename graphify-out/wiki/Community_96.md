# Community 96

> 66 nodes

## Key Concepts

- **fused_moe.py** (22 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
- **fused_moe_triton_kernels.py** (20 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
- **invoke_fused_moe_kernel()** (14 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
- **_fused_moe_kernel_sequence()** (13 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
- **fused_moe()** (10 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
- **fused_experts()** (9 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
- **_prepare_fused_moe_run()** (9 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
- **moe_align_block_size()** (8 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py`
- **Tensor** (7 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
- **constexpr** (7 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
- **fused_experts_impl()** (6 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
- **fused_moe_triton_config.py** (6 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py`
- **get_moe_configs()** (6 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py`
- **act_and_mul_triton()** (6 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
- **try_get_optimal_moe_config()** (5 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py`
- **_apply_activation()** (5 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
- **moe_sum_reduce_triton()** (5 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
- **inplace_fused_experts()** (4 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
- **outplace_fused_experts()** (4 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
- **get_config_dtype_str()** (4 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py`
- **fused_moe_kernel_gptq_awq()** (4 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
- **fused_moe_kernel()** (4 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
- **_get_b_tma_desc_cached()** (4 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
- **Tensor** (4 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
- **act_and_mul_kernel()** (4 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
- *... and 41 more nodes in this community*

## Relationships

- [[MoE Dispatch/Combine (Cutlass)]] (4 shared connections)
- [[Context-Parallel Attention]] (3 shared connections)
- [[DeepSeek MLA Attention & MoE]] (2 shared connections)
- [[Community 48]] (2 shared connections)
- [[Weight Loading & EPLB]] (2 shared connections)
- [[Compressed-Tensors Quant Linear]] (1 shared connections)
- [[Activation Functions & Gemma]] (1 shared connections)
- [[Community 198]] (1 shared connections)
- [[Community 148]] (1 shared connections)
- [[Community 239]] (1 shared connections)
- [[Community 182]] (1 shared connections)
- [[Community 422]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
- `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py`
- `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
- `python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py`

## Audit Trail

- EXTRACTED: 236 (91%)
- INFERRED: 24 (9%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*