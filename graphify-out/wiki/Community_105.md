# Community 105

> 58 nodes

## Key Concepts

- **kernels.py** (42 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **constexpr** (20 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **Tensor** (11 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **cutlass_w4a8_moe()** (9 connections) — `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`
- **cutlass_w4a8_moe_deepep_ll()** (7 connections) — `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`
- **cutlass_w4a8_moe_deepep_normal()** (5 connections) — `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`
- **silu_and_mul_masked_post_quant_fwd()** (5 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **cutlass_w4a8_moe.py** (4 connections) — `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`
- **silu_and_mul_masked_fwd()** (4 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **silu_mul_static_tensorwise_quant_for_cutlass_moe()** (4 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **ep_scatter()** (4 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **tma_align_input_scale()** (4 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **moe_ep_deepgemm_preprocess()** (4 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **silu_and_mul_masked_post_per_tensor_quant_fwd()** (4 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **Tensor** (3 connections) — `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`
- **_get_launch_config_2d()** (3 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **deepep_run_moe_deep_preprocess()** (3 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **cutlass_w4_run_moe_ep_preproess()** (3 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **pre_reorder_for_cutlass_moe()** (3 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **post_reorder_for_cutlass_moe()** (3 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **ep_gather()** (3 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **deepep_ll_get_cutlass_w4a8_moe_mm_data()** (3 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **_silu_and_mul_post_per_tensor_quant_kernel()** (3 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **fp8_per_token_to_per_tensor_quant_triton()** (3 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- **_get_launch_config_1d()** (2 connections) — `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- *... and 33 more nodes in this community*

## Relationships

- [[MoE Dispatch/Combine (Cutlass)]] (6 shared connections)
- [[Compressed-Tensors Quant Linear]] (3 shared connections)
- [[Community 45]] (3 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)
- [[Community 48]] (1 shared connections)
- [[DeepSeek MLA Attention & MoE]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`
- `python/sglang/srt/layers/moe/ep_moe/kernels.py`

## Audit Trail

- EXTRACTED: 184 (86%)
- INFERRED: 29 (14%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*