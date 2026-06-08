# Community 88

> 71 nodes

## Key Concepts

- **marlin_utils.py** (33 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **Tensor** (25 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **marlin_make_workspace()** (14 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **ScalarType** (12 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **device** (12 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **Module** (10 connections) — `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py`
- **marlin_utils_fp4.py** (10 connections) — `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`
- **prepare_fp8_layer_for_marlin()** (10 connections) — `python/sglang/srt/layers/quantization/marlin_utils_fp8.py`
- **.process_weights_after_loading()** (9 connections) — `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py`
- **.process_weights_after_loading()** (9 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
- **marlin_permute_scales()** (9 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **should_use_atomic_add_reduce()** (9 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **prepare_nvfp4_layer_for_marlin()** (9 connections) — `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`
- **.process_weights_after_loading()** (7 connections) — `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py`
- **marlin_zero_points()** (7 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **apply_gptq_marlin_linear()** (7 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **Tensor** (7 connections) — `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`
- **marlin_utils_fp8.py** (7 connections) — `python/sglang/srt/layers/quantization/marlin_utils_fp8.py`
- **prepare_moe_fp8_layer_for_marlin()** (7 connections) — `python/sglang/srt/layers/quantization/marlin_utils_fp8.py`
- **Tensor** (6 connections) — `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py`
- **.process_weights_after_loading()** (6 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`
- **marlin_make_empty_g_idx()** (6 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **marlin_permute_bias()** (6 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **marlin_moe_permute_scales()** (6 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **awq_to_marlin_zero_points()** (6 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- *... and 46 more nodes in this community*

## Relationships

- [[Linear Layer Parameters]] (28 shared connections)
- [[Community 48]] (9 shared connections)
- [[Compressed-Tensors Quant Linear]] (9 shared connections)
- [[GPTQ / Marlin Quantization]] (8 shared connections)
- [[Community 41]] (7 shared connections)
- [[Weight Loading & EPLB]] (6 shared connections)
- [[Vision-Language Model Configs]] (5 shared connections)
- [[DeepSeek MLA Attention & MoE]] (3 shared connections)
- [[Community 47]] (3 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (2 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (2 shared connections)
- [[Community 53]] (2 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py`
- `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`
- `python/sglang/srt/layers/quantization/marlin_utils.py`
- `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`
- `python/sglang/srt/layers/quantization/marlin_utils_fp8.py`

## Audit Trail

- EXTRACTED: 272 (67%)
- INFERRED: 135 (33%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*