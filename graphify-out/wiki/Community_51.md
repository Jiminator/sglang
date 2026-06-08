# Community 51

> 123 nodes

## Key Concepts

- **NPUW4A8Int8DynamicMoEMethod** (25 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- **NPUW8A8Int8DynamicMoEMethod** (20 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- **fused_moe_method_npu.py** (14 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- **NPUW4A4Int4DynamicMoEMethod** (14 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- **npu_format_cast()** (14 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **Module** (13 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- **NPUCompressedTensorsW4A8Int8DynamicMoE** (13 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int8_moe.py`
- **_NPUFusedMoEMethodBase** (12 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- **Tensor** (11 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- **process_fuseep_weights()** (10 connections) — `python/sglang/srt/hardware_backend/npu/moe/fuseep.py`
- **NPUCompressedTensorsW8A8Int8DynamicMoE** (10 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py`
- **ModelSlimW4A4Int4MoE** (10 connections) — `python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a4_int4_moe.py`
- **ModelSlimW4A8Int8MoE** (10 connections) — `python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a8_int8_moe.py`
- **ModelSlimW8A8Int8MoE** (10 connections) — `python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w8a8_int8_moe.py`
- **fuseep.py** (8 connections) — `python/sglang/srt/hardware_backend/npu/moe/fuseep.py`
- **.process_weights_after_loading()** (8 connections) — `python/sglang/srt/layers/quantization/unquant.py`
- **._maybe_apply_deepep()** (7 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- **CompressedTensorsMoEScheme** (7 connections)
- **Module** (7 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int8_moe.py`
- **Tensor** (6 connections) — `python/sglang/srt/hardware_backend/npu/moe/fuseep.py`
- **maybe_apply_fuseep_weights()** (6 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- **.process_weights_after_loading()** (6 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- **.create_weights()** (6 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int8_moe.py`
- **npu_fused_experts()** (5 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- **maybe_apply_deepep_npu()** (5 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- *... and 98 more nodes in this community*

## Relationships

- [[Community 53]] (14 shared connections)
- [[Compressed-Tensors Quant Linear]] (12 shared connections)
- [[ROCm MoE Quantization]] (10 shared connections)
- [[Activation Functions & Gemma]] (7 shared connections)
- [[Community 92]] (7 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (6 shared connections)
- [[Vision-Language Model Configs]] (6 shared connections)
- [[Breakable CUDA Graph (TBO)]] (3 shared connections)
- [[Community 150]] (3 shared connections)
- [[Community 330]] (3 shared connections)
- [[CLI Arg Parsing & Deprecation]] (2 shared connections)
- [[Community 66]] (2 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/npu/moe/fuseep.py`
- `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- `python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py`
- `python/sglang/srt/hardware_backend/npu/utils.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int8_moe.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py`
- `python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a4_int4_moe.py`
- `python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a8_int8_moe.py`
- `python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w8a8_int8_moe.py`
- `python/sglang/srt/layers/quantization/unquant.py`

## Audit Trail

- EXTRACTED: 380 (76%)
- INFERRED: 117 (24%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*