# Community 53

> 122 nodes

## Key Concepts

- **CompressedTensorsConfig** (62 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- **CompressedTensorsFusedMoEMethod** (43 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- **NPUW4A16Int4DynamicMoEMethod** (27 connections) — `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- **.get_moe_scheme()** (24 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- **Module** (23 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- **._get_scheme_from_parts()** (19 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- **BaseModel** (17 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- **CompressedTensorsWNA16MoE** (17 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`
- **Module** (16 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`
- **NPUCompressedTensorsW4A16Int4DynamicMoE** (14 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`
- **QuantizationArgs** (13 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- **CompressedTensorsWNA16TritonMoE** (13 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`
- **CompressedTensorsMxInt4MoE** (12 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py`
- **.get_quant_method()** (11 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- **.get_linear_scheme()** (11 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- **.get_scheme_dict()** (7 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- **compressed_tensors_wNa16_moe.py** (7 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`
- **MoeRunnerConfig** (7 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`
- **utils.py** (7 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/utils.py`
- **find_matched_target()** (7 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/utils.py`
- **._quantization_scheme_map_from_config()** (6 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- **._check_scheme_supported()** (6 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- **GPTQMarlinState** (6 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`
- **dtype** (6 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`
- **StandardDispatchOutput** (6 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`
- *... and 97 more nodes in this community*

## Relationships

- [[Compressed-Tensors Quant Linear]] (82 shared connections)
- [[Weight Loading & EPLB]] (18 shared connections)
- [[Community 51]] (14 shared connections)
- [[DeepSeek MLA Attention & MoE]] (14 shared connections)
- [[ROCm MoE Quantization]] (8 shared connections)
- [[Vision-Language Model Configs]] (7 shared connections)
- [[Community 80]] (7 shared connections)
- [[Community 41]] (6 shared connections)
- [[NCCL Symmetric Memory]] (4 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (4 shared connections)
- [[GPTQ / Marlin Quantization]] (3 shared connections)
- [[Community 92]] (3 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/utils.py`

## Audit Trail

- EXTRACTED: 445 (70%)
- INFERRED: 195 (30%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*