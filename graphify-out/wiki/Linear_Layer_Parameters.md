# Linear Layer Parameters

> 229 nodes

## Key Concepts

- **ChannelQuantScaleParameter** (113 connections) — `python/sglang/srt/layers/parameter.py`
- **PackedvLLMParameter** (78 connections) — `python/sglang/srt/layers/parameter.py`
- **GroupQuantScaleParameter** (74 connections) — `python/sglang/srt/layers/parameter.py`
- **BasevLLMParameter** (58 connections) — `python/sglang/srt/layers/parameter.py`
- **RowvLLMParameter** (49 connections) — `python/sglang/srt/layers/parameter.py`
- **PackedColumnParameter** (48 connections) — `python/sglang/srt/layers/parameter.py`
- **Tensor** (28 connections) — `python/sglang/srt/layers/linear.py`
- **MergedColumnParallelRepeatedLinear** (28 connections) — `python/sglang/srt/layers/linear.py`
- **Parameter** (22 connections) — `python/sglang/srt/layers/linear.py`
- **QoQConfig** (22 connections) — `python/sglang/srt/layers/quantization/qoq.py`
- **MarlinConfig** (21 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **divide()** (20 connections) — `python/sglang/srt/distributed/utils.py`
- **GPTQLinearScheme** (20 connections) — `python/sglang/srt/layers/quantization/gptq/schemes/gptq_linear.py`
- **GPTQMoEIntelAMXMethod** (18 connections) — `python/sglang/srt/layers/quantization/gptq_cpu.py`
- **dtype** (17 connections) — `python/sglang/srt/layers/linear.py`
- **Module** (17 connections) — `python/sglang/srt/layers/quantization/gptq_cpu.py`
- **GPTQLinearIntelAMXMethod** (17 connections) — `python/sglang/srt/layers/quantization/gptq_cpu.py`
- **QuantizationConfig** (16 connections) — `python/sglang/srt/layers/linear.py`
- **BasevLLMParameter** (16 connections) — `python/sglang/srt/layers/linear.py`
- **CompressedTensorsWNA16** (16 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
- **CPUGPTQConfig** (16 connections) — `python/sglang/srt/layers/quantization/gptq_cpu.py`
- **MarlinLinearMethod** (16 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- **parameter.py** (15 connections) — `python/sglang/srt/layers/parameter.py`
- **QoQLinearMethod** (15 connections) — `python/sglang/srt/layers/quantization/qoq.py`
- **MarlinLinearLayerConfig** (14 connections) — `python/sglang/srt/layers/quantization/marlin_utils.py`
- *... and 204 more nodes in this community*

## Relationships

- [[Compressed-Tensors Quant Linear]] (164 shared connections)
- [[GPTQ / Marlin Quantization]] (63 shared connections)
- [[DeepSeek MLA Attention & MoE]] (40 shared connections)
- [[Weight Loading & EPLB]] (34 shared connections)
- [[Vision-Language Model Configs]] (30 shared connections)
- [[Community 88]] (28 shared connections)
- [[Qwen3 / Kimi Model Configs]] (26 shared connections)
- [[Community 41]] (26 shared connections)
- [[ROCm MoE Quantization]] (21 shared connections)
- [[Community 150]] (17 shared connections)
- [[Context-Parallel Attention]] (13 shared connections)
- [[Model Configs & Pooler]] (8 shared connections)

## Source Files

- `python/sglang/srt/distributed/utils.py`
- `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py`
- `python/sglang/srt/hardware_backend/npu/quantization/gptq_kernels.py`
- `python/sglang/srt/layers/linear.py`
- `python/sglang/srt/layers/parameter.py`
- `python/sglang/srt/layers/quantization/awq/schemes/awq_marlin.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
- `python/sglang/srt/layers/quantization/gptq/schemes/gptq_linear.py`
- `python/sglang/srt/layers/quantization/gptq/schemes/gptq_marlin.py`
- `python/sglang/srt/layers/quantization/gptq_cpu.py`
- `python/sglang/srt/layers/quantization/marlin_utils.py`
- `python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w8a8_int8.py`
- `python/sglang/srt/layers/quantization/qoq.py`
- `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`
- `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`
- `python/sglang/srt/layers/utils/common.py`
- `python/sglang/srt/model_loader/weight_utils.py`

## Audit Trail

- EXTRACTED: 739 (45%)
- INFERRED: 920 (55%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*