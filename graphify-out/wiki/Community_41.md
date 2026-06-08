# Community 41

> 147 nodes

## Key Concepts

- **AWQConfig** (58 connections) — `python/sglang/srt/layers/quantization/awq/awq.py`
- **AWQMarlinConfig** (42 connections) — `python/sglang/srt/layers/quantization/awq/awq.py`
- **AWQLinearScheme** (25 connections) — `python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py`
- **AWQMoEScheme** (25 connections) — `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py`
- **Module** (24 connections) — `python/sglang/srt/layers/quantization/awq/awq.py`
- **AWQLinearMethod** (24 connections) — `python/sglang/srt/layers/quantization/awq/awq.py`
- **AWQMoEMethod** (24 connections) — `python/sglang/srt/layers/quantization/awq/awq.py`
- **AWQCPUConfig** (18 connections) — `python/sglang/srt/layers/quantization/awq/awq.py`
- **AWQLinearSchemeBase** (18 connections) — `python/sglang/srt/layers/quantization/awq/schemes/awq_scheme.py`
- **_()** (17 connections) — `python/sglang/srt/layers/quantization/awq/awq.py`
- **AWQAscendMoEKernel** (16 connections) — `python/sglang/srt/hardware_backend/npu/quantization/awq_kernels.py`
- **__init__.py** (14 connections) — `python/sglang/srt/layers/quantization/awq/__init__.py`
- **__init__.py** (14 connections) — `python/sglang/srt/layers/quantization/awq/schemes/__init__.py`
- **AWQMarlinLinearScheme** (14 connections) — `python/sglang/srt/layers/quantization/awq/schemes/awq_marlin.py`
- **dtype** (13 connections) — `python/sglang/srt/layers/quantization/awq/awq.py`
- **AWQAscendLinearScheme** (13 connections) — `python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py`
- **AWQAscendMoEScheme** (13 connections) — `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py`
- **AWQLinearKernel** (12 connections) — `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py`
- **AWQMoEKernel** (12 connections) — `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py`
- **AWQAscendLinearKernel** (12 connections) — `python/sglang/srt/hardware_backend/npu/quantization/awq_kernels.py`
- **.get_quant_method()** (12 connections) — `python/sglang/srt/layers/quantization/awq/awq.py`
- **AWQMarlinLinearKernel** (11 connections) — `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py`
- **LinearMethodBase** (11 connections) — `python/sglang/srt/layers/quantization/awq/awq.py`
- **AWQIntelAMXMoEScheme** (11 connections) — `python/sglang/srt/layers/quantization/awq/schemes/awq_cpu.py`
- **Module** (10 connections) — `python/sglang/srt/layers/quantization/awq/schemes/awq_cpu.py`
- *... and 122 more nodes in this community*

## Relationships

- [[Compressed-Tensors Quant Linear]] (69 shared connections)
- [[GPTQ / Marlin Quantization]] (37 shared connections)
- [[Linear Layer Parameters]] (26 shared connections)
- [[Vision-Language Model Configs]] (18 shared connections)
- [[ROCm MoE Quantization]] (12 shared connections)
- [[DeepSeek MLA Attention & MoE]] (8 shared connections)
- [[Community 88]] (7 shared connections)
- [[Community 53]] (6 shared connections)
- [[Weight Loading & EPLB]] (5 shared connections)
- [[Community 47]] (2 shared connections)
- [[Community 48]] (1 shared connections)
- [[Community 92]] (1 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py`
- `python/sglang/srt/hardware_backend/npu/quantization/awq_kernels.py`
- `python/sglang/srt/layers/quantization/awq/__init__.py`
- `python/sglang/srt/layers/quantization/awq/awq.py`
- `python/sglang/srt/layers/quantization/awq/awq_triton.py`
- `python/sglang/srt/layers/quantization/awq/schemes/__init__.py`
- `python/sglang/srt/layers/quantization/awq/schemes/awq_cpu.py`
- `python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py`
- `python/sglang/srt/layers/quantization/awq/schemes/awq_marlin.py`
- `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py`
- `python/sglang/srt/layers/quantization/awq/schemes/awq_scheme.py`

## Audit Trail

- EXTRACTED: 567 (65%)
- INFERRED: 307 (35%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*