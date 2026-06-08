# ROCm MoE Quantization

> 217 nodes

## Key Concepts

- **QuarkConfig** (54 connections) — `python/sglang/srt/layers/quantization/quark/quark.py`
- **BaseMoEScheme** (31 connections) — `python/sglang/srt/layers/quantization/base_scheme.py`
- **BaseLinearScheme** (30 connections) — `python/sglang/srt/layers/quantization/base_scheme.py`
- **CompressedTensorsW8A8Fp8** (29 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`
- **__init__.py** (28 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py`
- **AWQMoESchemeBase** (14 connections) — `python/sglang/srt/layers/quantization/awq/schemes/awq_scheme.py`
- **quark_w4a4_mxfp4.py** (14 connections) — `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`
- **__init__.py** (13 connections) — `python/sglang/srt/layers/quantization/modelslim/schemes/__init__.py`
- **QuarkW4A4MXFp4MoE** (13 connections) — `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`
- **CompressedTensorsW8A16Fp8** (12 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py`
- **QuarkMoEScheme** (12 connections) — `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py`
- **QuarkW4A4MXFP4** (12 connections) — `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`
- **QuarkW8A8Fp8** (12 connections) — `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`
- **__init__.py** (11 connections) — `python/sglang/srt/layers/quantization/quark/schemes/__init__.py`
- **QuarkLinearScheme** (11 connections) — `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py`
- **Tensor** (11 connections) — `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`
- **QuarkW8A8FP8MoE** (11 connections) — `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py`
- **CompressedTensorsMoEScheme** (10 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_scheme.py`
- **rocm_moe_utils.py** (9 connections) — `python/sglang/srt/layers/moe/rocm_moe_utils.py`
- **CompressedTensorsLinearScheme** (9 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_scheme.py`
- **ModelSlimMoEScheme** (9 connections) — `python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_scheme.py`
- **._get_scheme_from_config()** (9 connections) — `python/sglang/srt/layers/quantization/quark/quark.py`
- **.get_moe_scheme()** (9 connections) — `python/sglang/srt/layers/quantization/quark/quark.py`
- **utils.py** (9 connections) — `python/sglang/srt/layers/quantization/quark/utils.py`
- **ModelSlimLinearScheme** (8 connections) — `python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_scheme.py`
- *... and 192 more nodes in this community*

## Relationships

- [[Compressed-Tensors Quant Linear]] (48 shared connections)
- [[Linear Layer Parameters]] (21 shared connections)
- [[Vision-Language Model Configs]] (20 shared connections)
- [[DeepSeek MLA Attention & MoE]] (18 shared connections)
- [[Community 41]] (12 shared connections)
- [[Weight Loading & EPLB]] (11 shared connections)
- [[GPTQ / Marlin Quantization]] (11 shared connections)
- [[Community 51]] (10 shared connections)
- [[Community 53]] (8 shared connections)
- [[Aibrix KV Cache Storage]] (6 shared connections)
- [[Community 150]] (6 shared connections)
- [[Community 48]] (5 shared connections)

## Source Files

- `python/sglang/srt/layers/moe/rocm_moe_utils.py`
- `python/sglang/srt/layers/quantization/awq/schemes/awq_scheme.py`
- `python/sglang/srt/layers/quantization/base_scheme.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_scheme.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
- `python/sglang/srt/layers/quantization/gptq/schemes/gptq_scheme.py`
- `python/sglang/srt/layers/quantization/modelslim/schemes/__init__.py`
- `python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_scheme.py`
- `python/sglang/srt/layers/quantization/mxfp4.py`
- `python/sglang/srt/layers/quantization/quark/quark.py`
- `python/sglang/srt/layers/quantization/quark/schemes/__init__.py`
- `python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py`
- `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`
- `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`
- `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`

## Audit Trail

- EXTRACTED: 642 (76%)
- INFERRED: 206 (24%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*