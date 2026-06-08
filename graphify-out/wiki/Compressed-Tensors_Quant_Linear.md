# Compressed-Tensors Quant Linear

> 466 nodes

## Key Concepts

- **LinearBase** (283 connections) — `python/sglang/srt/layers/linear.py`
- **QuantizeMethodBase** (279 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **FusedMoEMethodBase** (261 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **UnquantizedLinearMethod** (249 connections) — `python/sglang/srt/layers/quantization/unquant.py`
- **LinearMethodBase** (218 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **ModelWeightParameter** (157 connections) — `python/sglang/srt/layers/parameter.py`
- **TritonMoeQuantInfo** (143 connections) — `python/sglang/srt/layers/moe/moe_runner/triton.py`
- **Fp8LinearMethod** (97 connections) — `python/sglang/srt/layers/quantization/fp8.py`
- **UnquantizedFusedMoEMethod** (89 connections) — `python/sglang/srt/layers/quantization/unquant.py`
- **AiterMoeQuantInfo** (82 connections) — `python/sglang/srt/layers/moe/moe_runner/aiter.py`
- **BlockQuantScaleParameter** (67 connections) — `python/sglang/srt/layers/parameter.py`
- **CPUQuantMethod** (63 connections) — `python/sglang/srt/layers/amx_utils.py`
- **AiterQuantType** (60 connections) — `python/sglang/srt/layers/moe/moe_runner/aiter.py`
- **Fp8MoEMethod** (53 connections) — `python/sglang/srt/layers/quantization/fp8.py`
- **W4AFp8Config** (42 connections) — `python/sglang/srt/layers/quantization/w4afp8.py`
- **Module** (41 connections) — `python/sglang/srt/layers/quantization/fp8.py`
- **Mxfp4FlashinferCutlassMoEMethod** (31 connections) — `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`
- **Mxfp4MarlinMoEMethod** (31 connections) — `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`
- **TritonKernelsQuantInfo** (30 connections) — `python/sglang/srt/layers/moe/moe_runner/triton_kernels.py`
- **Fp8KVCacheMethod** (27 connections) — `python/sglang/srt/layers/quantization/fp8.py`
- **Module** (27 connections) — `python/sglang/srt/layers/quantization/unquant.py`
- **dtype** (26 connections) — `python/sglang/srt/layers/quantization/fp8.py`
- **W8A8Int8Config** (26 connections) — `python/sglang/srt/layers/quantization/w8a8_int8.py`
- **Mxfp4Config** (25 connections) — `python/sglang/srt/layers/quantization/mxfp4.py`
- **W4AFp8MoEMethod** (25 connections) — `python/sglang/srt/layers/quantization/w4afp8.py`
- *... and 441 more nodes in this community*

## Relationships

- [[Weight Loading & EPLB]] (302 shared connections)
- [[Vision-Language Model Configs]] (173 shared connections)
- [[Linear Layer Parameters]] (164 shared connections)
- [[GPTQ / Marlin Quantization]] (132 shared connections)
- [[DeepSeek MLA Attention & MoE]] (104 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (101 shared connections)
- [[Community 53]] (82 shared connections)
- [[Community 92]] (80 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (70 shared connections)
- [[Community 41]] (69 shared connections)
- [[ROCm MoE Quantization]] (48 shared connections)
- [[Community 48]] (44 shared connections)

## Source Files

- `python/sglang/srt/layers/amx_utils.py`
- `python/sglang/srt/layers/linear.py`
- `python/sglang/srt/layers/moe/moe_runner/aiter.py`
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_mxfp4.py`
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- `python/sglang/srt/layers/moe/moe_runner/triton.py`
- `python/sglang/srt/layers/moe/moe_runner/triton_kernels.py`
- `python/sglang/srt/layers/moe/topk.py`
- `python/sglang/srt/layers/parameter.py`
- `python/sglang/srt/layers/quantization/awq/awq.py`
- `python/sglang/srt/layers/quantization/base_config.py`
- `python/sglang/srt/layers/quantization/bitsandbytes.py`
- `python/sglang/srt/layers/quantization/blockwise_int8.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py`
- `python/sglang/srt/layers/quantization/fp8.py`
- `python/sglang/srt/layers/quantization/fpgemm_fp8.py`
- `python/sglang/srt/layers/quantization/gguf.py`
- `python/sglang/srt/layers/quantization/gptq/gptq.py`
- `python/sglang/srt/layers/quantization/modelslim/modelslim.py`

## Audit Trail

- EXTRACTED: 1468 (28%)
- INFERRED: 3805 (72%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*