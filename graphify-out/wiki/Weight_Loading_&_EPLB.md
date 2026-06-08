# Weight Loading & EPLB

> 306 nodes

## Key Concepts

- **PerTensorScaleParameter** (126 connections) — `python/sglang/srt/layers/parameter.py`
- **Fp8Config** (124 connections) — `python/sglang/srt/layers/quantization/fp8.py`
- **MarlinMoeQuantInfo** (66 connections) — `python/sglang/srt/layers/moe/moe_runner/marlin.py`
- **BaseKVCacheMethod** (53 connections) — `python/sglang/srt/layers/quantization/kv_cache.py`
- **FlashInferTrtllmFp8MoeQuantInfo** (51 connections) — `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- **weight_utils.py** (51 connections) — `python/sglang/srt/model_loader/weight_utils.py`
- **ModelOptFp8Config** (45 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **ModelOptFp4LinearMethod** (45 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **ModelOptFp4Config** (44 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **ModelOptNvFp4FusedMoEMethod** (44 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **tqdm** (36 connections) — `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`
- **CutlassMoEParams** (34 connections) — `python/sglang/srt/layers/moe/cutlass_moe_params.py`
- **Module** (34 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **CuteDslFp4MoeQuantInfo** (33 connections) — `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py`
- **CutlassMoEType** (31 connections) — `python/sglang/srt/layers/moe/cutlass_moe_params.py`
- **ModelOptMixedPrecisionConfig** (31 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **FlashInferTrtllmFp4MoeQuantInfo** (28 connections) — `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- **ModelOptQuantConfig** (28 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **ActivationType** (27 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **dtype** (26 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **ModelOptFp8MoEMethod** (26 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **ModelOptFp8LinearMethod** (25 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **Tensor** (24 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **ModelOptFp8KVCacheMethod** (22 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- **Any** (21 connections) — `python/sglang/srt/layers/quantization/modelopt_quant.py`
- *... and 281 more nodes in this community*

## Relationships

- [[Compressed-Tensors Quant Linear]] (302 shared connections)
- [[DeepSeek MLA Attention & MoE]] (67 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (42 shared connections)
- [[Linear Layer Parameters]] (34 shared connections)
- [[Community 35]] (34 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (30 shared connections)
- [[Vision-Language Model Configs]] (30 shared connections)
- [[NCCL Symmetric Memory]] (24 shared connections)
- [[Hybrid Attention Backend]] (22 shared connections)
- [[Qwen3 / Kimi Model Configs]] (20 shared connections)
- [[Community 53]] (18 shared connections)
- [[GPTQ / Marlin Quantization]] (17 shared connections)

## Source Files

- `python/sglang/srt/eplb/eplb_simulator/__init__.py`
- `python/sglang/srt/eplb/eplb_simulator/reader.py`
- `python/sglang/srt/layers/moe/cutlass_moe.py`
- `python/sglang/srt/layers/moe/cutlass_moe_params.py`
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py`
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- `python/sglang/srt/layers/moe/moe_runner/marlin.py`
- `python/sglang/srt/layers/parameter.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`
- `python/sglang/srt/layers/quantization/fp4_utils.py`
- `python/sglang/srt/layers/quantization/fp8.py`
- `python/sglang/srt/layers/quantization/fp8_utils.py`
- `python/sglang/srt/layers/quantization/kv_cache.py`
- `python/sglang/srt/layers/quantization/modelopt_quant.py`
- `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`
- `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py`
- `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`
- `python/sglang/srt/layers/quantization/utils.py`
- `python/sglang/srt/layers/quantization/w8a8_fp8.py`

## Audit Trail

- EXTRACTED: 936 (44%)
- INFERRED: 1188 (56%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*