# Community 45

> 134 nodes

## Key Concepts

- **fp8_utils.py** (57 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **get_bool_env_var()** (41 connections) — `python/sglang/srt/utils/common.py`
- **Tensor** (32 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **normalize_e4m3fn_to_e4m3fnuz()** (16 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **Fp8GemmRunnerBackend** (16 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **apply_fp8_linear()** (15 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **_dispatch_explicit_backend()** (14 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **.post_load_weights()** (13 connections) — `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- **ceil_div()** (13 connections) — `python/sglang/srt/utils/common.py`
- **dtype** (10 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **block_quant_dequant()** (10 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **.post_load_weights()** (10 connections) — `python/sglang/srt/models/longcat_flash.py`
- **.post_load_weights()** (10 connections) — `python/sglang/srt/models/longcat_flash_nextn.py`
- **get_fp8_gemm_runner_backend()** (9 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **_raw_triton_mxfp8_blockscaled_linear()** (9 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **flashinfer_mxfp8_blockscaled_linear()** (9 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **block_quant_to_tensor_quant()** (9 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **.post_load_weights()** (9 connections) — `python/sglang/srt/models/bailing_moe_linear.py`
- **.create_weights()** (8 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`
- **.__init__()** (8 connections) — `python/sglang/srt/layers/quantization/fp8.py`
- **.process_weights_after_loading_block_quant()** (8 connections) — `python/sglang/srt/layers/quantization/fp8.py`
- **dispatch_w8a8_block_fp8_linear()** (8 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **channel_quant_to_tensor_quant()** (8 connections) — `python/sglang/srt/layers/quantization/fp8_utils.py`
- **MXFP4QuantizeUtil** (8 connections) — `python/sglang/srt/layers/quantization/mxfp4_tensor.py`
- **_postprocess_tensors()** (8 connections) — `python/sglang/srt/utils/weight_checker.py`
- *... and 109 more nodes in this community*

## Relationships

- [[Compressed-Tensors Quant Linear]] (29 shared connections)
- [[Community 48]] (17 shared connections)
- [[DeepSeek MLA Attention & MoE]] (13 shared connections)
- [[Weight Loading & EPLB]] (10 shared connections)
- [[Community 42]] (7 shared connections)
- [[Community 405]] (6 shared connections)
- [[Linear Layer Parameters]] (6 shared connections)
- [[ROCm MoE Quantization]] (5 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (5 shared connections)
- [[CLI Arg Parsing & Deprecation]] (5 shared connections)
- [[Context-Parallel Attention]] (4 shared connections)
- [[Community 105]] (3 shared connections)

## Source Files

- `python/sglang/srt/debug_utils/dumper.py`
- `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`
- `python/sglang/srt/layers/quantization/fp8.py`
- `python/sglang/srt/layers/quantization/fp8_utils.py`
- `python/sglang/srt/layers/quantization/fpgemm_fp8.py`
- `python/sglang/srt/layers/quantization/mxfp4_tensor.py`
- `python/sglang/srt/model_loader/utils.py`
- `python/sglang/srt/models/bailing_moe_linear.py`
- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- `python/sglang/srt/models/deepseek_common/utils.py`
- `python/sglang/srt/models/longcat_flash.py`
- `python/sglang/srt/models/longcat_flash_nextn.py`
- `python/sglang/srt/utils/common.py`
- `python/sglang/srt/utils/weight_checker.py`

## Audit Trail

- EXTRACTED: 463 (66%)
- INFERRED: 237 (34%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*