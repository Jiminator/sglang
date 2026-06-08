# Community 48

> 132 nodes

## Key Concepts

- **_()** (40 connections) — `python/sglang/srt/layers/quantization/fp8_kernel.py`
- **UnquantizedEmbeddingMethod** (30 connections) — `python/sglang/srt/layers/quantization/unquant.py`
- **utils.py** (25 connections) — `python/sglang/srt/layers/quantization/utils.py`
- **ceil_align()** (23 connections) — `python/sglang/srt/utils/common.py`
- **Tensor** (22 connections) — `python/sglang/srt/layers/quantization/fp8_kernel.py`
- **Tensor** (22 connections) — `python/sglang/srt/layers/quantization/utils.py`
- **scaled_fp8_quant()** (20 connections) — `python/sglang/srt/layers/quantization/fp8_kernel.py`
- **VocabParallelEmbeddingShardIndices** (16 connections) — `python/sglang/srt/layers/vocab_parallel_embedding.py`
- **.__init__()** (13 connections) — `python/sglang/srt/layers/vocab_parallel_embedding.py`
- **.process_weights_after_loading()** (12 connections) — `python/sglang/srt/layers/quantization/fp8.py`
- **constexpr** (12 connections) — `python/sglang/srt/layers/quantization/fp8_kernel.py`
- **dtype** (12 connections) — `python/sglang/srt/layers/quantization/fp8_kernel.py`
- **replace_parameter()** (12 connections) — `python/sglang/srt/layers/quantization/utils.py`
- **sglang_per_token_group_quant_fp8()** (11 connections) — `python/sglang/srt/layers/quantization/fp8_kernel.py`
- **print_warning_once()** (10 connections) — `python/sglang/srt/utils/common.py`
- **.process_weights_after_loading()** (9 connections) — `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py`
- **w8a8_block_fp8_matmul_triton()** (9 connections) — `python/sglang/srt/layers/quantization/fp8_kernel.py`
- **requantize_with_max_scale()** (9 connections) — `python/sglang/srt/layers/quantization/utils.py`
- **get_linear_quant_method()** (9 connections) — `python/sglang/srt/layers/quantization/utils.py`
- **QuantizationConfig** (8 connections) — `python/sglang/srt/layers/quantization/utils.py`
- **vocab_parallel_embedding.py** (8 connections) — `python/sglang/srt/layers/vocab_parallel_embedding.py`
- **.process_weights_after_loading()** (7 connections) — `python/sglang/srt/layers/quantization/fp8.py`
- **_per_token_group_quant_8bit()** (7 connections) — `python/sglang/srt/layers/quantization/fp8_kernel.py`
- **_w8a8_block_fp8_matmul()** (7 connections) — `python/sglang/srt/layers/quantization/fp8_kernel.py`
- **w8a8_block_fp8_matmul_deepgemm()** (7 connections) — `python/sglang/srt/layers/quantization/fp8_kernel.py`
- *... and 107 more nodes in this community*

## Relationships

- [[Compressed-Tensors Quant Linear]] (44 shared connections)
- [[DeepSeek MLA Attention & MoE]] (21 shared connections)
- [[Community 45]] (17 shared connections)
- [[Weight Loading & EPLB]] (11 shared connections)
- [[Vision-Language Model Configs]] (11 shared connections)
- [[Community 88]] (9 shared connections)
- [[Context-Parallel Attention]] (7 shared connections)
- [[Linear Layer Parameters]] (7 shared connections)
- [[ROCm MoE Quantization]] (5 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (5 shared connections)
- [[NCCL Symmetric Memory]] (5 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (5 shared connections)

## Source Files

- `python/sglang/srt/entrypoints/tool.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py`
- `python/sglang/srt/layers/quantization/fp8.py`
- `python/sglang/srt/layers/quantization/fp8_kernel.py`
- `python/sglang/srt/layers/quantization/kv_cache.py`
- `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`
- `python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py`
- `python/sglang/srt/layers/quantization/unquant.py`
- `python/sglang/srt/layers/quantization/utils.py`
- `python/sglang/srt/layers/vocab_parallel_embedding.py`
- `python/sglang/srt/session/streaming_session.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 468 (68%)
- INFERRED: 223 (32%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*