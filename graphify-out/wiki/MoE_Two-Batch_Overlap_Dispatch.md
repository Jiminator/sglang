# MoE Two-Batch Overlap Dispatch

> 199 nodes

## Key Concepts

- **get_moe_a2a_backend()** (70 connections) — `python/sglang/srt/layers/moe/utils.py`
- **MaybeTboDeepEPDispatcher** (60 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **get_moe_runner_backend()** (53 connections) — `python/sglang/srt/layers/moe/utils.py`
- **FusedMoE** (37 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- **DownGemmOverlapArgs** (32 connections) — `python/sglang/srt/batch_overlap/single_batch_overlap.py`
- **utils.py** (27 connections) — `python/sglang/srt/layers/moe/utils.py`
- **Tensor** (26 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- **FlashinferDispatcher** (22 connections) — `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`
- **TorchDistributedCommBackend** (20 connections) — `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`
- **StandardDispatcher** (19 connections) — `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
- **MoeRunnerBackend** (19 connections) — `python/sglang/srt/layers/moe/utils.py`
- **.__init__()** (18 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- **MoeA2ABackend** (18 connections) — `python/sglang/srt/layers/moe/utils.py`
- **should_use_flashinfer_cutlass_moe_fp4_allgather()** (18 connections) — `python/sglang/srt/layers/moe/utils.py`
- **Parameter** (17 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- **is_deepep_class_backend()** (14 connections) — `python/sglang/srt/layers/moe/utils.py`
- **.process_weights_after_loading_block_quant()** (13 connections) — `python/sglang/srt/layers/quantization/fp8.py`
- **get_moe_expert_parallel_rank()** (12 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **MoeRunnerConfig** (12 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- **FusedMoeWeightScaleSupported** (12 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- **._weight_loader_impl()** (12 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- **TopKOutput** (12 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- **should_use_dp_reduce_scatterv()** (12 connections) — `python/sglang/srt/layers/moe/utils.py`
- **layer.py** (11 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- **BaseDispatcher** (11 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- *... and 174 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (70 shared connections)
- [[Compressed-Tensors Quant Linear]] (70 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (34 shared connections)
- [[Context-Parallel Attention]] (31 shared connections)
- [[Weight Loading & EPLB]] (30 shared connections)
- [[NCCL Symmetric Memory]] (19 shared connections)
- [[Vision-Language Model Configs]] (15 shared connections)
- [[Batch-Overlap Operations]] (14 shared connections)
- [[CLI Arg Parsing & Deprecation]] (13 shared connections)
- [[Qwen3 / Kimi Model Configs]] (9 shared connections)
- [[Breakable CUDA Graph (TBO)]] (7 shared connections)
- [[Community 107]] (7 shared connections)

## Source Files

- `python/sglang/srt/batch_overlap/single_batch_overlap.py`
- `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`
- `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`
- `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
- `python/sglang/srt/layers/moe/utils.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py`
- `python/sglang/srt/layers/quantization/fp8.py`
- `python/sglang/srt/layers/quantization/modelopt_quant.py`
- `python/sglang/srt/layers/quantization/mxfp4.py`
- `python/sglang/srt/layers/quantization/unquant.py`
- `python/sglang/srt/layers/quantization/utils.py`
- `python/sglang/srt/layers/utils/common.py`
- `python/sglang/srt/models/deepseek_v2.py`
- `python/sglang/srt/models/glm4_moe_lite.py`

## Audit Trail

- EXTRACTED: 609 (54%)
- INFERRED: 522 (46%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*