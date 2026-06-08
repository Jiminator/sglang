# MoE Dispatch/Combine (Cutlass)

> 316 nodes

## Key Concepts

- **StandardCombineInput** (94 connections) — `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
- **DeepEPLLCombineInput** (67 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **MoeRunner** (58 connections) — `python/sglang/srt/layers/moe/moe_runner/runner.py`
- **StandardDispatchOutput** (57 connections) — `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
- **DeepEPLLDispatchOutput** (49 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **DeepGemmMoeQuantInfo** (46 connections) — `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`
- **DeepEPNormalCombineInput** (45 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **NamedTuple** (38 connections)
- **DeepEPNormalDispatchOutput** (37 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **TritonRunnerCore** (29 connections) — `python/sglang/srt/layers/moe/moe_runner/triton.py`
- **MoriEPNormalDispatchOutput** (24 connections) — `python/sglang/srt/layers/moe/token_dispatcher/moriep.py`
- **MoriEPLLDispatchOutput** (24 connections) — `python/sglang/srt/layers/moe/token_dispatcher/moriep.py`
- **DeepGemmRunnerCore** (23 connections) — `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`
- **MoriEPNormalCombineInput** (23 connections) — `python/sglang/srt/layers/moe/token_dispatcher/moriep.py`
- **MoriEPLLCombineInput** (23 connections) — `python/sglang/srt/layers/moe/token_dispatcher/moriep.py`
- **AiterRunnerCore** (22 connections) — `python/sglang/srt/layers/moe/moe_runner/aiter.py`
- **TritonRunnerInput** (22 connections) — `python/sglang/srt/layers/moe/moe_runner/triton.py`
- **TritonRunnerOutput** (22 connections) — `python/sglang/srt/layers/moe/moe_runner/triton.py`
- **AiterRunnerOutput** (18 connections) — `python/sglang/srt/layers/moe/moe_runner/aiter.py`
- **aiter.py** (17 connections) — `python/sglang/srt/layers/moe/moe_runner/aiter.py`
- **DeepGemmRunnerInput** (17 connections) — `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`
- **FlashinferDispatchOutput** (17 connections) — `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`
- **FlashinferCombineInput** (17 connections) — `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`
- **NixlEPDispatcher** (17 connections) — `python/sglang/srt/layers/moe/token_dispatcher/nixl.py`
- **AiterRunnerInput** (16 connections) — `python/sglang/srt/layers/moe/moe_runner/aiter.py`
- *... and 291 more nodes in this community*

## Relationships

- [[Compressed-Tensors Quant Linear]] (101 shared connections)
- [[Weight Loading & EPLB]] (42 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (34 shared connections)
- [[DeepSeek MLA Attention & MoE]] (28 shared connections)
- [[Community 85]] (14 shared connections)
- [[Batch-Overlap Operations]] (14 shared connections)
- [[Hybrid Attention Backend]] (13 shared connections)
- [[NCCL Symmetric Memory]] (12 shared connections)
- [[Vision-Language Model Configs]] (8 shared connections)
- [[Community 97]] (8 shared connections)
- [[Community 80]] (7 shared connections)
- [[Aibrix KV Cache Storage]] (6 shared connections)

## Source Files

- `python/sglang/srt/layers/moe/ep_moe/layer.py`
- `python/sglang/srt/layers/moe/moe_runner/aiter.py`
- `python/sglang/srt/layers/moe/moe_runner/base.py`
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py`
- `python/sglang/srt/layers/moe/moe_runner/marlin.py`
- `python/sglang/srt/layers/moe/moe_runner/runner.py`
- `python/sglang/srt/layers/moe/moe_runner/triton.py`
- `python/sglang/srt/layers/moe/moe_runner/triton_kernels.py`
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`
- `python/sglang/srt/layers/moe/token_dispatcher/moriep.py`
- `python/sglang/srt/layers/moe/token_dispatcher/nixl.py`
- `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
- `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py`
- `python/sglang/srt/lora/lora_moe_runner_marlin.py`
- `python/sglang/srt/lora/lora_moe_runners.py`
- `python/sglang/srt/lora/trtllm_lora_temp/sgl_fp8_moe.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 1055 (48%)
- INFERRED: 1159 (52%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*