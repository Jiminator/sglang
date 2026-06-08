# Community 36

> 161 nodes

## Key Concepts

- **GDNKernelDispatcher** (24 connections) — `python/sglang/srt/layers/attention/linear/gdn_backend.py`
- **AscendGDNAttnBackend** (16 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py`
- **CuteDSLGDNKernel** (16 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py`
- **LinearAttnKernelBase** (16 connections) — `python/sglang/srt/layers/attention/linear/kernels/kernel_backend.py`
- **FlashInferGDNKernel** (15 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`
- **TritonGDNKernel** (15 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`
- **GDNAttnBackend** (14 connections) — `python/sglang/srt/layers/attention/linear/gdn_backend.py`
- **CuteDSLKDAKernel** (14 connections) — `python/sglang/srt/layers/attention/linear/kernels/kda_cutedsl.py`
- **TritonKDAKernel** (14 connections) — `python/sglang/srt/layers/attention/linear/kernels/kda_triton.py`
- **Tensor** (13 connections) — `python/sglang/srt/layers/attention/linear/gdn_backend.py`
- **ForwardBatch** (12 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py`
- **attn_backend_wrapper()** (12 connections) — `python/sglang/srt/layers/attention/attention_registry.py`
- **fused_recurrent.py** (12 connections) — `python/sglang/srt/layers/attention/fla/fused_recurrent.py`
- **KDAKernelDispatcher** (12 connections) — `python/sglang/srt/layers/attention/linear/kda_backend.py`
- **Tensor** (11 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py`
- **KDAAttnBackend** (11 connections) — `python/sglang/srt/layers/attention/linear/kda_backend.py`
- **LinearAttnKernelBackend** (11 connections) — `python/sglang/srt/layers/attention/linear/utils.py`
- **RadixLinearAttention** (10 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py`
- **ForwardBatch** (10 connections) — `python/sglang/srt/layers/attention/linear/gdn_backend.py`
- **Tensor** (10 connections) — `python/sglang/srt/layers/attention/linear/kda_backend.py`
- **ModelRunner** (9 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py`
- **ForwardMode** (9 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py`
- **EagleDraftInput** (9 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py`
- **EagleVerifyInput** (9 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py`
- **RadixLinearAttention** (9 connections) — `python/sglang/srt/layers/attention/linear/gdn_backend.py`
- *... and 136 more nodes in this community*

## Relationships

- [[Qwen3 / Kimi Model Configs]] (22 shared connections)
- [[Vision-Language Model Configs]] (22 shared connections)
- [[Aiter Attention Backend]] (16 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (16 shared connections)
- [[Disaggregation Utils & Cache Tests]] (15 shared connections)
- [[Community 86]] (14 shared connections)
- [[Community 35]] (3 shared connections)
- [[Community 47]] (3 shared connections)
- [[Hybrid Attention Backend]] (2 shared connections)
- [[Mamba2 / Hybrid Linear Attention]] (2 shared connections)
- [[Aibrix KV Cache Storage]] (2 shared connections)
- [[Community 107]] (2 shared connections)

## Source Files

- `python/sglang/srt/configs/linear_attn_model_registry.py`
- `python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py`
- `python/sglang/srt/layers/attention/attention_registry.py`
- `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`
- `python/sglang/srt/layers/attention/fla/fused_recurrent.py`
- `python/sglang/srt/layers/attention/fla/utils.py`
- `python/sglang/srt/layers/attention/linear/gdn_backend.py`
- `python/sglang/srt/layers/attention/linear/kda_backend.py`
- `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py`
- `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`
- `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`
- `python/sglang/srt/layers/attention/linear/kernels/kda_cutedsl.py`
- `python/sglang/srt/layers/attention/linear/kernels/kda_triton.py`
- `python/sglang/srt/layers/attention/linear/kernels/kernel_backend.py`
- `python/sglang/srt/layers/attention/linear/utils.py`

## Audit Trail

- EXTRACTED: 432 (61%)
- INFERRED: 272 (39%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*