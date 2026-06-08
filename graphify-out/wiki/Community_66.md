# Community 66

> 93 nodes

## Key Concepts

- **AscendAttnBackend** (36 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **Tensor** (24 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **ForwardBatch** (23 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **AscendAttnMultiStepDraftBackend** (20 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **AscendTorchNativeAttnBackend** (19 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py`
- **AscendAttnMaskBuilder** (18 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **RadixAttention** (16 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **NPUFusedMLAPreprocess** (14 connections) — `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`
- **ForwardMetadata** (12 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **ModelRunner** (12 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **dtype** (11 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **SpecInput** (11 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **ForwardMode** (11 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **.forward_extend()** (11 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **.forward_decode()** (11 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **is_fia_nz()** (9 connections) — `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`
- **.forward_sparse()** (8 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **.__init__()** (7 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **.forward_mtp()** (7 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **.forward_decode_graph()** (7 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **mla_preprocess.py** (7 connections) — `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`
- **is_mla_preprocess_enabled()** (7 connections) — `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`
- **ascend_backend.py** (6 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **_reshape_kv_for_fia_nz()** (6 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- **.__init__()** (6 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- *... and 68 more nodes in this community*

## Relationships

- [[Aiter Attention Backend]] (40 shared connections)
- [[Vision-Language Model Configs]] (13 shared connections)
- [[Hybrid Attention Backend]] (11 shared connections)
- [[Grammar Manager & HiCache Clear]] (11 shared connections)
- [[Model Configs & Pooler]] (11 shared connections)
- [[DeepSeek MLA Attention & MoE]] (11 shared connections)
- [[Community 49]] (7 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (4 shared connections)
- [[Context-Parallel Attention]] (3 shared connections)
- [[Community 45]] (3 shared connections)
- [[Community 51]] (2 shared connections)
- [[Mamba2 / Hybrid Linear Attention]] (1 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- `python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py`
- `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`
- `python/sglang/srt/layers/utils/cp_utils.py`
- `python/sglang/srt/speculative/draft_utils.py`

## Audit Trail

- EXTRACTED: 357 (70%)
- INFERRED: 152 (30%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*