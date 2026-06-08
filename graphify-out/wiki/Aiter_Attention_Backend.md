# Aiter Attention Backend

> 501 nodes

## Key Concepts

- **ForwardMode** (610 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **AttentionBackend** (273 connections) — `python/sglang/srt/layers/attention/base_attn_backend.py`
- **SpecInput** (259 connections) — `python/sglang/srt/speculative/spec_info.py`
- **SWAKVPool** (137 connections) — `python/sglang/srt/mem_cache/swa_memory_pool.py`
- **FlashInferMLAAttnBackend** (46 connections) — `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`
- **FlashInferAttnBackend** (40 connections) — `python/sglang/srt/layers/attention/flashinfer_backend.py`
- **AiterAttnBackend** (38 connections) — `python/sglang/srt/layers/attention/aiter_backend.py`
- **FlashAttentionBackend** (37 connections) — `python/sglang/srt/layers/attention/flashattention_backend.py`
- **FlashAttentionMetadata** (29 connections) — `python/sglang/srt/layers/attention/flashattention_backend.py`
- **FlashInferMultiStepDraftBackend** (29 connections) — `python/sglang/srt/layers/attention/flashinfer_backend.py`
- **ServerArgs** (29 connections) — `python/sglang/srt/speculative/draft_utils.py`
- **AttentionBackend** (28 connections)
- **FlashInferMLAMultiStepDraftBackend** (28 connections) — `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`
- **Tensor** (26 connections) — `python/sglang/srt/layers/attention/aiter_backend.py`
- **FlashInferMlaAttnBackend** (25 connections) — `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`
- **AscendMambaAttnBackendBase** (24 connections) — `python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py`
- **Tensor** (24 connections) — `python/sglang/srt/layers/attention/flashinfer_backend.py`
- **FlashAttentionMultiStepBackend** (23 connections) — `python/sglang/srt/layers/attention/flashattention_backend.py`
- **Tensor** (19 connections) — `python/sglang/srt/layers/attention/trtllm_mha_backend.py`
- **SpecInput** (18 connections) — `python/sglang/srt/layers/attention/flashinfer_backend.py`
- **Tensor** (18 connections) — `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`
- **WaveAttnBackend** (18 connections) — `python/sglang/srt/layers/attention/wave_backend.py`
- **XPUAttentionBackend** (18 connections) — `python/sglang/srt/layers/attention/xpu_backend.py`
- **MusaFlashAttentionBackend** (17 connections) — `python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py`
- **.init_forward_metadata()** (17 connections) — `python/sglang/srt/layers/attention/aiter_backend.py`
- *... and 476 more nodes in this community*

## Relationships

- [[Multi-Step Draft Attention (FP8)]] (182 shared connections)
- [[Vision-Language Model Configs]] (143 shared connections)
- [[DeepSeek MLA Attention & MoE]] (128 shared connections)
- [[Grammar Manager & HiCache Clear]] (76 shared connections)
- [[Hybrid Attention Backend]] (70 shared connections)
- [[CLI Arg Parsing & Deprecation]] (61 shared connections)
- [[Community 86]] (58 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (57 shared connections)
- [[Community 43]] (51 shared connections)
- [[Community 67]] (44 shared connections)
- [[Community 66]] (40 shared connections)
- [[Model Configs & Pooler]] (38 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/musa/attention/__init__.py`
- `python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py`
- `python/sglang/srt/hardware_backend/musa/layers/utils/cp_utils.py`
- `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- `python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py`
- `python/sglang/srt/layers/attention/aiter_backend.py`
- `python/sglang/srt/layers/attention/attention_registry.py`
- `python/sglang/srt/layers/attention/base_attn_backend.py`
- `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- `python/sglang/srt/layers/attention/flashattention_backend.py`
- `python/sglang/srt/layers/attention/flashinfer_backend.py`
- `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`
- `python/sglang/srt/layers/attention/flashmla_backend.py`
- `python/sglang/srt/layers/attention/hybrid_attn_backend.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `python/sglang/srt/layers/attention/tbo_backend.py`
- `python/sglang/srt/layers/attention/triton_backend.py`
- `python/sglang/srt/layers/attention/triton_ops/metadata.py`
- `python/sglang/srt/layers/attention/trtllm_mha_backend.py`
- `python/sglang/srt/layers/attention/trtllm_mla_backend.py`

## Audit Trail

- EXTRACTED: 1806 (43%)
- INFERRED: 2352 (57%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*