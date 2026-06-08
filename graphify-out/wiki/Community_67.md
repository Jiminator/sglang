# Community 67

> 90 nodes

## Key Concepts

- **attention_registry.py** (24 connections) — `python/sglang/srt/layers/attention/attention_registry.py`
- **DualChunkFlashAttentionBackend** (24 connections) — `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- **Tensor** (18 connections) — `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- **TorchNativeAttnBackend** (15 connections) — `python/sglang/srt/layers/attention/torch_native_backend.py`
- **IntelAMXAttnBackend** (14 connections) — `python/sglang/srt/layers/attention/intel_amx_backend.py`
- **flash_attn_varlen_func()** (14 connections) — `python/sglang/srt/models/mimo_audio.py`
- **DualChunkFlashAttentionMetadata** (10 connections) — `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- **ForwardBatch** (9 connections) — `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- **dual_chunk_flashattention_backend.py** (7 connections) — `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- **ForwardMode** (7 connections) — `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- **._dual_chunk_flash_attn_prefill_func()** (7 connections) — `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- **Tensor** (7 connections) — `python/sglang/srt/layers/attention/torch_native_backend.py`
- **ForwardBatch** (7 connections) — `python/sglang/srt/layers/attention/torch_native_backend.py`
- **.forward_extend()** (6 connections) — `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- **._bind_metadata_buffers()** (6 connections) — `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- **ForwardBatch** (6 connections) — `python/sglang/srt/layers/attention/intel_amx_backend.py`
- **RadixAttention** (6 connections) — `python/sglang/srt/layers/attention/torch_native_backend.py`
- **._apply_cuda_graph_metadata()** (5 connections) — `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- **._do_flash_attn()** (5 connections) — `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- **._dual_chunk_flash_attn_decoding()** (5 connections) — `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- **Tensor** (5 connections) — `python/sglang/srt/layers/attention/intel_amx_backend.py`
- **RadixAttention** (5 connections) — `python/sglang/srt/layers/attention/intel_amx_backend.py`
- **ModelRunner** (5 connections) — `python/sglang/srt/layers/attention/torch_native_backend.py`
- **._make_sliding_window_mask()** (5 connections) — `python/sglang/srt/layers/attention/torch_native_backend.py`
- **device** (5 connections) — `python/sglang/srt/layers/attention/torch_native_backend.py`
- *... and 65 more nodes in this community*

## Relationships

- [[Aiter Attention Backend]] (44 shared connections)
- [[DeepSeek MLA Attention & MoE]] (16 shared connections)
- [[Vision-Language Model Configs]] (16 shared connections)
- [[Model Configs & Pooler]] (6 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (5 shared connections)
- [[Community 99]] (3 shared connections)
- [[Community 43]] (2 shared connections)
- [[Community 36]] (1 shared connections)
- [[Community 73]] (1 shared connections)
- [[Community 287]] (1 shared connections)
- [[Community 66]] (1 shared connections)
- [[Community 110]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/attention_registry.py`
- `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`
- `python/sglang/srt/layers/attention/intel_amx_backend.py`
- `python/sglang/srt/layers/attention/torch_native_backend.py`
- `python/sglang/srt/models/mimo_audio.py`

## Audit Trail

- EXTRACTED: 273 (73%)
- INFERRED: 102 (27%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*