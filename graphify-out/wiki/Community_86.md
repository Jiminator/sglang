# Community 86

> 73 nodes

## Key Concepts

- **MambaAttnBackendBase** (74 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **ForwardMetadata** (32 connections) — `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`
- **Mamba2Metadata** (29 connections) — `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`
- **ForwardBatch** (28 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **Tensor** (25 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **ForwardMode** (15 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **ForwardMetadata** (15 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **RadixAttention** (15 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **EagleDraftInput** (14 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **EagleVerifyInput** (14 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **ModelRunner** (13 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **SpecInput** (12 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **._replay_metadata()** (8 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **.forward()** (8 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **._forward_metadata()** (7 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **._capture_metadata()** (7 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **.forward()** (7 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **.prepare_mixed()** (7 connections) — `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`
- **hybrid_linear_attn_backend.py** (6 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **._init_track_conv_indices()** (6 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **._init_track_ssm_indices()** (6 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **.init_forward_metadata_capture_cpu_graph()** (6 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **._track_mamba_state_extend()** (6 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **.forward_decode()** (6 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- **.forward_extend()** (6 connections) — `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- *... and 48 more nodes in this community*

## Relationships

- [[Aiter Attention Backend]] (58 shared connections)
- [[Mamba2 / Hybrid Linear Attention]] (32 shared connections)
- [[Vision-Language Model Configs]] (20 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (20 shared connections)
- [[Model Configs & Pooler]] (14 shared connections)
- [[Community 36]] (14 shared connections)
- [[DeepSeek MLA Attention & MoE]] (10 shared connections)
- [[HiCache Controller & Radix Tree]] (10 shared connections)
- [[Community 125]] (8 shared connections)
- [[Context-Parallel Attention]] (4 shared connections)
- [[Community 249]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`
- `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py`

## Audit Trail

- EXTRACTED: 276 (57%)
- INFERRED: 211 (43%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*