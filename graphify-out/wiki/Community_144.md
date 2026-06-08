# Community 144

> 46 nodes

## Key Concepts

- **Tensor** (20 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **TritonMultiStepDraftBackend** (19 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **ForwardBatch** (17 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **ForwardMetadata** (12 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **SpecInput** (12 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **ForwardMode** (11 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **RadixAttention** (11 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **triton_backend.py** (10 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **ModelRunner** (10 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **._apply_cuda_graph_metadata()** (10 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **.__init__()** (9 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **._update_draft_extend_buffers()** (7 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **.init_forward_metadata()** (7 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **._forward_extend_unified()** (7 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **._fill_kv_indptr_and_indices()** (6 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **._update_decode_kv_buffers()** (6 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **._update_target_verify_buffers()** (6 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **._build_cuda_graph_forward_metadata()** (6 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **.forward_extend()** (6 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **.common_template()** (6 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **.get_num_kv_splits()** (5 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **.init_forward_metadata_out_graph()** (5 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **.forward_decode()** (5 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **.__init__()** (5 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **.init_forward_metadata_out_graph()** (5 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- *... and 21 more nodes in this community*

## Relationships

- [[Aiter Attention Backend]] (37 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (19 shared connections)
- [[Hybrid Attention Backend]] (8 shared connections)
- [[Model Configs & Pooler]] (8 shared connections)
- [[DeepSeek MLA Attention & MoE]] (8 shared connections)
- [[Vision-Language Model Configs]] (8 shared connections)
- [[Context-Parallel Attention]] (2 shared connections)
- [[Community 45]] (1 shared connections)
- [[Community 148]] (1 shared connections)
- [[Community 85]] (1 shared connections)
- [[Community 47]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/triton_backend.py`
- `python/sglang/srt/speculative/draft_utils.py`

## Audit Trail

- EXTRACTED: 181 (70%)
- INFERRED: 77 (30%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*