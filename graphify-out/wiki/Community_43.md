# Community 43

> 142 nodes

## Key Concepts

- **DeepseekSparseAttnBackend** (45 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **Tensor** (41 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **DSAIndexerMetadata** (24 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **ForwardBatch** (22 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **PrecomputedMetadata** (21 connections) — `python/sglang/srt/layers/attention/dsa/dsa_backend_mtp_precompute.py`
- **DeepseekSparseAttnBackendMTPPrecomputeMixin** (21 connections) — `python/sglang/srt/layers/attention/dsa/dsa_backend_mtp_precompute.py`
- **DSATopKBackend** (21 connections) — `python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py`
- **DeepseekSparseAttnMultiStepBackend** (19 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **.init_forward_metadata()** (17 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **.forward_extend()** (17 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **TopkTransformMethod** (16 connections) — `python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py`
- **DSAMetadata** (16 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **._apply_cuda_graph_metadata()** (16 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **RadixAttention** (16 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **DSAFlashMLAMetadata** (15 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **ForwardMode** (15 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **._build_forward_metadata_cuda_graph()** (15 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **.forward_decode()** (14 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **._forward_trtllm()** (13 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **ModelRunner** (12 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **SpecInput** (12 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **PrecomputedMetadata** (11 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **.set_dsa_prefill_impl()** (11 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **TopkTransformMethod** (11 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- **dsa_backend.py** (10 connections) — `python/sglang/srt/layers/attention/dsa_backend.py`
- *... and 117 more nodes in this community*

## Relationships

- [[Aiter Attention Backend]] (51 shared connections)
- [[Community 49]] (14 shared connections)
- [[DeepSeek MLA Attention & MoE]] (13 shared connections)
- [[Vision-Language Model Configs]] (13 shared connections)
- [[Context-Parallel Attention]] (9 shared connections)
- [[Community 47]] (5 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (4 shared connections)
- [[Community 73]] (4 shared connections)
- [[Community 107]] (2 shared connections)
- [[Community 67]] (2 shared connections)
- [[Community 132]] (2 shared connections)
- [[Community 419]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/dsa/dsa_backend_mtp_precompute.py`
- `python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py`
- `python/sglang/srt/layers/attention/dsa/transform_index.py`
- `python/sglang/srt/layers/attention/dsa/triton_kernel.py`
- `python/sglang/srt/layers/attention/dsa/utils.py`
- `python/sglang/srt/layers/attention/dsa_backend.py`
- `python/sglang/srt/layers/attention/triton_ops/pad.py`
- `python/sglang/srt/layers/attention/utils.py`
- `python/sglang/srt/speculative/draft_utils.py`

## Audit Trail

- EXTRACTED: 584 (69%)
- INFERRED: 259 (31%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*