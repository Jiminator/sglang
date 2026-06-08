# Community 129

> 51 nodes

## Key Concepts

- **ChunkedSgmvLoRABackend** (22 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **Tensor** (10 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **LoRABatchInfo** (9 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **__init__.py** (9 connections) — `python/sglang/srt/lora/triton_ops/__init__.py`
- **ForwardBatch** (8 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **.prepare_lora_batch()** (8 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **._prepare_lm_head_batch_info()** (8 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **chunked_sgmv_lora_expand_forward()** (8 connections) — `python/sglang/srt/lora/triton_ops/chunked_sgmv_expand.py`
- **chunked_sgmv_lora_shrink_forward()** (8 connections) — `python/sglang/srt/lora/triton_ops/chunked_sgmv_shrink.py`
- **device** (7 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **._build_req_seg_indptr()** (6 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **._get_permutation()** (6 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **chunked_embedding_lora_a_forward()** (6 connections) — `python/sglang/srt/lora/triton_ops/chunked_embedding_lora_a.py`
- **._determine_chunk_size()** (5 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **._get_segments_info()** (5 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **ServerArgs** (4 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **.run_lora_a_sgemm()** (4 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **.run_lora_b_sgemm()** (4 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **.run_qkv_lora()** (4 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **.run_gate_up_lora()** (4 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **._determine_chunk_size_for_tokens()** (4 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **.__init__()** (3 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **.run_lora_a_embedding()** (3 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **.init_cuda_graph_batch_info()** (3 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- **._build_lm_head_batch_info()** (3 connections) — `python/sglang/srt/lora/backend/chunked_backend.py`
- *... and 26 more nodes in this community*

## Relationships

- [[Community 111]] (6 shared connections)
- [[Vision-Language Model Configs]] (6 shared connections)
- [[CLI Arg Parsing & Deprecation]] (6 shared connections)
- [[Community 389]] (2 shared connections)
- [[Community 251]] (2 shared connections)
- [[Community 1652]] (2 shared connections)
- [[Community 496]] (2 shared connections)
- [[Community 445]] (1 shared connections)
- [[Community 250]] (1 shared connections)
- [[Community 466]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/backend/chunked_backend.py`
- `python/sglang/srt/lora/triton_ops/__init__.py`
- `python/sglang/srt/lora/triton_ops/chunked_embedding_lora_a.py`
- `python/sglang/srt/lora/triton_ops/chunked_sgmv_expand.py`
- `python/sglang/srt/lora/triton_ops/chunked_sgmv_shrink.py`

## Audit Trail

- EXTRACTED: 160 (80%)
- INFERRED: 39 (20%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*