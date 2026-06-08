# Community 251

> 25 nodes

## Key Concepts

- **TritonLoRABackend** (16 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **LoRABatchInfo** (9 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **Tensor** (7 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **._prepare_lm_head_batch_info()** (7 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **._sgemm_info()** (6 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **.prepare_lora_batch()** (5 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **get_lm_head_pruned_lens()** (5 connections) — `python/sglang/srt/lora/utils.py`
- **device** (4 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **.run_lora_a_embedding()** (4 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **.run_lora_a_sgemm()** (4 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **.run_lora_b_sgemm()** (4 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **.compute_sgemm_routing()** (4 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **ForwardBatch** (4 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **merge_and_chunk_segments()** (4 connections) — `python/sglang/srt/lora/utils.py`
- **.run_qkv_lora()** (3 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **.run_gate_up_lora()** (3 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **.init_cuda_graph_batch_info()** (3 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **._build_lm_head_batch_info()** (3 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **.__init__()** (2 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **triton_backend.py** (1 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **Run LoRA A embedding lookup using Triton kernel.** (1 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **Return the sgemm batch_info (merged segments when available).** (1 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **Sort tokens by adapter and build merged segments for sgemm LoRA.** (1 connections) — `python/sglang/srt/lora/backend/triton_backend.py`
- **Compute per-sequence pruned lengths for lm_head LoRA.      Returns a list of pru** (1 connections) — `python/sglang/srt/lora/utils.py`
- **Merge consecutive same-adapter sequences and chunk at chunk_size boundaries.** (1 connections) — `python/sglang/srt/lora/utils.py`

## Relationships

- [[Community 111]] (5 shared connections)
- [[Vision-Language Model Configs]] (5 shared connections)
- [[Community 389]] (3 shared connections)
- [[Community 129]] (2 shared connections)
- [[Community 445]] (1 shared connections)
- [[Community 1652]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/backend/triton_backend.py`
- `python/sglang/srt/lora/utils.py`

## Audit Trail

- EXTRACTED: 86 (83%)
- INFERRED: 17 (17%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*