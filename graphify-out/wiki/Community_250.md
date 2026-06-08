# Community 250

> 25 nodes

## Key Concepts

- **TorchNativeLoRABackend** (13 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **lora_registry.py** (7 connections) — `python/sglang/srt/lora/backend/lora_registry.py`
- **Tensor** (7 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **TorchNativeLoRABatchInfo** (6 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **get_backend_from_name()** (4 connections) — `python/sglang/srt/lora/backend/lora_registry.py`
- **BaseLoRABackend** (4 connections) — `python/sglang/srt/lora/backend/lora_registry.py`
- **device** (4 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **.prepare_lora_batch()** (4 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **.init_cuda_graph_batch_info()** (3 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **ForwardBatch** (3 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **torch_backend.py** (2 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **.__init__()** (2 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **.run_lora_a_embedding()** (2 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **.run_lora_a_sgemm()** (2 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **.run_lora_b_sgemm()** (2 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **.run_qkv_lora()** (2 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **.run_gate_up_lora()** (2 connections) — `python/sglang/srt/lora/backend/torch_backend.py`
- **register_lora_backend()** (1 connections) — `python/sglang/srt/lora/backend/lora_registry.py`
- **create_triton_backend()** (1 connections) — `python/sglang/srt/lora/backend/lora_registry.py`
- **create_triton_csgmv_backend()** (1 connections) — `python/sglang/srt/lora/backend/lora_registry.py`
- **create_ascend_backend()** (1 connections) — `python/sglang/srt/lora/backend/lora_registry.py`
- **create_torch_native_backend()** (1 connections) — `python/sglang/srt/lora/backend/lora_registry.py`
- **create_flashinfer_backend()** (1 connections) — `python/sglang/srt/lora/backend/lora_registry.py`
- **Get corresponding backend class from backend's name** (1 connections) — `python/sglang/srt/lora/backend/lora_registry.py`
- **LoRABatchInfo** (1 connections)

## Relationships

- [[Community 111]] (6 shared connections)
- [[Vision-Language Model Configs]] (5 shared connections)
- [[Community 116]] (1 shared connections)
- [[Community 129]] (1 shared connections)
- [[Community 445]] (1 shared connections)
- [[Community 389]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/backend/lora_registry.py`
- `python/sglang/srt/lora/backend/torch_backend.py`

## Audit Trail

- EXTRACTED: 61 (79%)
- INFERRED: 16 (21%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*