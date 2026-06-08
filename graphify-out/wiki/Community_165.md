# Community 165

> 40 nodes

## Key Concepts

- **ensure_workspace_initialized()** (19 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **flashinfer_comm_fusion.py** (16 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **get_moe_tensor_parallel_world_size()** (11 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **flashinfer_allreduce_residual_rmsnorm()** (10 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **_preflight_check_workspace_memory()** (9 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **get_attn_tensor_model_parallel_world_size()** (7 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **_flashinfer_trtllm_workspace_allocation_sizes()** (6 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **dtype** (6 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **FlashInferWorkspaceManager** (6 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **.initialize()** (6 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **get_attn_tensor_model_parallel_rank()** (5 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **_FixedTorchDistBackend** (5 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **pre_initialize_workspaces()** (5 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **_get_workspace_manager()** (4 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **_sync_allreduce_unavailable_across_tp()** (4 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **_should_force_posix_fd_transport()** (3 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **_flashinfer_posix_fd_transport_override_if_needed()** (3 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **_make_flashinfer_workspace_allocation_prop()** (3 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **.is_buffer_size_sufficient()** (3 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **.cleanup()** (3 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **.__init__()** (2 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **is_flashinfer_allreduce_unavailable()** (2 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **_probe_cumem_create_sequence()** (2 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **.__init__()** (2 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- **fake_flashinfer_allreduce_residual_rmsnorm()** (2 connections) — `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- *... and 15 more nodes in this community*

## Relationships

- [[NCCL Symmetric Memory]] (8 shared connections)
- [[Context-Parallel Attention]] (6 shared connections)
- [[Breakable CUDA Graph (TBO)]] (3 shared connections)
- [[Community 117]] (3 shared connections)
- [[Community 42]] (3 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (2 shared connections)
- [[Community 161]] (1 shared connections)
- [[Llama / GPT-OSS Model Layers]] (1 shared connections)
- [[DeepSeek MLA Attention & MoE]] (1 shared connections)
- [[Community 47]] (1 shared connections)
- [[Community 48]] (1 shared connections)
- [[Qwen3 / Kimi Model Configs]] (1 shared connections)

## Source Files

- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 125 (77%)
- INFERRED: 37 (23%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*