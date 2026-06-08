# Breakable CUDA Graph (TBO)

> 242 nodes

## Key Concepts

- **GroupCoordinator** (125 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **parallel_state.py** (59 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **TboCudaGraphRunnerPlugin** (36 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **LogitsProcessorOutput** (28 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **NgramEmbeddingInfo** (27 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **BreakableCUDAGraphCapture** (26 connections) — `python/sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py`
- **BreakableCUDAGraph** (25 connections) — `python/sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py`
- **.__init__()** (24 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **.__init__()** (24 connections) — `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
- **ForwardBatch** (23 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **DeepEPBuffer** (22 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **PPProxyTensors** (21 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **.init_torch_distributed()** (21 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **get_world_group()** (20 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **SimpleNamespace** (20 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **Module** (20 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **ModelRunner** (20 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **device** (19 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **dtype** (19 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **Tensor** (19 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **GroupCoordinator** (19 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **cuda_graph_runner.py** (17 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **._dummy_run()** (17 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **ForwardBatch** (15 connections) — `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
- **require_gathered_buffer()** (15 connections) — `python/sglang/srt/utils/common.py`
- *... and 217 more nodes in this community*

## Relationships

- [[Multi-Step Draft Attention (FP8)]] (145 shared connections)
- [[Hybrid Attention Backend]] (109 shared connections)
- [[Model Configs & Pooler]] (35 shared connections)
- [[Disaggregation Bootstrap & Decode]] (30 shared connections)
- [[Context-Parallel Attention]] (28 shared connections)
- [[Vision-Language Model Configs]] (27 shared connections)
- [[DeepSeek MLA Attention & MoE]] (25 shared connections)
- [[Aiter Attention Backend]] (24 shared connections)
- [[Community 101]] (23 shared connections)
- [[Community 240]] (23 shared connections)
- [[NCCL Symmetric Memory]] (17 shared connections)
- [[CLI Arg Parsing & Deprecation]] (14 shared connections)

## Source Files

- `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- `python/sglang/srt/compilation/piecewise_context_manager.py`
- `python/sglang/srt/disaggregation/common/conn.py`
- `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`
- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/elastic_ep/expert_backup_client.py`
- `python/sglang/srt/hardware_backend/musa/utils/patch_torch.py`
- `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- `python/sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py`
- `python/sglang/srt/model_executor/breakable_cuda_graph/context.py`
- `python/sglang/srt/model_executor/breakable_cuda_graph/cuda_utils.py`
- `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py`
- `python/sglang/srt/model_executor/cpu_graph_runner.py`
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
- `python/sglang/srt/multiplex/pdmux_context.py`
- `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`

## Audit Trail

- EXTRACTED: 761 (50%)
- INFERRED: 750 (50%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*