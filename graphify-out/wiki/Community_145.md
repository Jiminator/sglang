# Community 145

> 46 nodes

## Key Concepts

- **GraphSlot** (12 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **build_prefill_registry()** (11 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **build_decode_registry()** (10 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **cuda_graph_buffer_registry.py** (9 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **.fill_from()** (8 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **.register_slot()** (7 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **.slice_for()** (6 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **share_input_buffer()** (6 connections) — `python/sglang/srt/model_executor/input_buffers.py`
- **._init_buffers()** (5 connections) — `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py`
- **_grouped_foreach_copy_()** (4 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **Tensor** (4 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **PaddingPolicy** (4 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **FillContext** (4 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **.reset_padding()** (4 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **device** (4 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **Any** (4 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **share_input_buffers_in()** (4 connections) — `python/sglang/srt/model_executor/input_buffers.py`
- **._share_one_buffer()** (4 connections) — `python/sglang/srt/model_executor/input_buffers.py`
- **get_draft_hidden_dim()** (4 connections) — `python/sglang/srt/speculative/eagle_utils.py`
- **._padded_n()** (3 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **.extract_buffer()** (3 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **dtype** (3 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- **input_buffers.py** (3 connections) — `python/sglang/srt/model_executor/input_buffers.py`
- **.share_buffers()** (3 connections) — `python/sglang/srt/model_executor/input_buffers.py`
- **._raw_n()** (2 connections) — `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- *... and 21 more nodes in this community*

## Relationships

- [[Hybrid Attention Backend]] (11 shared connections)
- [[Vision-Language Model Configs]] (7 shared connections)
- [[Breakable CUDA Graph (TBO)]] (4 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (4 shared connections)
- [[Community 107]] (2 shared connections)
- [[Community 47]] (2 shared connections)
- [[Community 382]] (1 shared connections)
- [[CLI Arg Parsing & Deprecation]] (1 shared connections)

## Source Files

- `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py`
- `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- `python/sglang/srt/model_executor/input_buffers.py`
- `python/sglang/srt/speculative/eagle_utils.py`

## Audit Trail

- EXTRACTED: 134 (86%)
- INFERRED: 22 (14%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*