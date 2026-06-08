# Community 97

> 65 nodes

## Key Concepts

- **DeepEPDispatcher** (19 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **_DeepEPDispatcherImplBase** (17 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **Tensor** (14 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **_DeepEPDispatcherImplNormal** (12 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **_DeepEPDispatcherImplLowLatency** (12 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **.__init__()** (9 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **._get_buffer()** (8 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **.__init__()** (7 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **.get_deepep_buffer()** (7 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **._dispatch_core()** (7 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **.dispatch()** (7 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **_deepep_precompile_tp_barrier()** (6 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **TopKOutput** (6 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **.set_deepep_dispatcher_dtype()** (6 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **._combine_core()** (6 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **._dispatch_core()** (6 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **._combine_core()** (6 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **.dispatch_a()** (6 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **.combine()** (6 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **._get_impl()** (6 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **DeepEPMode** (5 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **.__init__()** (5 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **.dispatch_a()** (5 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **.combine_a()** (5 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- **._update_stage()** (5 connections) — `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- *... and 40 more nodes in this community*

## Relationships

- [[Batch-Overlap Operations]] (16 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (8 shared connections)
- [[Breakable CUDA Graph (TBO)]] (5 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (3 shared connections)
- [[Community 85]] (3 shared connections)
- [[DeepSeek MLA Attention & MoE]] (2 shared connections)
- [[Context-Parallel Attention]] (2 shared connections)
- [[Community 210]] (1 shared connections)
- [[NCCL Symmetric Memory]] (1 shared connections)
- [[Community 47]] (1 shared connections)
- [[Community 48]] (1 shared connections)

## Source Files

- `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`

## Audit Trail

- EXTRACTED: 266 (90%)
- INFERRED: 29 (10%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*