# Batch-Overlap Operations

> 187 nodes

## Key Concepts

- **CombineOverlapArgs** (100 connections) — `python/sglang/srt/batch_overlap/single_batch_overlap.py`
- **CommunicateContext** (46 connections) — `python/sglang/srt/layers/communicator.py`
- **CommunicateSummableTensorPairFn** (36 connections) — `python/sglang/srt/layers/communicator.py`
- **OperationsStrategy** (29 connections) — `python/sglang/srt/batch_overlap/operations_strategy.py`
- **TboForwardBatchPreparer** (29 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **two_batch_overlap.py** (25 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **TboDPAttentionPreparer** (24 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **ForwardBatch** (23 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **Tensor** (18 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **BaseDispatcher** (18 connections) — `python/sglang/srt/layers/moe/token_dispatcher/base.py`
- **ForwardMode** (17 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **base.py** (17 connections) — `python/sglang/srt/layers/moe/token_dispatcher/base.py`
- **SpecInput** (14 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **ScatterMode** (14 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **BumpAllocator** (14 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **operations.py** (13 connections) — `python/sglang/srt/batch_overlap/operations.py`
- **_StateDict** (13 connections) — `python/sglang/srt/batch_overlap/operations.py`
- **operations_strategy.py** (13 connections) — `python/sglang/srt/batch_overlap/operations_strategy.py`
- **EagleVerifyInput** (13 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **model_forward_maybe_tbo()** (13 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **OperationsStrategy** (13 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **ScheduleBatch** (12 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **DispatchOutput** (12 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **CombineOverlapArgs** (12 connections) — `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- **TypeGuard** (12 connections) — `python/sglang/srt/layers/moe/token_dispatcher/base.py`
- *... and 162 more nodes in this community*

## Relationships

- [[Community 37]] (39 shared connections)
- [[Aiter Attention Backend]] (30 shared connections)
- [[Community 85]] (25 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (24 shared connections)
- [[Vision-Language Model Configs]] (22 shared connections)
- [[CLI Arg Parsing & Deprecation]] (22 shared connections)
- [[Hybrid Attention Backend]] (20 shared connections)
- [[NCCL Symmetric Memory]] (18 shared connections)
- [[Community 97]] (16 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (14 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (14 shared connections)
- [[Breakable CUDA Graph (TBO)]] (9 shared connections)

## Source Files

- `python/sglang/srt/batch_overlap/operations.py`
- `python/sglang/srt/batch_overlap/operations_strategy.py`
- `python/sglang/srt/batch_overlap/single_batch_overlap.py`
- `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- `python/sglang/srt/layers/communicator.py`
- `python/sglang/srt/layers/moe/token_dispatcher/base.py`
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- `python/sglang/srt/layers/moe/utils.py`

## Audit Trail

- EXTRACTED: 782 (65%)
- INFERRED: 421 (35%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*