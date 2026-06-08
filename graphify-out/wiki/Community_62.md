# Community 62

> 96 nodes

## Key Concepts

- **FrozenKVMTPWorker** (69 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- **FrozenKVMTPContext** (35 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_info.py`
- **FrozenKVMTPDraftInput** (31 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_info.py`
- **FrozenKVMTPCudaGraphRunner** (30 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py`
- **FrozenKVMTPDraftExtendInput** (27 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_info.py`
- **ScheduleBatch** (26 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- **ForwardBatch** (24 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- **FrozenKVMTPVerifyInput** (22 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_info.py`
- **Tensor** (22 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- **LogitsProcessorOutput** (18 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- **FrozenKVMTPDraftInput** (18 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- **ServerArgs** (17 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- **FrozenKVMTPDraftExtendInput** (17 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- **GenerationBatchResult** (17 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- **ForwardBatch** (13 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py`
- **.forward_batch_generation()** (13 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- **ForwardBatch** (11 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_utils.py`
- **Tensor** (10 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_utils.py`
- **FrozenKVMTPContext** (9 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_utils.py`
- **ScheduleBatch** (9 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_utils.py`
- **.__init__()** (9 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- **.draft()** (9 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- **frozen_kv_mtp_utils.py** (8 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_utils.py`
- **FrozenKVMTPDraftExtendInput** (8 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_utils.py`
- **LogitsProcessorOutput** (8 connections) — `python/sglang/srt/speculative/frozen_kv_mtp_utils.py`
- *... and 71 more nodes in this community*

## Relationships

- [[Multi-Step Draft Attention (FP8)]] (85 shared connections)
- [[CLI Arg Parsing & Deprecation]] (51 shared connections)
- [[Aiter Attention Backend]] (23 shared connections)
- [[Vision-Language Model Configs]] (18 shared connections)
- [[Model Configs & Pooler]] (16 shared connections)
- [[Hybrid Attention Backend]] (11 shared connections)
- [[Community 34]] (9 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (9 shared connections)
- [[Disaggregation Utils & Cache Tests]] (4 shared connections)
- [[Community 47]] (3 shared connections)
- [[Breakable CUDA Graph (TBO)]] (2 shared connections)
- [[Community 382]] (1 shared connections)

## Source Files

- `python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py`
- `python/sglang/srt/speculative/frozen_kv_mtp_info.py`
- `python/sglang/srt/speculative/frozen_kv_mtp_utils.py`
- `python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- `python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py`

## Audit Trail

- EXTRACTED: 313 (45%)
- INFERRED: 382 (55%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*