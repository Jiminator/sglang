# Pipeline Parallel & Custom Allreduce

> 275 nodes

## Key Concepts

- **GenerationBatchResult** (142 connections) — `python/sglang/srt/dllm/mixin/scheduler.py`
- **SchedulerPPMixin** (91 connections) — `python/sglang/srt/managers/scheduler_pp_mixin.py`
- **P2PWork** (53 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **CustomAllreduce** (51 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- **BaseTpWorker** (45 connections) — `python/sglang/srt/managers/tp_worker.py`
- **Scheduler** (42 connections) — `python/sglang/srt/managers/scheduler_pp_mixin.py`
- **TpModelWorker** (39 connections) — `python/sglang/srt/managers/tp_worker.py`
- **ForwardBatch** (25 connections) — `python/sglang/srt/managers/tp_worker.py`
- **ScheduleBatch** (25 connections) — `python/sglang/srt/managers/tp_worker.py`
- **GenerationBatchResult** (25 connections) — `python/sglang/srt/managers/tp_worker.py`
- **DllmManager** (24 connections) — `python/sglang/srt/dllm/mixin/scheduler.py`
- **deque** (24 connections) — `python/sglang/srt/managers/scheduler_pp_mixin.py`
- **Req** (20 connections) — `python/sglang/srt/dllm/mixin/scheduler.py`
- **.event_loop_pp_disagg_decode()** (19 connections) — `python/sglang/srt/managers/scheduler_pp_mixin.py`
- **ScheduleBatch** (17 connections) — `python/sglang/srt/managers/scheduler_pp_mixin.py`
- **.event_loop_pp_disagg_prefill()** (17 connections) — `python/sglang/srt/managers/scheduler_pp_mixin.py`
- **PrefillAdder** (16 connections) — `python/sglang/srt/dllm/mixin/scheduler.py`
- **PPProxyTensors** (16 connections) — `python/sglang/srt/managers/scheduler_pp_mixin.py`
- **ChunkSizePredictor** (16 connections) — `python/sglang/srt/managers/scheduler_pp_mixin.py`
- **.__init__()** (16 connections) — `python/sglang/srt/managers/tp_worker.py`
- **DllmReqPhase** (15 connections) — `python/sglang/srt/dllm/mixin/req.py`
- **GenerationBatchResult** (15 connections) — `python/sglang/srt/managers/scheduler_pp_mixin.py`
- **ConcurrentCounter** (15 connections) — `python/sglang/srt/utils/common.py`
- **PPBatchMetadata** (14 connections) — `python/sglang/srt/managers/scheduler_pp_mixin.py`
- **.profile_and_init_predictor()** (14 connections) — `python/sglang/srt/managers/scheduler_pp_mixin.py`
- *... and 250 more nodes in this community*

## Relationships

- [[Grammar Manager & HiCache Clear]] (199 shared connections)
- [[Disaggregation Bootstrap & Decode]] (79 shared connections)
- [[CLI Arg Parsing & Deprecation]] (74 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (50 shared connections)
- [[Community 42]] (34 shared connections)
- [[Community 71]] (25 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (24 shared connections)
- [[HiCache Controller & Radix Tree]] (24 shared connections)
- [[DeepSeek MLA Attention & MoE]] (21 shared connections)
- [[Aiter Attention Backend]] (21 shared connections)
- [[Vision-Language Model Configs]] (17 shared connections)
- [[Hybrid Attention Backend]] (16 shared connections)

## Source Files

- `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/dllm/mixin/req.py`
- `python/sglang/srt/dllm/mixin/scheduler.py`
- `python/sglang/srt/managers/scheduler_components/request_receiver.py`
- `python/sglang/srt/managers/scheduler_pp_mixin.py`
- `python/sglang/srt/managers/tp_worker.py`
- `python/sglang/srt/managers/utils.py`
- `python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 1008 (56%)
- INFERRED: 780 (44%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*