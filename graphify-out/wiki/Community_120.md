# Community 120

> 54 nodes

## Key Concepts

- **MlxModelRunnerStub** (20 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner_stub.py`
- **MlxPendingDecode** (18 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **MlxPendingPrefill** (17 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **MlxPendingExtend** (17 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **ScheduleBatch** (13 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **array** (13 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **MlxPendingDecode** (13 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **GenerationBatchResult** (12 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **MlxPendingPrefill** (12 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **MlxPendingExtend** (12 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **ForwardBatch** (10 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **PPProxyTensors** (10 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **.async_forward_batch_generation_mlx()** (10 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **MlxPendingJob** (9 connections) — `python/sglang/srt/hardware_backend/mlx/scheduler_mixin.py`
- **._async_extend_batch()** (9 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **.forward_batch_generation()** (8 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **model_runner.py** (6 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **._forward_batch_generation_mlx()** (6 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **.finalize_mlx_result()** (6 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **.initialize()** (5 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner_stub.py`
- **model_runner_stub.py** (4 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner_stub.py`
- **.load_model()** (4 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner_stub.py`
- **._init_model_runner()** (4 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **._ensure_mlx_pool_initialized()** (4 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- **._cleanup_stale_rids()** (4 connections) — `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- *... and 29 more nodes in this community*

## Relationships

- [[Grammar Manager & HiCache Clear]] (20 shared connections)
- [[Community 87]] (17 shared connections)
- [[CLI Arg Parsing & Deprecation]] (9 shared connections)
- [[Disaggregation Bootstrap & Decode]] (8 shared connections)
- [[Model Configs & Pooler]] (8 shared connections)
- [[Vision-Language Model Configs]] (8 shared connections)
- [[DeepSeek MLA Attention & MoE]] (8 shared connections)
- [[Community 133]] (3 shared connections)
- [[Community 84]] (2 shared connections)
- [[Community 152]] (2 shared connections)
- [[Disaggregation Utils & Cache Tests]] (1 shared connections)
- [[HiCache Controller & Radix Tree]] (1 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- `python/sglang/srt/hardware_backend/mlx/model_runner_stub.py`
- `python/sglang/srt/hardware_backend/mlx/scheduler_mixin.py`
- `python/sglang/srt/hardware_backend/mlx/tp_worker.py`

## Audit Trail

- EXTRACTED: 153 (53%)
- INFERRED: 134 (47%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*