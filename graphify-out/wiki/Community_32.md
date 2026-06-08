# Community 32

> 173 nodes

## Key Concepts

- **Scheduler** (230 connections) — `python/sglang/srt/managers/scheduler.py`
- **.__init__()** (40 connections) — `python/sglang/srt/managers/scheduler.py`
- **trace.py** (18 connections) — `python/sglang/srt/observability/trace.py`
- **scheduler.py** (15 connections) — `python/sglang/srt/managers/scheduler.py`
- **id_generator** (14 connections) — `python/sglang/srt/observability/trace.py`
- **.init_model_worker()** (12 connections) — `python/sglang/srt/managers/scheduler.py`
- **.handle_generate_request()** (12 connections) — `python/sglang/srt/managers/scheduler.py`
- **.get_next_batch_to_run()** (12 connections) — `python/sglang/srt/managers/scheduler.py`
- **.run_batch()** (12 connections) — `python/sglang/srt/managers/scheduler.py`
- **.event_loop_overlap()** (10 connections) — `python/sglang/srt/managers/scheduler.py`
- **._add_request_to_queue()** (10 connections) — `python/sglang/srt/managers/scheduler.py`
- **run_scheduler_process()** (10 connections) — `python/sglang/srt/managers/scheduler.py`
- **.handle_embedding_request()** (9 connections) — `python/sglang/srt/managers/scheduler.py`
- **.process_batch_result()** (9 connections) — `python/sglang/srt/managers/scheduler.py`
- **process_tracing_init()** (9 connections) — `python/sglang/srt/observability/trace.py`
- **trace_set_thread_info()** (9 connections) — `python/sglang/srt/observability/trace.py`
- **.event_loop_normal()** (8 connections) — `python/sglang/srt/managers/scheduler.py`
- **._get_new_batch_prefill_raw()** (8 connections) — `python/sglang/srt/managers/scheduler.py`
- **TraceCustomIdGenerator** (8 connections) — `python/sglang/srt/observability/trace.py`
- **load_plugins()** (8 connections) — `python/sglang/srt/plugins/__init__.py`
- **use_mlx()** (8 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **.__init__()** (7 connections) — `python/sglang/srt/entrypoints/engine.py`
- **.on_idle()** (7 connections) — `python/sglang/srt/managers/scheduler.py`
- **.is_fully_idle()** (7 connections) — `python/sglang/srt/managers/scheduler.py`
- **.init_moe_gemm_config()** (6 connections) — `python/sglang/srt/managers/scheduler.py`
- *... and 148 more nodes in this community*

## Relationships

- [[Grammar Manager & HiCache Clear]] (212 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (19 shared connections)
- [[Disaggregation Bootstrap & Decode]] (12 shared connections)
- [[CLI Arg Parsing & Deprecation]] (8 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (7 shared connections)
- [[Community 199]] (7 shared connections)
- [[Context-Parallel Attention]] (5 shared connections)
- [[Community 39]] (4 shared connections)
- [[Community 207]] (4 shared connections)
- [[Community 30]] (3 shared connections)
- [[NCCL Symmetric Memory]] (3 shared connections)
- [[Community 33]] (2 shared connections)

## Source Files

- `python/sglang/srt/entrypoints/engine.py`
- `python/sglang/srt/entrypoints/http_server.py`
- `python/sglang/srt/managers/overlap_utils.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/scheduler_components/invariant_checker.py`
- `python/sglang/srt/managers/utils.py`
- `python/sglang/srt/observability/trace.py`
- `python/sglang/srt/plugins/__init__.py`
- `python/sglang/srt/utils/tensor_bridge.py`

## Audit Trail

- EXTRACTED: 633 (71%)
- INFERRED: 258 (29%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*