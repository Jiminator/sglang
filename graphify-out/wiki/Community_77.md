# Community 77

> 80 nodes

## Key Concepts

- **tokenizer_manager.py** (19 connections) — `python/sglang/srt/managers/tokenizer_manager.py`
- **PoolStats** (16 connections) — `python/sglang/srt/managers/scheduler_components/pool_stats_observer.py`
- **PoolStats** (11 connections) — `python/sglang/srt/managers/scheduler_components/invariant_checker.py`
- **kill_process_tree()** (11 connections) — `python/sglang/srt/utils/common.py`
- **.init_tokenizer_and_processor()** (9 connections) — `python/sglang/srt/managers/tokenizer_manager.py`
- **.dump_requests_before_crash()** (9 connections) — `python/sglang/srt/managers/tokenizer_manager.py`
- **WatchdogRaw** (8 connections) — `python/sglang/srt/utils/watchdog.py`
- **._check_all_pools()** (6 connections) — `python/sglang/srt/managers/scheduler_components/invariant_checker.py`
- **.get_pool_stats()** (6 connections) — `python/sglang/srt/managers/scheduler_components/pool_stats_observer.py`
- **collect_scheduler_processes()** (6 connections) — `python/sglang/srt/utils/cudacore_pyspy_dump_utils.py`
- **pyspy_dump_schedulers()** (6 connections) — `python/sglang/srt/utils/cudacore_pyspy_dump_utils.py`
- **trigger_cuda_user_coredump()** (6 connections) — `python/sglang/srt/utils/cudacore_pyspy_dump_utils.py`
- **_WatchdogReal** (6 connections) — `python/sglang/srt/utils/watchdog.py`
- **._check_pool_invariant()** (5 connections) — `python/sglang/srt/managers/scheduler_components/invariant_checker.py`
- **._check_full_pool()** (5 connections) — `python/sglang/srt/managers/scheduler_components/invariant_checker.py`
- **._check_swa_pool()** (5 connections) — `python/sglang/srt/managers/scheduler_components/invariant_checker.py`
- **.update_scheduler_stats()** (5 connections) — `python/sglang/srt/managers/scheduler_components/pool_stats_observer.py`
- **.sigterm_watchdog()** (5 connections) — `python/sglang/srt/managers/tokenizer_manager.py`
- **print_exception_wrapper()** (5 connections) — `python/sglang/srt/managers/tokenizer_manager.py`
- **cudacore_pyspy_dump_utils.py** (5 connections) — `python/sglang/srt/utils/cudacore_pyspy_dump_utils.py`
- **Process** (5 connections) — `python/sglang/srt/utils/cudacore_pyspy_dump_utils.py`
- **watchdog.py** (5 connections) — `python/sglang/srt/utils/watchdog.py`
- **.shutdown()** (4 connections) — `python/sglang/srt/entrypoints/engine.py`
- **import_processors()** (4 connections) — `python/sglang/srt/managers/multimodal_processor.py`
- **._check_mamba_pool()** (4 connections) — `python/sglang/srt/managers/scheduler_components/invariant_checker.py`
- *... and 55 more nodes in this community*

## Relationships

- [[Anthropic/OpenAI API Entrypoints]] (31 shared connections)
- [[Grammar Manager & HiCache Clear]] (16 shared connections)
- [[Disaggregation Bootstrap & Decode]] (7 shared connections)
- [[Community 124]] (4 shared connections)
- [[HiCache Controller & Radix Tree]] (3 shared connections)
- [[Community 184]] (2 shared connections)
- [[Community 42]] (2 shared connections)
- [[Community 33]] (1 shared connections)
- [[CLI Arg Parsing & Deprecation]] (1 shared connections)
- [[Community 48]] (1 shared connections)
- [[Community 107]] (1 shared connections)
- [[Community 45]] (1 shared connections)

## Source Files

- `python/sglang/srt/entrypoints/engine.py`
- `python/sglang/srt/entrypoints/http_server_engine.py`
- `python/sglang/srt/managers/multimodal_processor.py`
- `python/sglang/srt/managers/scheduler_components/invariant_checker.py`
- `python/sglang/srt/managers/scheduler_components/pool_stats_observer.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/utils/common.py`
- `python/sglang/srt/utils/cudacore_pyspy_dump_utils.py`
- `python/sglang/srt/utils/watchdog.py`

## Audit Trail

- EXTRACTED: 237 (83%)
- INFERRED: 50 (17%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*