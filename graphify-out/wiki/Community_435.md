# Community 435

> 14 nodes

## Key Concepts

- **PollBasedBarrier** (9 connections) — `python/sglang/srt/utils/poll_based_barrier.py`
- **.handle()** (4 connections) — `python/sglang/srt/managers/scheduler_input_blocker.py`
- **Any** (4 connections) — `python/sglang/srt/managers/scheduler_input_blocker.py`
- **._handle_recv_req()** (4 connections) — `python/sglang/srt/managers/scheduler_input_blocker.py`
- **._change_state()** (4 connections) — `python/sglang/srt/managers/scheduler_input_blocker.py`
- **._execute_block_req()** (3 connections) — `python/sglang/srt/managers/scheduler_input_blocker.py`
- **._execute_unblock_req()** (3 connections) — `python/sglang/srt/managers/scheduler_input_blocker.py`
- **._handle_arrive_unblock_barrier()** (3 connections) — `python/sglang/srt/managers/scheduler_input_blocker.py`
- **._compute_global_arrived()** (3 connections) — `python/sglang/srt/utils/poll_based_barrier.py`
- **.__init__()** (2 connections) — `python/sglang/srt/managers/scheduler_input_blocker.py`
- **.poll_global_arrived()** (2 connections) — `python/sglang/srt/utils/poll_based_barrier.py`
- **poll_based_barrier.py** (1 connections) — `python/sglang/srt/utils/poll_based_barrier.py`
- **.__init__()** (1 connections) — `python/sglang/srt/utils/poll_based_barrier.py`
- **.local_arrive()** (1 connections) — `python/sglang/srt/utils/poll_based_barrier.py`

## Relationships

- [[Grammar Manager & HiCache Clear]] (8 shared connections)
- [[Disaggregation Bootstrap & Decode]] (2 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (1 shared connections)
- [[Breakable CUDA Graph (TBO)]] (1 shared connections)

## Source Files

- `python/sglang/srt/managers/scheduler_input_blocker.py`
- `python/sglang/srt/utils/poll_based_barrier.py`

## Audit Trail

- EXTRACTED: 35 (80%)
- INFERRED: 9 (20%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*