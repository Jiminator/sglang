# Community 390

> 16 nodes

## Key Concepts

- **MemoryMetrics** (13 connections) — `python/sglang/srt/managers/io_struct.py`
- **SpeculativeMetrics** (13 connections) — `python/sglang/srt/managers/io_struct.py`
- **LoRAMetrics** (13 connections) — `python/sglang/srt/managers/io_struct.py`
- **DisaggregationMetrics** (13 connections) — `python/sglang/srt/managers/io_struct.py`
- **QueueMetrics** (13 connections) — `python/sglang/srt/managers/io_struct.py`
- **GetLoadsReqInput** (12 connections) — `python/sglang/srt/managers/scheduler_components/load_inquirer.py`
- **GetLoadsReqOutput** (12 connections) — `python/sglang/srt/managers/scheduler_components/load_inquirer.py`
- **.get_loads()** (10 connections) — `python/sglang/srt/managers/scheduler_components/load_inquirer.py`
- **.get_num_waiting_uncached_tokens()** (3 connections) — `python/sglang/srt/managers/scheduler_components/load_inquirer.py`
- **Memory breakdown metrics.** (1 connections) — `python/sglang/srt/managers/io_struct.py`
- **Speculative decoding metrics.** (1 connections) — `python/sglang/srt/managers/io_struct.py`
- **LoRA adapter pool metrics.** (1 connections) — `python/sglang/srt/managers/io_struct.py`
- **PD disaggregation metrics.** (1 connections) — `python/sglang/srt/managers/io_struct.py`
- **Detailed queue breakdown.** (1 connections) — `python/sglang/srt/managers/io_struct.py`
- **Get uncached input tokens waiting for prefill compute.** (1 connections) — `python/sglang/srt/managers/scheduler_components/load_inquirer.py`
- **Get comprehensive load metrics for /v1/loads endpoint.          Args:** (1 connections) — `python/sglang/srt/managers/scheduler_components/load_inquirer.py`

## Relationships

- [[Anthropic/OpenAI API Entrypoints]] (25 shared connections)
- [[Grammar Manager & HiCache Clear]] (25 shared connections)
- [[Vision-Language Model Configs]] (5 shared connections)
- [[CLI Arg Parsing & Deprecation]] (4 shared connections)

## Source Files

- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/scheduler_components/load_inquirer.py`

## Audit Trail

- EXTRACTED: 27 (25%)
- INFERRED: 82 (75%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*