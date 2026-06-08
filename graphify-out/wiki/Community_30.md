# Community 30

> 184 nodes

## Key Concepts

- **SchedulerMetricsCollector** (139 connections) — `python/sglang/srt/observability/metrics_collector.py`
- **SchedulerMetricsReporter** (95 connections) — `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
- **ScheduleBatch** (27 connections) — `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
- **Req** (27 connections) — `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
- **LogitsProcessorOutput** (22 connections) — `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
- **GenerationBatchResult** (22 connections) — `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
- **ScheduleBatch** (22 connections) — `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
- **GaugeHistogram** (22 connections) — `python/sglang/srt/utils/gauge_histogram.py`
- **.process_batch_result_prefill()** (17 connections) — `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
- **EmbeddingBatchResult** (17 connections) — `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
- **PrefillAdder** (16 connections) — `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
- **Req** (16 connections) — `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
- **DPCooperationInfo** (16 connections) — `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
- **GenerationBatchResult** (16 connections) — `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
- **EmbeddingBatchResult** (16 connections) — `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
- **metrics_collector.py** (16 connections) — `python/sglang/srt/observability/metrics_collector.py`
- **QueueCount** (16 connections) — `python/sglang/srt/observability/metrics_collector.py`
- **DeviceTimer** (16 connections) — `python/sglang/srt/utils/device_timer.py`
- **Tensor** (15 connections) — `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
- **WelfordAccumulator** (15 connections) — `python/sglang/srt/observability/forward_pass_metrics.py`
- **ForwardPassMetrics** (15 connections) — `python/sglang/srt/observability/forward_pass_metrics.py`
- **SchedulerStats** (14 connections) — `python/sglang/srt/observability/metrics_collector.py`
- **SchedulerMetricsCollectorContext** (14 connections) — `python/sglang/srt/observability/metrics_collector.py`
- **SchedulerStatusLogger** (14 connections) — `python/sglang/srt/utils/scheduler_status_logger.py`
- **._handle_finish_state_updated_req()** (12 connections) — `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
- *... and 159 more nodes in this community*

## Relationships

- [[Grammar Manager & HiCache Clear]] (173 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (57 shared connections)
- [[HiCache Controller & Radix Tree]] (36 shared connections)
- [[CLI Arg Parsing & Deprecation]] (31 shared connections)
- [[Disaggregation Bootstrap & Decode]] (20 shared connections)
- [[Aiter Attention Backend]] (9 shared connections)
- [[Model Config & Encode Server]] (7 shared connections)
- [[Model Configs & Pooler]] (6 shared connections)
- [[Community 69]] (5 shared connections)
- [[Aibrix KV Cache Storage]] (4 shared connections)
- [[Community 468]] (4 shared connections)
- [[Context-Parallel Attention]] (3 shared connections)

## Source Files

- `python/sglang/srt/managers/prefill_delayer.py`
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
- `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
- `python/sglang/srt/observability/forward_pass_metrics.py`
- `python/sglang/srt/observability/metrics_collector.py`
- `python/sglang/srt/observability/utils.py`
- `python/sglang/srt/utils/common.py`
- `python/sglang/srt/utils/device_timer.py`
- `python/sglang/srt/utils/gauge_histogram.py`
- `python/sglang/srt/utils/scheduler_status_logger.py`

## Audit Trail

- EXTRACTED: 593 (53%)
- INFERRED: 530 (47%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*