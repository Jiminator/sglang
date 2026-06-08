# Community 273

> 24 nodes

## Key Concepts

- **RequestMetricsExporter** (11 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **FileRequestMetricsExporter** (11 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **ServerArgs** (7 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **GenerateReqInput** (6 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **EmbeddingReqInput** (6 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **create_request_metrics_exporters()** (6 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **request_metrics_exporter.py** (5 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **._format_output_data()** (5 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **.write_record()** (5 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **.write_record()** (4 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **._ensure_file_handler()** (4 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **.__init__()** (4 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **._create_exporters()** (4 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **.__init__()** (3 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **.close()** (3 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **.__init__()** (2 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **Abstract base class for exporting request-level performance metrics to a data de** (1 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **Format request-level output data containing performance metrics. This method** (1 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **Write a data record corresponding to a single request, containing performance me** (1 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **Lightweight `RequestMetricsExporter` implementation that writes records to files** (1 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **Ensure the file handler is open for the current hour suffix.** (1 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **Close the current file handler.** (1 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **Create and configure RequestMetricsExporter instances based on server args.** (1 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`
- **Create and configure `RequestMetricsExporter`s based on server args.** (1 connections) — `python/sglang/srt/observability/request_metrics_exporter.py`

## Relationships

- [[Anthropic/OpenAI API Entrypoints]] (13 shared connections)
- [[CLI Arg Parsing & Deprecation]] (5 shared connections)
- [[Aibrix KV Cache Storage]] (2 shared connections)

## Source Files

- `python/sglang/srt/observability/request_metrics_exporter.py`

## Audit Trail

- EXTRACTED: 79 (84%)
- INFERRED: 15 (16%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*