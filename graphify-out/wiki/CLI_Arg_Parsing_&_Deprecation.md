# CLI Arg Parsing & Deprecation

> 472 nodes

## Key Concepts

- **ServerArgs** (1116 connections) — `python/sglang/srt/server_args.py`
- **ScheduleBatch** (448 connections) — `python/sglang/srt/managers/schedule_batch.py`
- **SpeculativeAlgorithm** (311 connections) — `python/sglang/srt/speculative/spec_info.py`
- **.__post_init__()** (53 connections) — `python/sglang/srt/server_args.py`
- **FutureMap** (45 connections) — `python/sglang/srt/managers/overlap_utils.py`
- **MultiLayerEagleWorker** (44 connections) — `python/sglang/srt/speculative/multi_layer_eagle_worker.py`
- **DFlashWorker** (42 connections) — `python/sglang/srt/speculative/dflash_worker.py`
- **MultiLayerEagleWorkerV2** (40 connections) — `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`
- **CustomSpecAlgo** (39 connections) — `python/sglang/srt/speculative/spec_registry.py`
- **NGRAMWorker** (35 connections) — `python/sglang/srt/speculative/ngram_worker.py`
- **schedule_batch.py** (28 connections) — `python/sglang/srt/managers/schedule_batch.py`
- **ObjectStorageModel** (25 connections) — `python/sglang/srt/utils/runai_utils.py`
- **ScheduleBatch** (23 connections) — `python/sglang/srt/speculative/multi_layer_eagle_worker.py`
- **StandaloneWorkerV2** (23 connections) — `python/sglang/srt/speculative/standalone_worker_v2.py`
- **NgramCorpus** (22 connections) — `python/sglang/srt/speculative/cpp_ngram/ngram_corpus.py`
- **StandaloneWorker** (21 connections) — `python/sglang/srt/speculative/standalone_worker.py`
- **WorkerFactory** (20 connections) — `python/sglang/srt/speculative/spec_registry.py`
- **ServerArgsValidator** (20 connections) — `python/sglang/srt/speculative/spec_registry.py`
- **DFlashDraftInput** (19 connections) — `python/sglang/srt/speculative/dflash_info.py`
- **CustomSpecAlgo** (18 connections) — `python/sglang/srt/speculative/spec_info.py`
- **FutureMap** (18 connections) — `python/sglang/srt/speculative/spec_info.py`
- **ScheduleBatch** (18 connections) — `python/sglang/srt/speculative/spec_info.py`
- **ServerArgs** (18 connections) — `python/sglang/srt/speculative/spec_info.py`
- **MLPSyncBatchInfo** (17 connections) — `python/sglang/srt/managers/scheduler_components/dp_attn.py`
- **Tensor** (17 connections) — `python/sglang/srt/speculative/dflash_worker.py`
- *... and 447 more nodes in this community*

## Relationships

- [[Multi-Step Draft Attention (FP8)]] (335 shared connections)
- [[Grammar Manager & HiCache Clear]] (288 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (229 shared connections)
- [[HiCache Controller & Radix Tree]] (122 shared connections)
- [[Disaggregation Bootstrap & Decode]] (117 shared connections)
- [[Hybrid Attention Backend]] (74 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (74 shared connections)
- [[Aiter Attention Backend]] (61 shared connections)
- [[Model Config & Encode Server]] (55 shared connections)
- [[Disaggregation Utils & Cache Tests]] (54 shared connections)
- [[Community 62]] (51 shared connections)
- [[Community 30]] (31 shared connections)

## Source Files

- `python/sglang/srt/arg_groups/argparse_actions.py`
- `python/sglang/srt/arg_groups/pd_disaggregation_hook.py`
- `python/sglang/srt/configs/model_config.py`
- `python/sglang/srt/disaggregation/common/conn.py`
- `python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py`
- `python/sglang/srt/disaggregation/kv_events.py`
- `python/sglang/srt/disaggregation/mori/conn.py`
- `python/sglang/srt/elastic_ep/expert_backup_client.py`
- `python/sglang/srt/eplb/expert_location.py`
- `python/sglang/srt/layers/dp_attention.py`
- `python/sglang/srt/managers/overlap_utils.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler_components/dp_attn.py`
- `python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py`
- `python/sglang/srt/managers/scheduler_recv_skipper.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/multiplex/multiplexing_mixin.py`
- `python/sglang/srt/multiplex/pdmux_context.py`
- `python/sglang/srt/observability/req_time_stats.py`
- `python/sglang/srt/observability/trace.py`

## Audit Trail

- EXTRACTED: 1406 (34%)
- INFERRED: 2729 (66%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*