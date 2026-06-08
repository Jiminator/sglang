# Disaggregation Bootstrap & Decode

> 480 nodes

## Key Concepts

- **ReqToTokenPool** (274 connections) — `python/sglang/srt/mem_cache/memory_pool.py`
- **NetworkAddress** (244 connections) — `python/sglang/srt/utils/network.py`
- **FINISH_ABORT** (172 connections) — `python/sglang/srt/managers/schedule_batch.py`
- **TorchMemorySaverAdapter** (169 connections) — `python/sglang/srt/utils/torch_memory_saver_adapter.py`
- **Scheduler** (117 connections) — `python/sglang/srt/dllm/mixin/scheduler.py`
- **DecodePreallocQueue** (98 connections) — `python/sglang/srt/disaggregation/decode.py`
- **Mamba2CacheParams** (56 connections) — `python/sglang/srt/configs/mamba_utils.py`
- **FINISH_LENGTH** (49 connections) — `python/sglang/srt/managers/schedule_batch.py`
- **DataParallelController** (43 connections) — `python/sglang/srt/managers/data_parallel_controller.py`
- **DecodeStagingHandler** (42 connections) — `python/sglang/srt/disaggregation/common/staging_handler.py`
- **DecodeRequest** (36 connections) — `python/sglang/srt/disaggregation/decode.py`
- **Req** (33 connections) — `python/sglang/srt/disaggregation/decode.py`
- **DecodeReqToTokenPool** (32 connections) — `python/sglang/srt/disaggregation/decode.py`
- **DecodePrefixMatch** (30 connections) — `python/sglang/srt/disaggregation/decode_hicache_mixin.py`
- **DecodeHiCacheTransferMixin** (30 connections) — `python/sglang/srt/disaggregation/decode_hicache_mixin.py`
- **DecodeHiCachePreallocMixin** (29 connections) — `python/sglang/srt/disaggregation/decode_hicache_mixin.py`
- **HiCacheRestoreGatedKVReceiver** (29 connections) — `python/sglang/srt/disaggregation/decode_hicache_mixin.py`
- **encode_receiver.py** (29 connections) — `python/sglang/srt/disaggregation/encode_receiver.py`
- **Scheduler** (28 connections) — `python/sglang/srt/disaggregation/decode.py`
- **HybridMambaDecodeReqToTokenPool** (27 connections) — `python/sglang/srt/disaggregation/decode.py`
- **HiCacheRestoreResult** (27 connections) — `python/sglang/srt/disaggregation/decode_hicache_mixin.py`
- **MMReceiverBase** (24 connections) — `python/sglang/srt/disaggregation/encode_receiver.py`
- **Req** (24 connections) — `python/sglang/srt/disaggregation/prefill.py`
- **BlockReqInput** (24 connections) — `python/sglang/srt/managers/io_struct.py`
- **ScheduleBatch** (23 connections) — `python/sglang/srt/disaggregation/decode.py`
- *... and 455 more nodes in this community*

## Relationships

- [[Grammar Manager & HiCache Clear]] (372 shared connections)
- [[HiCache Controller & Radix Tree]] (192 shared connections)
- [[Disaggregation Utils & Cache Tests]] (146 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (130 shared connections)
- [[CLI Arg Parsing & Deprecation]] (117 shared connections)
- [[Hybrid Attention Backend]] (80 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (79 shared connections)
- [[Model Config & Encode Server]] (54 shared connections)
- [[Vision-Language Model Configs]] (30 shared connections)
- [[Breakable CUDA Graph (TBO)]] (30 shared connections)
- [[Community 68]] (29 shared connections)
- [[Aiter Attention Backend]] (25 shared connections)

## Source Files

- `python/sglang/srt/configs/mamba_utils.py`
- `python/sglang/srt/constrained/grammar_manager.py`
- `python/sglang/srt/disaggregation/common/conn.py`
- `python/sglang/srt/disaggregation/common/staging_handler.py`
- `python/sglang/srt/disaggregation/decode.py`
- `python/sglang/srt/disaggregation/decode_hicache_mixin.py`
- `python/sglang/srt/disaggregation/encode_receiver.py`
- `python/sglang/srt/disaggregation/prefill.py`
- `python/sglang/srt/disaggregation/utils.py`
- `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- `python/sglang/srt/dllm/mixin/scheduler.py`
- `python/sglang/srt/elastic_ep/expert_backup_manager.py`
- `python/sglang/srt/hardware_backend/mlx/model_runner_stub.py`
- `python/sglang/srt/managers/data_parallel_controller.py`
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/scheduler_components/invariant_checker.py`
- `python/sglang/srt/managers/scheduler_input_blocker.py`
- `python/sglang/srt/managers/tokenizer_manager.py`

## Audit Trail

- EXTRACTED: 1639 (41%)
- INFERRED: 2346 (59%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*