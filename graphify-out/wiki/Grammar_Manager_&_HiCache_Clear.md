# Grammar Manager & HiCache Clear

> 380 nodes

## Key Concepts

- **UpdateWeightsFromTensorReqInput** (228 connections) — `python/sglang/srt/managers/io_struct.py`
- **UpdateWeightsFromIPCReqInput** (224 connections) — `python/sglang/srt/managers/io_struct.py`
- **UpdateWeightsFromDistributedReqInput** (198 connections) — `python/sglang/srt/managers/io_struct.py`
- **InitWeightsUpdateGroupReqInput** (198 connections) — `python/sglang/srt/managers/io_struct.py`
- **DestroyWeightsUpdateGroupReqInput** (198 connections) — `python/sglang/srt/managers/io_struct.py`
- **GetWeightsByNameReqInput** (198 connections) — `python/sglang/srt/managers/io_struct.py`
- **UnloadLoRAAdapterReqInput** (188 connections) — `python/sglang/srt/managers/io_struct.py`
- **LoadLoRAAdapterFromTensorsReqInput** (188 connections) — `python/sglang/srt/managers/io_struct.py`
- **DllmConfig** (175 connections) — `python/sglang/srt/dllm/config.py`
- **InitWeightsSendGroupForRemoteInstanceReqInput** (175 connections) — `python/sglang/srt/managers/io_struct.py`
- **SendWeightsToRemoteInstanceReqInput** (175 connections) — `python/sglang/srt/managers/io_struct.py`
- **ReleaseMemoryOccupationReqInput** (175 connections) — `python/sglang/srt/managers/io_struct.py`
- **ResumeMemoryOccupationReqInput** (175 connections) — `python/sglang/srt/managers/io_struct.py`
- **OpenSessionReqInput** (173 connections) — `python/sglang/srt/managers/io_struct.py`
- **CloseSessionReqInput** (173 connections) — `python/sglang/srt/managers/io_struct.py`
- **CheckWeightsReqInput** (163 connections) — `python/sglang/srt/managers/io_struct.py`
- **BaseFinishReason** (160 connections) — `python/sglang/srt/managers/schedule_batch.py`
- **AddExternalCorpusReqInput** (159 connections) — `python/sglang/srt/managers/io_struct.py`
- **AttachHiCacheStorageReqInput** (154 connections) — `python/sglang/srt/managers/io_struct.py`
- **SlowDownReqInput** (152 connections) — `python/sglang/srt/managers/io_struct.py`
- **SetInternalStateReq** (152 connections) — `python/sglang/srt/managers/io_struct.py`
- **DumperControlReqInput** (152 connections) — `python/sglang/srt/managers/io_struct.py`
- **SamplingBatchInfo** (145 connections) — `python/sglang/srt/sampling/sampling_batch_info.py`
- **HiSparseCoordinator** (134 connections) — `python/sglang/srt/managers/hisparse_coordinator.py`
- **ProfileReq** (132 connections) — `python/sglang/srt/managers/io_struct.py`
- *... and 355 more nodes in this community*

## Relationships

- [[Anthropic/OpenAI API Entrypoints]] (2388 shared connections)
- [[Disaggregation Bootstrap & Decode]] (372 shared connections)
- [[CLI Arg Parsing & Deprecation]] (288 shared connections)
- [[Community 32]] (212 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (199 shared connections)
- [[Community 30]] (173 shared connections)
- [[Vision-Language Model Configs]] (148 shared connections)
- [[Model Config & Encode Server]] (144 shared connections)
- [[Community 33]] (140 shared connections)
- [[HiCache Controller & Radix Tree]] (137 shared connections)
- [[Community 39]] (104 shared connections)
- [[Hybrid Attention Backend]] (103 shared connections)

## Source Files

- `python/sglang/srt/configs/model_config.py`
- `python/sglang/srt/constrained/grammar_manager.py`
- `python/sglang/srt/disaggregation/decode.py`
- `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- `python/sglang/srt/disaggregation/prefill.py`
- `python/sglang/srt/distributed/parallel_state_wrapper.py`
- `python/sglang/srt/dllm/config.py`
- `python/sglang/srt/dllm/mixin/scheduler.py`
- `python/sglang/srt/hardware_backend/mlx/scheduler_mixin.py`
- `python/sglang/srt/hardware_backend/mlx/tp_worker.py`
- `python/sglang/srt/lora/lora_drainer.py`
- `python/sglang/srt/lora/lora_overlap_loader.py`
- `python/sglang/srt/managers/hisparse_coordinator.py`
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/load_snapshot.py`
- `python/sglang/srt/managers/prefill_delayer.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`

## Audit Trail

- EXTRACTED: 1297 (6%)
- INFERRED: 18768 (94%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*