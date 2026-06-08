# Hybrid Attention Backend

> 401 nodes

## Key Concepts

- **ModelRunner** (139 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **LoadConfig** (108 connections) — `python/sglang/srt/configs/load_config.py`
- **DefaultModelLoader** (94 connections) — `python/sglang/srt/model_loader/loader.py`
- **DpPaddingMode** (92 connections) — `python/sglang/srt/layers/dp_attention.py`
- **AttentionArch** (91 connections) — `python/sglang/srt/configs/model_config.py`
- **DeviceConfig** (70 connections) — `python/sglang/srt/configs/device_config.py`
- **ExpertLocationMetadata** (70 connections) — `python/sglang/srt/eplb/expert_location.py`
- **MemoryPoolConfig** (70 connections) — `python/sglang/srt/model_executor/pool_configurator.py`
- **ForwardBatch** (69 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **TboAttnBackend** (67 connections) — `python/sglang/srt/layers/attention/tbo_backend.py`
- **NPUGraphRunner** (66 connections) — `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`
- **NgramVerifyInput** (66 connections) — `python/sglang/srt/speculative/ngram_info.py`
- **ElasticEPStateManager** (64 connections) — `python/sglang/srt/elastic_ep/elastic_ep.py`
- **LogitsProcessorOutput** (64 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **PPProxyTensors** (64 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **Tensor** (62 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **RankZeroFilter** (61 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **ModelRunnerOutput** (61 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **LocalSerializedTensor** (61 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **_EagerBufferRegistry** (60 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **LoRARef** (60 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **LoRAManager** (59 connections) — `python/sglang/srt/lora/lora_manager.py`
- **Module** (58 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **ModelConfig** (58 connections) — `python/sglang/srt/model_executor/model_runner.py`
- **ServerArgs** (58 connections) — `python/sglang/srt/model_executor/model_runner.py`
- *... and 376 more nodes in this community*

## Relationships

- [[Multi-Step Draft Attention (FP8)]] (187 shared connections)
- [[Community 35]] (113 shared connections)
- [[Breakable CUDA Graph (TBO)]] (109 shared connections)
- [[Grammar Manager & HiCache Clear]] (103 shared connections)
- [[DeepSeek MLA Attention & MoE]] (96 shared connections)
- [[Disaggregation Bootstrap & Decode]] (80 shared connections)
- [[Model Configs & Pooler]] (79 shared connections)
- [[CLI Arg Parsing & Deprecation]] (74 shared connections)
- [[Model Config & Encode Server]] (73 shared connections)
- [[Aiter Attention Backend]] (70 shared connections)
- [[Vision-Language Model Configs]] (48 shared connections)
- [[Community 69]] (28 shared connections)

## Source Files

- `python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py`
- `python/sglang/srt/configs/device_config.py`
- `python/sglang/srt/configs/load_config.py`
- `python/sglang/srt/configs/model_config.py`
- `python/sglang/srt/configs/modelopt_config.py`
- `python/sglang/srt/elastic_ep/elastic_ep.py`
- `python/sglang/srt/elastic_ep/expert_backup_client.py`
- `python/sglang/srt/eplb/eplb_manager.py`
- `python/sglang/srt/eplb/expert_distribution.py`
- `python/sglang/srt/eplb/expert_location.py`
- `python/sglang/srt/eplb/expert_location_updater.py`
- `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`
- `python/sglang/srt/hardware_backend/npu/utils.py`
- `python/sglang/srt/layers/attention/hybrid_attn_backend.py`
- `python/sglang/srt/layers/attention/tbo_backend.py`
- `python/sglang/srt/layers/dp_attention.py`
- `python/sglang/srt/layers/moe/deepep_waterfill.py`
- `python/sglang/srt/layers/moe/hash_topk.py`
- `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py`
- `python/sglang/srt/layers/n_gram_embedding.py`

## Audit Trail

- EXTRACTED: 1299 (30%)
- INFERRED: 3051 (70%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*