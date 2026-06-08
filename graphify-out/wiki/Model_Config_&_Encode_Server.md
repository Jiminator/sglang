# Model Config & Encode Server

> 258 nodes

## Key Concepts

- **ModelConfig** (321 connections) — `python/sglang/srt/configs/model_config.py`
- **MMEncoder** (77 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **ProfileReqType** (73 connections) — `python/sglang/srt/managers/io_struct.py`
- **EmbeddingResult** (52 connections) — `python/sglang/srt/mem_cache/multimodal_cache.py`
- **MultiModalStaticCache** (51 connections) — `python/sglang/srt/mem_cache/multimodal_cache.py`
- **EmbeddingData** (39 connections) — `python/sglang/srt/disaggregation/encode_receiver.py`
- **encode_server.py** (35 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **DPDispatcher** (33 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **Modality** (32 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **EmbeddingCacheController** (31 connections) — `python/sglang/srt/mem_cache/storage/mooncake_store/embedding_cache_controller.py`
- **EncoderScheduler** (30 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **MMError** (24 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **EncoderProfiler** (24 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **Tensor** (23 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **BadRequestError** (22 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **ServerArgs** (22 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **EmbeddingData** (22 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **InternalError** (21 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **PendingRequest** (21 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **TensorWrapper** (20 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **.__init__()** (20 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **ProfileReq** (19 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **Process** (18 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **Lock** (18 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **.batch_encode()** (17 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- *... and 233 more nodes in this community*

## Relationships

- [[Grammar Manager & HiCache Clear]] (144 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (112 shared connections)
- [[Vision-Language Model Configs]] (87 shared connections)
- [[Hybrid Attention Backend]] (73 shared connections)
- [[CLI Arg Parsing & Deprecation]] (55 shared connections)
- [[Disaggregation Bootstrap & Decode]] (54 shared connections)
- [[Community 35]] (21 shared connections)
- [[HiCache Controller & Radix Tree]] (19 shared connections)
- [[NCCL Symmetric Memory]] (14 shared connections)
- [[Community 132]] (12 shared connections)
- [[Weight Loading & EPLB]] (11 shared connections)
- [[Community 312]] (9 shared connections)

## Source Files

- `python/sglang/srt/configs/model_config.py`
- `python/sglang/srt/disaggregation/encode_grpc_server.py`
- `python/sglang/srt/disaggregation/encode_receiver.py`
- `python/sglang/srt/disaggregation/encode_server.py`
- `python/sglang/srt/dllm/config.py`
- `python/sglang/srt/layers/dp_attention.py`
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/mm_utils.py`
- `python/sglang/srt/mem_cache/multimodal_cache.py`
- `python/sglang/srt/mem_cache/storage/mooncake_store/embedding_cache_controller.py`
- `python/sglang/srt/model_loader/utils.py`

## Audit Trail

- EXTRACTED: 1030 (52%)
- INFERRED: 968 (48%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*