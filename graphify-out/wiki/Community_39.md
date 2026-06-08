# Community 39

> 152 nodes

## Key Concepts

- **FreezeGCReq** (110 connections) — `python/sglang/srt/managers/io_struct.py`
- **BatchTokenIDOutput** (67 connections) — `python/sglang/srt/managers/io_struct.py`
- **BatchEmbeddingOutput** (66 connections) — `python/sglang/srt/managers/io_struct.py`
- **BatchStrOutput** (63 connections) — `python/sglang/srt/managers/io_struct.py`
- **DetokenizerManager** (41 connections) — `python/sglang/srt/managers/detokenizer_manager.py`
- **BaseBatchReq** (36 connections) — `python/sglang/srt/managers/io_struct.py`
- **TokenizerWorkerRegistration** (27 connections) — `python/sglang/srt/managers/io_struct.py`
- **MultiHttpWorkerDetokenizerMixin** (27 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **PauseContinueBroadcast** (26 connections) — `python/sglang/srt/managers/io_struct.py`
- **get_zmq_socket()** (24 connections) — `python/sglang/srt/utils/network.py`
- **SocketMapping** (22 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **MultiDetokenizerRouter** (21 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **PortArgs** (20 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **Any** (18 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **ServerArgs** (18 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **PauseContinueBroadcast** (17 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **DetokenizerManager** (16 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **SharedMemory** (16 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **DecodeStatus** (15 connections) — `python/sglang/srt/managers/detokenizer_manager.py`
- **PauseGenerationReqInput** (15 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **ContinueGenerationReqInput** (15 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **BaseReq** (15 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **BaseBatchReq** (15 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **Socket** (15 connections) — `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- **Req** (15 connections) — `python/sglang/srt/managers/scheduler_components/output_streamer.py`
- *... and 127 more nodes in this community*

## Relationships

- [[Anthropic/OpenAI API Entrypoints]] (275 shared connections)
- [[Grammar Manager & HiCache Clear]] (104 shared connections)
- [[CLI Arg Parsing & Deprecation]] (31 shared connections)
- [[Disaggregation Bootstrap & Decode]] (17 shared connections)
- [[Vision-Language Model Configs]] (7 shared connections)
- [[Model Config & Encode Server]] (7 shared connections)
- [[HiCache Controller & Radix Tree]] (6 shared connections)
- [[Community 42]] (5 shared connections)
- [[Community 32]] (4 shared connections)
- [[Community 131]] (4 shared connections)
- [[Community 33]] (3 shared connections)
- [[Community 47]] (2 shared connections)

## Source Files

- `python/sglang/srt/managers/detokenizer_manager.py`
- `python/sglang/srt/managers/disagg_service.py`
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/multi_tokenizer_mixin.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/scheduler_components/ipc_channels.py`
- `python/sglang/srt/managers/scheduler_components/output_sender.py`
- `python/sglang/srt/managers/scheduler_components/output_streamer.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/utils/common.py`
- `python/sglang/srt/utils/network.py`
- `python/sglang/srt/utils/patch_tokenizer.py`

## Audit Trail

- EXTRACTED: 455 (36%)
- INFERRED: 794 (64%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*