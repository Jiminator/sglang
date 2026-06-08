# Anthropic/OpenAI API Entrypoints

> 901 nodes

## Key Concepts

- **TokenizerManager** (353 connections) — `python/sglang/srt/managers/tokenizer_manager.py`
- **GenerateReqInput** (254 connections) — `python/sglang/srt/managers/io_struct.py`
- **PortArgs** (247 connections) — `python/sglang/srt/server_args.py`
- **LoadLoRAAdapterReqInput** (214 connections) — `python/sglang/srt/managers/io_struct.py`
- **UpdateWeightFromDiskReqInput** (212 connections) — `python/sglang/srt/managers/io_struct.py`
- **APIServerReqTimeStats** (190 connections) — `python/sglang/srt/observability/req_time_stats.py`
- **SchedulerReqTimeStats** (184 connections) — `python/sglang/srt/observability/req_time_stats.py`
- **SamplingParams** (182 connections) — `python/sglang/srt/sampling/sampling_params.py`
- **PositionalEmbeds** (173 connections) — `python/sglang/srt/managers/embed_types.py`
- **ReasoningParser** (166 connections) — `python/sglang/srt/parser/reasoning_parser.py`
- **TemplateManager** (164 connections) — `python/sglang/srt/managers/template_manager.py`
- **DPControllerReqTimeStats** (162 connections) — `python/sglang/srt/observability/req_time_stats.py`
- **EmbeddingReqInput** (161 connections) — `python/sglang/srt/managers/io_struct.py`
- **PauseGenerationReqInput** (159 connections) — `python/sglang/srt/managers/io_struct.py`
- **ContinueGenerationReqInput** (157 connections) — `python/sglang/srt/managers/io_struct.py`
- **AbortReq** (152 connections) — `python/sglang/srt/managers/io_struct.py`
- **ConfigureLoggingReq** (150 connections) — `python/sglang/srt/managers/io_struct.py`
- **Engine** (137 connections) — `python/sglang/srt/entrypoints/engine.py`
- **FunctionCallParser** (121 connections) — `python/sglang/srt/function_call/function_call_parser.py`
- **Request** (116 connections) — `python/sglang/srt/entrypoints/http_server.py`
- **OpenAIServingBase** (111 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **BaseGrammarObject** (105 connections) — `python/sglang/srt/constrained/base_grammar_backend.py`
- **http_server.py** (101 connections) — `python/sglang/srt/entrypoints/http_server.py`
- **ServerStatus** (100 connections) — `python/sglang/srt/managers/tokenizer_manager.py`
- **OpenAIServingChat** (98 connections) — `python/sglang/srt/entrypoints/openai/serving_chat.py`
- *... and 876 more nodes in this community*

## Relationships

- [[Grammar Manager & HiCache Clear]] (2388 shared connections)
- [[Community 39]] (275 shared connections)
- [[CLI Arg Parsing & Deprecation]] (229 shared connections)
- [[Community 33]] (147 shared connections)
- [[Disaggregation Bootstrap & Decode]] (130 shared connections)
- [[Community 82]] (116 shared connections)
- [[Vision-Language Model Configs]] (114 shared connections)
- [[Model Config & Encode Server]] (112 shared connections)
- [[HiCache Controller & Radix Tree]] (105 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (83 shared connections)
- [[Community 65]] (61 shared connections)
- [[Aiter Attention Backend]] (57 shared connections)

## Source Files

- `python/sglang/srt/constrained/base_grammar_backend.py`
- `python/sglang/srt/constrained/reasoner_grammar_backend.py`
- `python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py`
- `python/sglang/srt/dllm/mixin/req.py`
- `python/sglang/srt/entrypoints/anthropic/protocol.py`
- `python/sglang/srt/entrypoints/anthropic/serving.py`
- `python/sglang/srt/entrypoints/engine.py`
- `python/sglang/srt/entrypoints/http_server.py`
- `python/sglang/srt/entrypoints/openai/serving_base.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `python/sglang/srt/entrypoints/openai/serving_classify.py`
- `python/sglang/srt/entrypoints/openai/serving_completions.py`
- `python/sglang/srt/entrypoints/openai/serving_embedding.py`
- `python/sglang/srt/entrypoints/openai/serving_rerank.py`
- `python/sglang/srt/entrypoints/openai/serving_responses.py`
- `python/sglang/srt/entrypoints/openai/serving_score.py`
- `python/sglang/srt/entrypoints/openai/serving_tokenize.py`
- `python/sglang/srt/entrypoints/openai/serving_transcription.py`
- `python/sglang/srt/entrypoints/openai/tool_server.py`
- `python/sglang/srt/entrypoints/openai/usage_processor.py`

## Audit Trail

- EXTRACTED: 2929 (20%)
- INFERRED: 11707 (80%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*