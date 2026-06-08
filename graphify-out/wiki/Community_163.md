# Community 163

> 40 nodes

## Key Concepts

- **CompletionRequest** (14 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- **._generate_completion_stream()** (13 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- **._build_completion_response()** (13 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- **Request** (9 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- **GenerateReqInput** (9 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- **._handle_non_streaming_request()** (9 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- **process_cached_tokens_details_from_ret()** (9 connections) — `python/sglang/srt/entrypoints/openai/utils.py`
- **._handle_streaming_request()** (8 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- **._convert_to_internal_request()** (7 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- **utils.py** (7 connections) — `python/sglang/srt/entrypoints/openai/utils.py`
- **process_hidden_states_from_ret()** (7 connections) — `python/sglang/srt/entrypoints/openai/utils.py`
- **process_routed_experts_from_ret()** (7 connections) — `python/sglang/srt/entrypoints/openai/utils.py`
- **cached_tokens_details_from_dict()** (7 connections) — `python/sglang/srt/entrypoints/openai/utils.py`
- **._build_sampling_params()** (5 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- **to_openai_style_logprobs()** (5 connections) — `python/sglang/srt/entrypoints/openai/utils.py`
- **._get_echo_text()** (4 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- **._prepare_echo_prompts()** (4 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- **Any** (4 connections) — `python/sglang/srt/entrypoints/openai/utils.py`
- **should_include_usage()** (4 connections) — `python/sglang/srt/entrypoints/openai/utils.py`
- **convert_embeds_to_tensors()** (4 connections) — `python/sglang/srt/entrypoints/openai/utils.py`
- **._validate_request()** (3 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- **ChatCompletionRequest** (3 connections) — `python/sglang/srt/entrypoints/openai/utils.py`
- **CompletionRequest** (3 connections) — `python/sglang/srt/entrypoints/openai/utils.py`
- **CachedTokensDetails** (2 connections) — `python/sglang/srt/entrypoints/openai/utils.py`
- **Validate that the input is valid.** (1 connections) — `python/sglang/srt/entrypoints/openai/serving_completions.py`
- *... and 15 more nodes in this community*

## Relationships

- [[Anthropic/OpenAI API Entrypoints]] (40 shared connections)
- [[Community 60]] (3 shared connections)
- [[Community 345]] (2 shared connections)
- [[Community 370]] (1 shared connections)

## Source Files

- `python/sglang/srt/entrypoints/openai/serving_completions.py`
- `python/sglang/srt/entrypoints/openai/utils.py`

## Audit Trail

- EXTRACTED: 133 (76%)
- INFERRED: 43 (24%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*