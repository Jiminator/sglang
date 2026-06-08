# Community 231

> 29 nodes

## Key Concepts

- **DS32EncodingError** (14 connections) — `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`
- **.handle_request()** (12 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **OpenAIServingRequest** (12 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **Request** (10 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **ErrorResponse** (10 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **._handle_streaming_request()** (10 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **._handle_non_streaming_request()** (10 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **GenerateReqInput** (8 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **ORJSONResponse** (8 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **Any** (7 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **StreamingResponse** (7 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **.create_error_response()** (7 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **._convert_to_internal_request()** (6 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **._generate_request_id_base()** (4 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **.extract_routed_dp_rank_from_header()** (4 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **._request_id_prefix()** (3 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **._compute_extra_key()** (3 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **._validate_request()** (3 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **.create_streaming_error_response()** (3 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **Handle the specific request type with common pattern         If you want to over** (1 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **Generate request ID based on request type** (1 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **Generate request ID based on request type** (1 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **Compute the final extra_key by concatenating cache_salt and extra_key if both ar** (1 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **Convert OpenAI request to internal format** (1 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- **Handle streaming request          Override this method in child classes that sup** (1 connections) — `python/sglang/srt/entrypoints/openai/serving_base.py`
- *... and 4 more nodes in this community*

## Relationships

- [[Anthropic/OpenAI API Entrypoints]] (35 shared connections)
- [[CLI Arg Parsing & Deprecation]] (7 shared connections)
- [[Community 371]] (4 shared connections)
- [[Model Config & Encode Server]] (1 shared connections)

## Source Files

- `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`
- `python/sglang/srt/entrypoints/openai/serving_base.py`

## Audit Trail

- EXTRACTED: 106 (70%)
- INFERRED: 45 (30%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*