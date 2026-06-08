# Community 209

> 33 nodes

## Key Concepts

- **TranscriptionAdapter** (15 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **.build_verbose_response()** (5 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **resolve_adapter()** (5 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **base.py** (4 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **.build_sampling_params()** (3 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **register_transcription_adapter()** (3 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **TranscriptionRequest** (2 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **.supports_language_detection()** (2 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **.build_fused_autodetect_params()** (2 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **.parse_fused_output()** (2 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **.strip_special_tokens()** (2 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **.supports_chunked_streaming()** (2 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **.model_sample_rate()** (2 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **.prompt_template()** (2 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **.chunked_streaming_config()** (2 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **.postprocess_text()** (2 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **TranscriptionUsage** (1 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **TranscriptionVerboseResponse** (1 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **callable** (1 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **Abstract base for model-specific transcription logic.      Subclass this and dec** (1 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **Return the ``sampling_params`` dict for ``GenerateReqInput``.** (1 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **Whether this model supports automatic language detection.          When True, th** (1 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **Return ``sampling_params`` dict for a fused detect+transcribe request.** (1 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **Parse the fused output into ``(language_code, user_visible_text)``.          Cal** (1 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- **Best-effort scrub of model-specific special-token strings.          Used as a fa** (1 connections) — `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`
- *... and 8 more nodes in this community*

## Relationships

- [[Aibrix KV Cache Storage]] (2 shared connections)
- [[Community 47]] (1 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (1 shared connections)

## Source Files

- `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`

## Audit Trail

- EXTRACTED: 70 (97%)
- INFERRED: 2 (3%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*