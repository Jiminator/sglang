# Community 65

> 94 nodes

## Key Concepts

- **StreamingASRState** (39 connections) — `python/sglang/srt/entrypoints/openai/streaming_asr.py`
- **RealtimeConnection** (27 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- **random_uuid()** (19 connections) — `python/sglang/srt/utils/common.py`
- **._generate_chunked_asr_stream()** (14 connections) — `python/sglang/srt/entrypoints/openai/serving_transcription.py`
- **handle_realtime_transcription()** (13 connections) — `python/sglang/srt/entrypoints/openai/realtime/handler.py`
- **process_asr_chunk()** (13 connections) — `python/sglang/srt/entrypoints/openai/streaming_asr.py`
- **.__init__()** (11 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- **TranscriptionRequest** (11 connections) — `python/sglang/srt/entrypoints/openai/serving_transcription.py`
- **._on_input_audio_buffer_commit()** (10 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- **._run_inference()** (10 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- **Request** (10 connections) — `python/sglang/srt/entrypoints/openai/serving_transcription.py`
- **._generate_transcription_stream()** (10 connections) — `python/sglang/srt/entrypoints/openai/serving_transcription.py`
- **GenerateReqInput** (9 connections) — `python/sglang/srt/entrypoints/openai/serving_transcription.py`
- **session.py** (8 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- **._dispatch()** (8 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- **._on_session_update()** (8 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- **._send()** (8 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- **._send_error()** (8 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- **._handle_streaming_request()** (8 connections) — `python/sglang/srt/entrypoints/openai/serving_transcription.py`
- **WebSocket** (7 connections) — `python/sglang/srt/entrypoints/openai/realtime/handler.py`
- **_reject_before_session()** (7 connections) — `python/sglang/srt/entrypoints/openai/realtime/handler.py`
- **._on_input_audio_buffer_append()** (7 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- **._emit_transcription_delta()** (7 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- **BaseModel** (6 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- **_SessionConfig** (6 connections) — `python/sglang/srt/entrypoints/openai/realtime/session.py`
- *... and 69 more nodes in this community*

## Relationships

- [[Anthropic/OpenAI API Entrypoints]] (61 shared connections)
- [[CLI Arg Parsing & Deprecation]] (20 shared connections)
- [[Community 60]] (6 shared connections)
- [[Model Config & Encode Server]] (3 shared connections)
- [[Community 257]] (2 shared connections)
- [[Community 82]] (2 shared connections)
- [[Community 47]] (1 shared connections)
- [[Community 444]] (1 shared connections)
- [[Community 42]] (1 shared connections)

## Source Files

- `python/sglang/srt/entrypoints/openai/realtime/handler.py`
- `python/sglang/srt/entrypoints/openai/realtime/session.py`
- `python/sglang/srt/entrypoints/openai/serving_transcription.py`
- `python/sglang/srt/entrypoints/openai/streaming_asr.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 313 (65%)
- INFERRED: 172 (35%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*