# Community 451

> 13 nodes

## Key Concepts

- **WhisperProcessor** (10 connections) — `python/sglang/srt/multimodal/processors/whisper.py`
- **._load_single_item()** (6 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **.process_mm_data_async()** (6 connections) — `python/sglang/srt/multimodal/processors/whisper.py`
- **Any** (5 connections) — `python/sglang/srt/multimodal/processors/whisper.py`
- **load_audio()** (5 connections) — `python/sglang/srt/utils/common.py`
- **load_video()** (5 connections) — `python/sglang/srt/utils/common.py`
- **normalize_language_to_code()** (3 connections) — `python/sglang/srt/multimodal/processors/whisper.py`
- **whisper.py** (2 connections) — `python/sglang/srt/multimodal/processors/whisper.py`
- **._pop_sampling_param()** (2 connections) — `python/sglang/srt/multimodal/processors/whisper.py`
- **._get_language_token_id()** (2 connections) — `python/sglang/srt/multimodal/processors/whisper.py`
- **Load a single multimodal data.         If data is precomputed, returns directly.** (1 connections) — `python/sglang/srt/disaggregation/encode_server.py`
- **.__init__()** (1 connections) — `python/sglang/srt/multimodal/processors/whisper.py`
- **Convert a language input (full name or code) to ISO 639-1 code.      Args:** (1 connections) — `python/sglang/srt/multimodal/processors/whisper.py`

## Relationships

- [[Vision-Language Model Configs]] (4 shared connections)
- [[Community 102]] (4 shared connections)
- [[Community 54]] (3 shared connections)
- [[Model Config & Encode Server]] (2 shared connections)
- [[Community 42]] (2 shared connections)
- [[Community 71]] (2 shared connections)
- [[Community 47]] (1 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (1 shared connections)

## Source Files

- `python/sglang/srt/disaggregation/encode_server.py`
- `python/sglang/srt/multimodal/processors/whisper.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 32 (65%)
- INFERRED: 17 (35%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*