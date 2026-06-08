# Community 399

> 16 nodes

## Key Concepts

- **VoxtralMultimodalProcessor** (15 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **MultimodalProcessorOutput** (7 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **.process_mm_data_async()** (6 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **._build_input_ids_with_audio()** (4 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **._compute_audio_token_count()** (3 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **._insert_audio_placeholders()** (3 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **._find_audio_offsets()** (3 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **._parse_mistral_prompt()** (3 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **voxtral.py** (2 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **.__init__()** (2 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **Multimodal processor for Voxtral (speech-to-text) models.** (1 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **Compute the number of [AUDIO] tokens for a given audio length.** (1 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **Insert [AUDIO] placeholder texts into the prompt for load_mm_data.** (1 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **Find consecutive runs of audio_token_id in input_ids.** (1 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **Build input_ids by tokenizing text and inserting audio tokens.          The inpu** (1 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`
- **Parse a Mistral-formatted prompt into a list of messages.** (1 connections) — `python/sglang/srt/multimodal/processors/voxtral.py`

## Relationships

- [[Vision-Language Model Configs]] (6 shared connections)
- [[Community 54]] (6 shared connections)
- [[Community 102]] (2 shared connections)

## Source Files

- `python/sglang/srt/multimodal/processors/voxtral.py`

## Audit Trail

- EXTRACTED: 41 (76%)
- INFERRED: 13 (24%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*