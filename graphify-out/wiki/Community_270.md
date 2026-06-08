# Community 270

> 24 nodes

## Key Concepts

- **TransformerEncoderBase** (22 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **ConformerEncoder** (17 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **.forward_embeddings()** (7 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **.forward()** (6 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **._streaming_mask()** (5 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **.compute_lens_change()** (3 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **._chunk_size_selection()** (3 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **._forward_embeddings_core()** (3 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **.calculate_hs_mask()** (3 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **unfold_tensor()** (3 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **.forward()** (2 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **._get_embed_class()** (2 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **._position_embedding()** (2 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **.get_offset()** (2 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **.init_relative_attention_bias()** (2 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **The Base class for Transformer based encoders      Please set causal = True in s** (1 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **feature_lens: int         return updated feature lens.          This used to ret** (1 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **Abstract forward method implementation.** (1 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **If chunk size is a list, we will randomly select a chunk size.** (1 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **Forwarding the inputs through the top embedding layers          Args:** (1 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **Returns offset used when retaining inputs for decoding.          This is essenti** (1 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **ConformerEncoder module.     see original paper for more details:         https:** (1 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **Conformer Forward function          Args:             xs_pad: torch.Tensor** (1 connections) — `python/sglang/srt/models/phi4mm_audio.py`
- **For a given tensor with shape of (N, T, D), if sequence length T is     longer t** (1 connections) — `python/sglang/srt/models/phi4mm_utils.py`

## Relationships

- [[Community 181]] (14 shared connections)
- [[Community 874]] (3 shared connections)
- [[Community 439]] (2 shared connections)
- [[Community 9602]] (2 shared connections)
- [[DeepSeek MLA Attention & MoE]] (2 shared connections)
- [[Community 437]] (1 shared connections)
- [[Community 394]] (1 shared connections)

## Source Files

- `python/sglang/srt/models/phi4mm_audio.py`
- `python/sglang/srt/models/phi4mm_utils.py`

## Audit Trail

- EXTRACTED: 70 (77%)
- INFERRED: 21 (23%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*