# Community 439

> 14 nodes

## Key Concepts

- **NemoConvSubsampling** (22 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **.forward()** (4 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **.conv_split_by_channel()** (4 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **.conv_split_by_batch()** (3 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **.channel_chunked_conv()** (3 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **.get_sampling_frames()** (1 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **.get_streaming_cache_size()** (1 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **.reset_parameters()** (1 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **.change_subsampling_conv_chunking_factor()** (1 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **Convlutional subsampling module, taken from NeMo ASR     (https://github.com/NVI** (1 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **Forward method for NeMo subsampling.          Args:             x[Batch, Time, F** (1 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **Tries to split input by batch, run conv and concat results** (1 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **For dw convs, tries to split input by time, run conv and concat         results** (1 connections) — `python/sglang/srt/models/phi4mm_utils.py`
- **Performs channel chunked convolution** (1 connections) — `python/sglang/srt/models/phi4mm_utils.py`

## Relationships

- [[Community 181]] (8 shared connections)
- [[Community 270]] (2 shared connections)
- [[Community 874]] (1 shared connections)
- [[Community 394]] (1 shared connections)
- [[Community 438]] (1 shared connections)

## Source Files

- `python/sglang/srt/models/phi4mm_utils.py`

## Audit Trail

- EXTRACTED: 34 (76%)
- INFERRED: 11 (24%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*