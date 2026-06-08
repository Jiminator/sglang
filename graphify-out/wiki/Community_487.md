# Community 487

> 11 nodes

## Key Concepts

- **prepare_chunk_indices()** (20 connections) — `python/sglang/srt/layers/attention/fla/index.py`
- **chunk_gated_delta_rule_fwd_h()** (5 connections) — `python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_delta_h.py`
- **chunk_gated_delta_rule_fwd_h()** (5 connections) — `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`
- **prepare_chunk_offsets()** (5 connections) — `python/sglang/srt/layers/attention/fla/index.py`
- **prepare_lens()** (4 connections) — `python/sglang/srt/layers/attention/fla/index.py`
- **index.py** (3 connections) — `python/sglang/srt/layers/attention/fla/index.py`
- **LongTensor** (3 connections) — `python/sglang/srt/layers/attention/fla/index.py`
- **Tensor** (1 connections) — `python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_delta_h.py`
- **LongTensor** (1 connections) — `python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_delta_h.py`
- **Tensor** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`
- **LongTensor** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`

## Relationships

- [[Community 258]] (5 shared connections)
- [[Community 453]] (2 shared connections)
- [[Community 475]] (2 shared connections)
- [[Community 527]] (2 shared connections)
- [[Community 1638]] (1 shared connections)
- [[Community 1637]] (1 shared connections)
- [[Community 413]] (1 shared connections)
- [[Community 1639]] (1 shared connections)
- [[Community 871]] (1 shared connections)
- [[Community 9574]] (1 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_delta_h.py`
- `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`
- `python/sglang/srt/layers/attention/fla/index.py`

## Audit Trail

- EXTRACTED: 26 (53%)
- INFERRED: 23 (47%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*