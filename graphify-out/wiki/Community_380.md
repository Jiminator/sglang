# Community 380

> 16 nodes

## Key Concepts

- **Sm100ChunkHKernel** (8 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **.kernel()** (7 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **.__call__()** (6 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **._make_bf16_tma_args()** (5 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **Tensor** (5 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **fence_before_tma_store()** (4 connections) — `python/sglang/srt/layers/attention/cute_utils/__init__.py`
- **._make_h_tma_args()** (4 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **.compile()** (3 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **.__init__()** (2 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **Numeric** (2 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **TmaCopyOp** (2 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **Constexpr** (1 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **CUstream** (1 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **CopyAtom** (1 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **ComposedLayout** (1 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`
- **For each sequence, compute the chunk recurrent update.      The input V tile is** (1 connections) — `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`

## Relationships

- [[Community 414]] (3 shared connections)
- [[Community 521]] (2 shared connections)
- [[Community 367]] (1 shared connections)
- [[Community 328]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/cute_utils/__init__.py`
- `python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/kernel_h.py`

## Audit Trail

- EXTRACTED: 48 (91%)
- INFERRED: 5 (9%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*