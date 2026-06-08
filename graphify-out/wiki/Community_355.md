# Community 355

> 18 nodes

## Key Concepts

- **.patch()** (9 connections) — `python/sglang/srt/utils/patch_tokenizer.py`
- **weak_ref_tensors()** (7 connections) — `python/sglang/srt/compilation/weak_ref_tensor.py`
- **patch_tokenizer.py** (6 connections) — `python/sglang/srt/utils/patch_tokenizer.py`
- **patch_tokenizer()** (4 connections) — `python/sglang/srt/utils/patch_tokenizer.py`
- **.__call__()** (3 connections) — `python/sglang/srt/compilation/npu_piecewise_backend.py`
- **.__init__()** (3 connections) — `python/sglang/srt/distributed/device_communicators/shm_broadcast.py`
- **_weak_ref_if_tensor()** (3 connections) — `python/sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py`
- **_SpecialTokensCachePatcher** (3 connections) — `python/sglang/srt/utils/patch_tokenizer.py`
- **unpatch_tokenizer()** (2 connections) — `python/sglang/srt/utils/patch_tokenizer.py`
- **_is_kimi_tiktoken_tokenizer()** (2 connections) — `python/sglang/srt/utils/patch_tokenizer.py`
- **.unpatch()** (2 connections) — `python/sglang/srt/utils/patch_tokenizer.py`
- **_make_cached_property()** (2 connections) — `python/sglang/srt/utils/patch_tokenizer.py`
- **weak_ref_tensor.py** (1 connections) — `python/sglang/srt/compilation/weak_ref_tensor.py`
- **Tensor** (1 connections) — `python/sglang/srt/compilation/weak_ref_tensor.py`
- **Any** (1 connections) — `python/sglang/srt/compilation/weak_ref_tensor.py`
- **Convenience function to create weak references to tensors,     for single tensor** (1 connections) — `python/sglang/srt/compilation/weak_ref_tensor.py`
- **A shared memory ring buffer implementation for broadcast communication.** (1 connections) — `python/sglang/srt/distributed/device_communicators/shm_broadcast.py`
- **Return a weak-ref tensor view (shared storage, no refcount) for tensors;     pas** (1 connections) — `python/sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py`

## Relationships

- [[Community 311]] (2 shared connections)
- [[Community 289]] (2 shared connections)
- [[Community 154]] (1 shared connections)
- [[Community 313]] (1 shared connections)
- [[Breakable CUDA Graph (TBO)]] (1 shared connections)
- [[Community 39]] (1 shared connections)
- [[Community 124]] (1 shared connections)
- [[Community 383]] (1 shared connections)

## Source Files

- `python/sglang/srt/compilation/npu_piecewise_backend.py`
- `python/sglang/srt/compilation/weak_ref_tensor.py`
- `python/sglang/srt/distributed/device_communicators/shm_broadcast.py`
- `python/sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py`
- `python/sglang/srt/utils/patch_tokenizer.py`

## Audit Trail

- EXTRACTED: 38 (73%)
- INFERRED: 14 (27%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*