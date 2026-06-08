# Community 148

> 44 nodes

## Key Concepts

- **batch_invariant_ops.py** (28 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **__init__.py** (11 connections) — `python/sglang/srt/batch_invariant_ops/__init__.py`
- **matmul_persistent()** (9 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **get_device_core_count()** (9 connections) — `python/sglang/srt/utils/common.py`
- **Tensor** (8 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **rms_norm_batch_invariant()** (8 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **.forward_cuda()** (7 connections) — `python/sglang/srt/layers/layernorm.py`
- **mean_dim()** (6 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **.forward_xpu()** (6 connections) — `python/sglang/srt/layers/layernorm.py`
- **constexpr** (5 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **log_softmax()** (5 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **is_batch_invariant_mode_enabled()** (5 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **enable_batch_invariant_mode()** (5 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **_matmul_persistent_triton()** (4 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **_matmul_persistent_deepgemm()** (4 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **bmm_kernel_persistent()** (4 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **set_batch_invariant_mode()** (4 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **_compute_pid()** (3 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **matmul_kernel_persistent()** (3 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **_log_softmax_kernel()** (3 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **mean_kernel()** (3 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **mean_batch_invariant()** (3 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **_rms_norm_kernel()** (3 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **_get_or_make_ones()** (3 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- **_rms_norm_aten_compat()** (3 connections) — `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- *... and 19 more nodes in this community*

## Relationships

- [[Community 117]] (8 shared connections)
- [[Community 42]] (4 shared connections)
- [[DeepSeek MLA Attention & MoE]] (2 shared connections)
- [[Context-Parallel Attention]] (2 shared connections)
- [[Community 47]] (1 shared connections)
- [[Community 96]] (1 shared connections)
- [[Hybrid Attention Backend]] (1 shared connections)
- [[Community 234]] (1 shared connections)
- [[Community 480]] (1 shared connections)
- [[Community 144]] (1 shared connections)
- [[Aiter Attention Backend]] (1 shared connections)
- [[Community 48]] (1 shared connections)

## Source Files

- `python/sglang/srt/batch_invariant_ops/__init__.py`
- `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- `python/sglang/srt/layers/layernorm.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 159 (86%)
- INFERRED: 25 (14%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*