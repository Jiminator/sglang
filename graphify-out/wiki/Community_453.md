# Community 453

> 13 nodes

## Key Concepts

- **safe_exp()** (7 connections) — `python/sglang/srt/layers/attention/fla/op.py`
- **chunk_gated_delta_rule_fwd_kernel_h_blockdim64_k_loop()** (4 connections) — `python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_delta_h.py`
- **chunk_gated_delta_rule_fwd_kernel_h_blockdim64()** (3 connections) — `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`
- **chunk_fwd_kernel_o()** (3 connections) — `python/sglang/srt/layers/attention/fla/chunk_o.py`
- **op.py** (3 connections) — `python/sglang/srt/layers/attention/fla/op.py`
- **chunk_delta_h.py** (2 connections) — `python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_delta_h.py`
- **chunk_delta_h.py** (2 connections) — `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`
- **gather()** (2 connections) — `python/sglang/srt/layers/attention/fla/op.py`
- **make_tensor_descriptor()** (2 connections) — `python/sglang/srt/layers/attention/fla/op.py`
- **constexpr** (1 connections) — `python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_delta_h.py`
- **constexpr** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`
- **constexpr** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_o.py`
- **Gather operation that works when tl.gather is not supported.         This is a f** (1 connections) — `python/sglang/srt/layers/attention/fla/op.py`

## Relationships

- [[Community 487]] (2 shared connections)
- [[Community 475]] (1 shared connections)
- [[Community 1638]] (1 shared connections)
- [[Community 1637]] (1 shared connections)
- [[Community 1639]] (1 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_delta_h.py`
- `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`
- `python/sglang/srt/layers/attention/fla/chunk_o.py`
- `python/sglang/srt/layers/attention/fla/op.py`

## Audit Trail

- EXTRACTED: 21 (66%)
- INFERRED: 11 (34%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*