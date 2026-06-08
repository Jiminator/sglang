# Community 284

> 23 nodes

## Key Concepts

- **extend_attention.py** (9 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **_get_block_sizes_for_extend_attention()** (4 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **_fwd_kernel_unified()** (4 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **context_attention_fwd()** (4 connections) — `python/sglang/srt/layers/attention/triton_ops/prefill_attention.py`
- **tanh()** (3 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **build_unified_kv_indices()** (3 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **_fwd_kernel()** (3 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **extend_attention_fwd()** (3 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **extend_attention_fwd_unified()** (3 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **_copy_unified_indices_kernel()** (2 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **constexpr** (2 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **redundant_attention()** (2 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **prefill_attention.py** (2 connections) — `python/sglang/srt/layers/attention/triton_ops/prefill_attention.py`
- **_fwd_kernel()** (2 connections) — `python/sglang/srt/layers/attention/triton_ops/prefill_attention.py`
- **Tensor** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **Get block sizes and configuration for extend attention kernels.      Args:** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **Triton kernel to copy indices to unified buffer (parallel per sequence).     Eac** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **Build unified KV indices efficiently:     - Use PyTorch's optimized cumsum (NVID** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **q_extend, k_extend, v_extend, o_extend: contiguous tensors      k_buffer, v_buff** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **Unified 1-stage kernel for deterministic extend attention.     Both prefix and e** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **Unified 1-stage extend attention for deterministic inference.      Args:** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- **constexpr** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/prefill_attention.py`
- **q, k, v: [b * s, head, head_dim]     b_start_loc: [b]     b_seq_len: [b]     out** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/prefill_attention.py`

## Relationships

- [[Community 99]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
- `python/sglang/srt/layers/attention/triton_ops/prefill_attention.py`

## Audit Trail

- EXTRACTED: 52 (95%)
- INFERRED: 3 (5%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*