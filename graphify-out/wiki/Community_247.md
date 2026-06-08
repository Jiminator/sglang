# Community 247

> 26 nodes

## Key Concepts

- **triton_mla_kernels_decode_optimized.py** (17 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`
- **_triton_sparse_attn_decode_dsv4()** (12 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`
- **fused_gather_attn_decode_dsv4_dual_scope_low_overhead()** (9 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **triton_sparse_attn_decode_dsv4()** (8 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **fused_gather_attn_decode_dsv4_dual_scope()** (8 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **fused_gather_attn_decode_dsv4()** (7 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **compute_token_ranges()** (6 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **_select_split_k()** (5 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **triton_sparse_attn_decode()** (5 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`
- **_fallback_gather_attention()** (5 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`
- **Tensor** (3 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Tensor** (3 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`
- **_should_use_fused_dual_scope()** (3 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`
- **_should_use_fused_nosplitk()** (3 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`
- **Compute token ranges for processing, chunking if buffer would exceed limit.** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **Sparse attention decode for DSV4 (d_qk=512).** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **Fused gather+dequant+attention for DSV4.     Uses Split-K optimization for large** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Fused gather+dequant+attention for DSV4 with dual scope (main + extra).     Uses** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Select optimal split_k based on topk, h_q, and total_tokens.      The split_k pa** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Low-overhead version of fused_gather_attn_decode_dsv4_dual_scope.      This vers** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Optimized Triton MLA Decode Kernels for DeepSeek V4.  This module provides optim** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`
- **Optimized sparse attention decode for DeepSeek V4 (d_qk=512).** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`
- **Determine whether to use fused kernel for dual-scope cases.      Returns True if** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`
- **Determine whether to use the fused no-splitk kernel for large batches.      Kern** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`
- **Optimized sparse attention decode for DeepSeek V4 (d_qk=512).** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`
- *... and 1 more nodes in this community*

## Relationships

- [[Community 221]] (13 shared connections)
- [[Community 283]] (5 shared connections)
- [[Community 1648]] (2 shared connections)
- [[Community 4266]] (1 shared connections)
- [[Community 425]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py`

## Audit Trail

- EXTRACTED: 105 (99%)
- INFERRED: 1 (1%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*