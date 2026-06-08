# Community 283

> 23 nodes

## Key Concepts

- **triton_mla_kernels_decode_fused.py** (19 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **constexpr** (8 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **_process_kv_block_aggressive()** (7 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **_fused_gather_attn_dsv4_kernel()** (4 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **_fused_gather_attn_dsv4_dual_scope_kernel()** (4 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **_fused_gather_attn_dsv4_dual_scope_splitk_kernel()** (4 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **_fused_gather_attn_dsv4_splitk_kernel()** (4 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **_combine_splitk_kernel()** (3 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **_combine_splitk_kernel_8_optimized()** (3 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **_combine_splitk_kernel_2()** (3 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **_prune_dual_scope_configs()** (2 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **_prune_splitk_configs()** (2 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Fused Gather+Dequant+Attention Kernel for DSV4 (d_qk=512)  This module implement** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Process one block of KV tokens with batch loading.     Key optimization: Load al** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Fused gather+dequant+attention kernel for DSV4.** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Prune configs where BLOCK_H > h_q for the dual-scope kernel.      When BLOCK_H >** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **OPTIMIZED fused gather+dequant+attention kernel for DSV4 with dual scope.      T** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Prune BLOCK_H=16 configs for large batch sizes to avoid CU oversubscription.** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Split-K fused gather+dequant+attention kernel for DSV4 with dual scope.      Thi** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Split-K fused gather+dequant+attention kernel for DSV4.** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Combine partial results from split-K kernel (SPLIT_K=4).** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Optimized combine kernel for split-K=8 with autotuning for BLOCK_H.** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`
- **Combine partial results from split-K kernel (SPLIT_K=2).** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`

## Relationships

- [[Community 247]] (5 shared connections)
- [[Community 221]] (2 shared connections)
- [[Community 4266]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`

## Audit Trail

- EXTRACTED: 74 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*