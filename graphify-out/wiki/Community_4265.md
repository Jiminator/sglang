# Community 4265

> 7 nodes

## Key Concepts

- **_gather_dequant_dsv4_kernel()** (3 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **constexpr** (3 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **_gather_dequant_dsv4_kernel_fixed_128()** (3 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **_gather_dequant_dsv4_1d_fused_kernel()** (3 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **Optimized gather + dequant kernel with batched scale loading.** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **Fixed-config gather kernel with BLOCK_TK=128 and batched scale loading.** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **1D fused gather kernel - single launch, no empty blocks.      Grid: (num_main_pi** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`

## Relationships

- [[Community 221]] (3 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`

## Audit Trail

- EXTRACTED: 15 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*