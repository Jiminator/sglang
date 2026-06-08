# Community 221

> 31 nodes

## Key Concepts

- **triton_mla_kernels_decode_dsv4.py** (21 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **triton_mla_kernels_decode_common.py** (15 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **_bucket_total_tokens()** (14 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **_triton_sparse_attn_decode_dsv4_impl()** (9 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **fused_gather_dequant_fp8_dsv4()** (7 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **run_unified_attention()** (6 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **gather_dequant_fp8_dsv4()** (6 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **run_splitk_attention()** (6 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_splitk.py`
- **_get_workload_size_category()** (5 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **run_chunked_attention_triton()** (5 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **run_splitk_unified_attention()** (5 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **_launch_gather_dequant_one_dsv4()** (5 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **_prepare_kv_cache_flat()** (4 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **truly_fused_gather_dequant_fp8_dsv4()** (4 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **Tensor** (3 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **Common utilities and attention kernels for Triton MLA Decode.  This module conta** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **Round total_tokens up to the nearest power of 2 for autotune key stability.** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **Compute workload size category for autotune key.     Returns:         0: small (** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **Run unified attention with single KV buffer.      Run unified sparse decode atte** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **Chunked attention using Triton kernels with cross-chunk softmax merging.** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **Run split-K attention for large topk cases.** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- **Triton MLA Decode Kernels for DSV4 (d_qk=512).  This module contains DSV4-specif** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **Unified DSV4 gather+dequant with optional topk_length mask.** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **Helper to prepare KV cache for gather operations.      Returns: (kv_flat, num_bl** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- **Helper to launch gather+dequant kernel for one KV cache (main or extra).      Th** (1 connections) — `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- *... and 6 more nodes in this community*

## Relationships

- [[Community 247]] (13 shared connections)
- [[Community 1648]] (4 shared connections)
- [[Community 4267]] (3 shared connections)
- [[Community 4265]] (3 shared connections)
- [[Community 283]] (2 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`
- `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`
- `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_splitk.py`

## Audit Trail

- EXTRACTED: 131 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*