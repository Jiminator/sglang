# Community 264

> 24 nodes

## Key Concepts

- **flash_mla_sparse_decode_triton()** (7 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **flash_mla_sm120_triton.py** (6 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **_tiled_sparse_decode_kernel()** (6 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **flash_mla_with_kvcache_sm120()** (5 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120.py`
- **flash_mla_sm120.py** (4 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120.py`
- **_run_triton_sparse_decode()** (4 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **Tensor** (4 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **_merge_partial_attn()** (4 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **_apply_attn_sink()** (4 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **_gather_and_dequant()** (3 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120.py`
- **_sm120_sparse_decode_fwd()** (3 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120.py`
- **SM120 FlashMLA sparse decode implementation.  On SM120 (Blackwell Desktop / RTX** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120.py`
- **Gather KV entries from the paged buffer using correct page-internal addressing.** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120.py`
- **SM120 FlashMLA sparse decode entry point.      Dispatches to the Triton kernel (** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120.py`
- **float32** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **int32** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **int64** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **constexpr** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **SM120-optimized Triton FlashMLA sparse decode kernel — Tiled V2.  Replaces V1's** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **Tiled sparse decode: vectorized gather + QK + softmax + V accumulation.      Gri** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **Run the tiled Triton sparse decode kernel on one paged KV cache.** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **Merge two attention outputs using LSE-weighted combination.      out: [B, 1, H,** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **Apply attention sink normalization.      The sink adds to the softmax denominato** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`
- **SM120-optimized sparse MLA decode using tiled Triton kernel.      Processes SWA** (1 connections) — `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`

## Relationships

- [[Community 110]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/flash_mla_sm120.py`
- `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`

## Audit Trail

- EXTRACTED: 60 (95%)
- INFERRED: 3 (5%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*