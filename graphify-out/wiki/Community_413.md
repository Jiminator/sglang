# Community 413

> 15 nodes

## Key Concepts

- **chunk_kda_fwd_intra()** (6 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra.py`
- **chunk_kda_fwd_intra_token_parallel()** (5 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra_token_parallel.py`
- **chunk_intra.py** (3 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra.py`
- **chunk_kda_fwd_kernel_inter_solve_fused()** (3 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra.py`
- **constexpr** (2 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra.py`
- **chunk_kda_fwd_kernel_intra_sub_chunk()** (2 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra.py`
- **chunk_intra_token_parallel.py** (2 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra_token_parallel.py`
- **chunk_kda_fwd_kernel_intra_token_parallel()** (2 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra_token_parallel.py`
- **Tensor** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra.py`
- **LongTensor** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra.py`
- **Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.     Prerequis** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra.py`
- **constexpr** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra_token_parallel.py`
- **Tensor** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra_token_parallel.py`
- **LongTensor** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra_token_parallel.py`
- **Token-parallel implementation: each token gets its own thread block.     Support** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_intra_token_parallel.py`

## Relationships

- [[Community 487]] (1 shared connections)
- [[Community 258]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/fla/chunk_intra.py`
- `python/sglang/srt/layers/attention/fla/chunk_intra_token_parallel.py`

## Audit Trail

- EXTRACTED: 28 (88%)
- INFERRED: 4 (12%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*