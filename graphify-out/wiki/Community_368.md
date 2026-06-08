# Community 368

> 17 nodes

## Key Concepts

- **causal_conv1d.py** (6 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d.py`
- **causal_conv1d_fn()** (5 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d.py`
- **causal_conv1d_triton.py** (5 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`
- **causal_conv1d_fn()** (5 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`
- **causal_conv1d_update()** (5 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`
- **causal_conv1d_update()** (4 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d.py`
- **_causal_conv1d_fwd_kernel()** (3 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`
- **_get_seq_lens_cpu()** (2 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d.py`
- **Tensor** (2 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d.py`
- **constexpr** (2 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`
- **Tensor** (2 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`
- **_causal_conv1d_update_kernel()** (2 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`
- **x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen         sequences are con** (1 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d.py`
- **x: (batch, dim) or (batch, dim, seqlen)     conv_state: (batch, dim, state_len),** (1 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d.py`
- **int32** (1 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`
- **support varlen + continuous batching when x is 2D tensor      x: (dim,cu_seq_len** (1 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`
- **x: (batch, dim) or (batch, dim, seqlen)         [shape=2: single token predictio** (1 connections) — `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `python/sglang/srt/layers/attention/mamba/causal_conv1d.py`
- `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`

## Audit Trail

- EXTRACTED: 48 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*