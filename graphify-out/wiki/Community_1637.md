# Community 1637

> 8 nodes

## Key Concepts

- **chunk_gated_delta_rule_fwd_intra()** (5 connections) — `python/sglang/srt/layers/attention/fla/chunk_fwd.py`
- **chunk_gated_delta_rule_fwd_kkt_solve_kernel()** (4 connections) — `python/sglang/srt/layers/attention/fla/chunk_fwd.py`
- **chunk_fwd.py** (2 connections) — `python/sglang/srt/layers/attention/fla/chunk_fwd.py`
- **constexpr** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_fwd.py`
- **Tensor** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_fwd.py`
- **LongTensor** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_fwd.py`
- **Fused kernel: compute beta * K @ K^T (lower triangular) + solve_tril (I+A)^{-1}** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_fwd.py`
- **r"""     GDN intra-chunk forward: fused kkt + solve_tril + recompute_w_u.      E** (1 connections) — `python/sglang/srt/layers/attention/fla/chunk_fwd.py`

## Relationships

- [[Community 453]] (1 shared connections)
- [[Community 487]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/fla/chunk_fwd.py`

## Audit Trail

- EXTRACTED: 14 (88%)
- INFERRED: 2 (12%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*