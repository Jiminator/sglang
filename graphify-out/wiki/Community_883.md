# Community 883

> 9 nodes

## Key Concepts

- **.materialize()** (5 connections) — `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
- **_fused_norm_rope()** (4 connections) — `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
- **fused_kv_materialize.py** (3 connections) — `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
- **_fused_norm_rope_kernel()** (3 connections) — `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
- **Tensor** (2 connections) — `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
- **constexpr** (1 connections) — `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
- **Fused RMSNorm(K) + RoPE(K) materialization. Grid: (total_ctx, num_kv_heads).** (1 connections) — `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
- **Fused RMSNorm + RoPE materialization for a single layer.** (1 connections) — `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
- **Materialize KV cache for all layers using batched projection.** (1 connections) — `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`

## Relationships

- [[CLI Arg Parsing & Deprecation]] (2 shared connections)
- [[Community 47]] (1 shared connections)

## Source Files

- `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`

## Audit Trail

- EXTRACTED: 20 (95%)
- INFERRED: 1 (5%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*