# Community 268

> 24 nodes

## Key Concepts

- **gemma4_fused_ops.py** (12 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **constexpr** (6 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **Tensor** (5 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **gemma_rmsnorm_residual_scalar()** (4 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **_gemma_qkv_rmsnorm_kernel()** (4 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **gemma_qkv_rmsnorm()** (4 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **gemma_dual_rmsnorm_residual_scalar()** (4 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **_gemma_rmsnorm_residual_kernel()** (3 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **_gemma_dual_rmsnorm_residual_kernel()** (3 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **_gemma_qkv_rmsnorm_store()** (3 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **_gemma_routing_post_topk_kernel()** (3 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **gemma_routing_post_topk()** (3 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **gemma4_fused_routing()** (3 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **_gemma4_routing_kernel()** (2 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **Fused triton kernels for Gemma4 decoder layer operations.  Fuses standard RMSNor** (1 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **Fused kernel: out = rmsnorm(x, w) + residual [* scalar]      When HAS_SCALAR is** (1 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **Fused (rmsnorm(x) + residual) * scalar.** (1 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **Fused: out = (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual) * scalar** (1 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **Fused per-head RMSNorm for Q, K, V.      The same kernel supports two launch sha** (1 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **In-place fused RMSNorm on Q, K, V for Gemma4 attention.      All three norms com** (1 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **Fused: softmax(topk_logits) * per_expert_scale[topk_ids] → float32 weights, int3** (1 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **Fused softmax + scale-gather + casts for Gemma4 routing.      Replaces: softmax(** (1 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **Fused (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual) * scalar.** (1 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`
- **One-launch Gemma4 router.      Args:         gating_output: [T, E] router logits** (1 connections) — `python/sglang/srt/layers/gemma4_fused_ops.py`

## Relationships

- [[Community 34]] (3 shared connections)

## Source Files

- `python/sglang/srt/layers/gemma4_fused_ops.py`

## Audit Trail

- EXTRACTED: 66 (96%)
- INFERRED: 3 (4%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*