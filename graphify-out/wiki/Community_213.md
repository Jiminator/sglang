# Community 213

> 32 nodes

## Key Concepts

- **topk.py** (40 connections) — `python/sglang/srt/layers/moe/topk.py`
- **Tensor** (26 connections) — `python/sglang/srt/layers/moe/topk.py`
- **_post_process_topk_ids()** (11 connections) — `python/sglang/srt/layers/moe/topk.py`
- **TopKConfig** (7 connections) — `python/sglang/srt/layers/moe/topk.py`
- **biased_grouped_topk_gpu()** (7 connections) — `python/sglang/srt/layers/moe/topk.py`
- **_remap_topk_for_deepep()** (7 connections) — `python/sglang/srt/layers/moe/topk.py`
- **.__init__()** (6 connections) — `python/sglang/srt/layers/moe/topk.py`
- **_mask_topk_ids_padded_region()** (4 connections) — `python/sglang/srt/layers/moe/topk.py`
- **_zero_topk_weights_padded_region()** (4 connections) — `python/sglang/srt/layers/moe/topk.py`
- **_biased_grouped_topk_postprocess()** (4 connections) — `python/sglang/srt/layers/moe/topk.py`
- **fused_topk_deepseek()** (3 connections) — `python/sglang/srt/layers/moe/topk.py`
- **fused_topk_torch_native()** (3 connections) — `python/sglang/srt/layers/moe/topk.py`
- **fused_topk_softmax_torch_raw_logits()** (3 connections) — `python/sglang/srt/layers/moe/topk.py`
- **fused_topk_cpu()** (3 connections) — `python/sglang/srt/layers/moe/topk.py`
- **kimi_k2_biased_topk_impl()** (3 connections) — `python/sglang/srt/layers/moe/topk.py`
- **biased_topk_impl()** (3 connections) — `python/sglang/srt/layers/moe/topk.py`
- **biased_topk_jit_kernel_impl()** (3 connections) — `python/sglang/srt/layers/moe/topk.py`
- **biased_grouped_topk_impl()** (3 connections) — `python/sglang/srt/layers/moe/topk.py`
- **QuantizationConfig** (2 connections) — `python/sglang/srt/layers/moe/topk.py`
- **grouped_topk_gpu()** (2 connections) — `python/sglang/srt/layers/moe/topk.py`
- **grouped_topk_cpu()** (2 connections) — `python/sglang/srt/layers/moe/topk.py`
- **is_power_of_two()** (2 connections) — `python/sglang/srt/layers/moe/topk.py`
- **biased_grouped_topk_cpu()** (2 connections) — `python/sglang/srt/layers/moe/topk.py`
- **_kimi_k2_moe_fused_gate()** (2 connections) — `python/sglang/srt/layers/moe/topk.py`
- **_moe_fused_gate()** (1 connections) — `python/sglang/srt/layers/moe/topk.py`
- *... and 7 more nodes in this community*

## Relationships

- [[Community 396]] (16 shared connections)
- [[Community 395]] (6 shared connections)
- [[Context-Parallel Attention]] (4 shared connections)
- [[Community 477]] (3 shared connections)
- [[DeepSeek MLA Attention & MoE]] (3 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (3 shared connections)
- [[Community 526]] (3 shared connections)
- [[Community 107]] (1 shared connections)
- [[Compressed-Tensors Quant Linear]] (1 shared connections)
- [[Community 454]] (1 shared connections)
- [[Community 833]] (1 shared connections)
- [[Community 96]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/moe/topk.py`

## Audit Trail

- EXTRACTED: 146 (91%)
- INFERRED: 14 (9%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*