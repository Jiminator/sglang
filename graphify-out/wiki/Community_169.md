# Community 169

> 39 nodes

## Key Concepts

- **Tensor** (17 connections) — `python/sglang/srt/layers/sampler.py`
- **Sampler** (15 connections) — `python/sglang/srt/layers/sampler.py`
- **sampler.py** (13 connections) — `python/sglang/srt/layers/sampler.py`
- **SamplingBatchInfo** (12 connections) — `python/sglang/srt/layers/sampler.py`
- **.forward()** (11 connections) — `python/sglang/srt/layers/sampler.py`
- **._sample_from_probs()** (8 connections) — `python/sglang/srt/layers/sampler.py`
- **apply_custom_logit_processor()** (8 connections) — `python/sglang/srt/layers/sampler.py`
- **._preprocess_logits()** (7 connections) — `python/sglang/srt/layers/sampler.py`
- **._attach_logprobs_to_output()** (7 connections) — `python/sglang/srt/layers/sampler.py`
- **.compute_logprobs_only()** (7 connections) — `python/sglang/srt/layers/sampler.py`
- **multinomial_with_seed()** (7 connections) — `python/sglang/srt/layers/sampler.py`
- **._sample_from_logprobs()** (6 connections) — `python/sglang/srt/layers/sampler.py`
- **._sample_from_logits()** (6 connections) — `python/sglang/srt/layers/sampler.py`
- **._forward_ascend_backend()** (6 connections) — `python/sglang/srt/layers/sampler.py`
- **create_sampler()** (6 connections) — `python/sglang/srt/layers/sampler.py`
- **LogitsProcessorOutput** (5 connections) — `python/sglang/srt/layers/sampler.py`
- **top_k_top_p_min_p_sampling_from_probs_torch()** (5 connections) — `python/sglang/srt/layers/sampler.py`
- **sampling_from_probs_torch()** (5 connections) — `python/sglang/srt/layers/sampler.py`
- **._sync_token_ids_across_tp()** (4 connections) — `python/sglang/srt/layers/sampler.py`
- **top_k_top_p_min_p_sampling_from_logits_ascend()** (4 connections) — `python/sglang/srt/layers/sampler.py`
- **get_token_ids_logprobs_batch_optimized()** (4 connections) — `python/sglang/srt/layers/sampler.py`
- **top_p_normalize_probs_torch()** (2 connections) — `python/sglang/srt/layers/sampler.py`
- **Apply custom logit processors.** (1 connections) — `python/sglang/srt/layers/sampler.py`
- **Run a sampler & compute logprobs and update logits_output accordingly.** (1 connections) — `python/sglang/srt/layers/sampler.py`
- **Sample from probability distribution (after softmax).          Used for standard** (1 connections) — `python/sglang/srt/layers/sampler.py`
- *... and 14 more nodes in this community*

## Relationships

- [[Model Configs & Pooler]] (4 shared connections)
- [[Grammar Manager & HiCache Clear]] (4 shared connections)
- [[Community 127]] (3 shared connections)
- [[Community 201]] (2 shared connections)
- [[Context-Parallel Attention]] (2 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (2 shared connections)
- [[NCCL Symmetric Memory]] (1 shared connections)
- [[Hybrid Attention Backend]] (1 shared connections)
- [[DeepSeek MLA Attention & MoE]] (1 shared connections)
- [[Community 484]] (1 shared connections)
- [[Community 47]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/sampler.py`

## Audit Trail

- EXTRACTED: 163 (90%)
- INFERRED: 19 (10%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*