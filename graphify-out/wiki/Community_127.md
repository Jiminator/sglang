# Community 127

> 52 nodes

## Key Concepts

- **LogitsMetadata** (36 connections) — `python/sglang/srt/layers/logits_processor.py`
- **Tensor** (18 connections) — `python/sglang/srt/layers/logits_processor.py`
- **._get_logits()** (15 connections) — `python/sglang/srt/layers/logits_processor.py`
- **.forward()** (14 connections) — `python/sglang/srt/layers/logits_processor.py`
- **logprob.py** (14 connections) — `python/sglang/srt/layers/utils/logprob.py`
- **VocabParallelEmbedding** (12 connections) — `python/sglang/srt/layers/logits_processor.py`
- **Tensor** (12 connections) — `python/sglang/srt/layers/utils/logprob.py`
- **.compute_logprobs_for_multi_item_scoring()** (11 connections) — `python/sglang/srt/layers/logits_processor.py`
- **InputLogprobsResult** (11 connections) — `python/sglang/srt/layers/utils/logprob.py`
- **add_output_logprobs_for_spec_v1()** (11 connections) — `python/sglang/srt/layers/utils/logprob.py`
- **.process_input_logprobs_by_chunk()** (10 connections) — `python/sglang/srt/layers/logits_processor.py`
- **logits_processor.py** (9 connections) — `python/sglang/srt/layers/logits_processor.py`
- **ForwardBatch** (9 connections) — `python/sglang/srt/layers/logits_processor.py`
- **LogprobStage** (9 connections) — `python/sglang/srt/layers/utils/logprob.py`
- **InputLogprobsResult** (8 connections) — `python/sglang/srt/layers/logits_processor.py`
- **LogitsMetadata** (8 connections) — `python/sglang/srt/layers/utils/logprob.py`
- **._compute_lm_head()** (7 connections) — `python/sglang/srt/layers/logits_processor.py`
- **._get_dllm_logits()** (7 connections) — `python/sglang/srt/layers/logits_processor.py`
- **get_top_logprobs()** (7 connections) — `python/sglang/srt/layers/utils/logprob.py`
- **compute_spec_v2_logprobs()** (7 connections) — `python/sglang/srt/layers/utils/logprob.py`
- **.process_input_logprobs()** (6 connections) — `python/sglang/srt/layers/logits_processor.py`
- **._gather_dp_attn_hidden_states()** (6 connections) — `python/sglang/srt/layers/logits_processor.py`
- **get_top_logprobs_prefill()** (6 connections) — `python/sglang/srt/layers/utils/logprob.py`
- **ScheduleBatch** (6 connections) — `python/sglang/srt/layers/utils/logprob.py`
- **EagleVerifyOutput** (6 connections) — `python/sglang/srt/layers/utils/logprob.py`
- *... and 27 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (19 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (16 shared connections)
- [[Hybrid Attention Backend]] (14 shared connections)
- [[Model Configs & Pooler]] (12 shared connections)
- [[CLI Arg Parsing & Deprecation]] (10 shared connections)
- [[Community 34]] (9 shared connections)
- [[Vision-Language Model Configs]] (6 shared connections)
- [[Aiter Attention Backend]] (5 shared connections)
- [[NCCL Symmetric Memory]] (5 shared connections)
- [[Community 169]] (3 shared connections)
- [[Community 107]] (2 shared connections)
- [[Community 98]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/logits_processor.py`
- `python/sglang/srt/layers/utils/logprob.py`

## Audit Trail

- EXTRACTED: 223 (65%)
- INFERRED: 120 (35%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*