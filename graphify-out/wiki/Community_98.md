# Community 98

> 65 nodes

## Key Concepts

- **ForwardBatchDeepSeekMHAMixin** (22 connections) — `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py`
- **Tensor** (20 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **forward_batch_info.py** (18 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **.init_new()** (16 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **ModelRunner** (16 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **ScheduleBatch** (15 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **ContextParallelMetadata** (14 connections) — `python/sglang/srt/layers/utils/cp_utils.py`
- **device** (14 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **MultimodalInputs** (12 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **.prepare_mlp_sync_batch()** (12 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **LogitsProcessorOutput** (11 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **get_attention_dp_rank()** (10 connections) — `python/sglang/srt/layers/dp_attention.py`
- **.is_extend()** (8 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **._compute_mrope_positions()** (8 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **.post_forward_mlp_sync_batch()** (8 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **compute_local_num_token_non_padded()** (7 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **.prepare_chunked_prefix_cache_info()** (6 connections) — `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py`
- **.is_decode()** (6 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **._init_ngram_embedding_info()** (6 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **enable_num_token_non_padded()** (6 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **compute_position()** (6 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **.is_target_verify()** (5 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **.is_draft_extend()** (5 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **._maybe_init_non_generation_fields()** (5 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **.adjust_num_token_non_padded_for_attn_tp()** (5 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- *... and 40 more nodes in this community*

## Relationships

- [[Vision-Language Model Configs]] (21 shared connections)
- [[Aiter Attention Backend]] (20 shared connections)
- [[Context-Parallel Attention]] (14 shared connections)
- [[CLI Arg Parsing & Deprecation]] (14 shared connections)
- [[Breakable CUDA Graph (TBO)]] (8 shared connections)
- [[Batch-Overlap Operations]] (7 shared connections)
- [[Hybrid Attention Backend]] (7 shared connections)
- [[Disaggregation Utils & Cache Tests]] (6 shared connections)
- [[Model Configs & Pooler]] (6 shared connections)
- [[Grammar Manager & HiCache Clear]] (6 shared connections)
- [[DeepSeek MLA Attention & MoE]] (4 shared connections)
- [[NCCL Symmetric Memory]] (3 shared connections)

## Source Files

- `python/sglang/srt/layers/dp_attention.py`
- `python/sglang/srt/layers/utils/cp_utils.py`
- `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/model_executor/triton_ops/position.py`

## Audit Trail

- EXTRACTED: 240 (67%)
- INFERRED: 117 (33%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*