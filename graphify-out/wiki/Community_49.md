# Community 49

> 127 nodes

## Key Concepts

- **Indexer** (51 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- **BaseIndexerMetadata** (47 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- **Tensor** (37 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- **get_token_to_kv_pool()** (30 connections) — `python/sglang/srt/model_executor/forward_context.py`
- **is_in_piecewise_cuda_graph()** (25 connections) — `python/sglang/srt/compilation/piecewise_context_manager.py`
- **.forward_cuda()** (21 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- **._get_topk_ragged()** (17 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- **ForwardBatch** (15 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- **ForwardBatch** (12 connections) — `python/sglang/srt/layers/attention/base_attn_backend.py`
- **._get_topk_paged()** (12 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- **Tensor** (11 connections) — `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`
- **dsa_indexer.py** (11 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- **ForwardBatch** (11 connections) — `python/sglang/srt/layers/attention/hybrid_attn_backend.py`
- **._get_topk_ragged_with_cp()** (10 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- **._forward_cuda_k_only()** (9 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- **.forward_indexer()** (9 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- **._select_backend()** (9 connections) — `python/sglang/srt/layers/attention/hybrid_attn_backend.py`
- **forward_context.py** (9 connections) — `python/sglang/srt/model_executor/forward_context.py`
- **get_req_to_token_pool()** (9 connections) — `python/sglang/srt/model_executor/forward_context.py`
- **forward_mla_prepare_npu()** (8 connections) — `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`
- **.forward()** (8 connections) — `python/sglang/srt/layers/attention/base_attn_backend.py`
- **Tensor** (8 connections) — `python/sglang/srt/layers/attention/base_attn_backend.py`
- **RadixAttention** (8 connections) — `python/sglang/srt/layers/attention/base_attn_backend.py`
- **rotate_activation()** (8 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- **._store_index_k_cache()** (8 connections) — `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- *... and 102 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (44 shared connections)
- [[Aiter Attention Backend]] (32 shared connections)
- [[Vision-Language Model Configs]] (17 shared connections)
- [[Community 43]] (14 shared connections)
- [[Hybrid Attention Backend]] (12 shared connections)
- [[Context-Parallel Attention]] (11 shared connections)
- [[Community 66]] (7 shared connections)
- [[Disaggregation Utils & Cache Tests]] (7 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (6 shared connections)
- [[Community 37]] (5 shared connections)
- [[NCCL Symmetric Memory]] (4 shared connections)
- [[Community 354]] (4 shared connections)

## Source Files

- `python/sglang/srt/compilation/piecewise_context_manager.py`
- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`
- `python/sglang/srt/layers/attention/base_attn_backend.py`
- `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- `python/sglang/srt/layers/attention/hybrid_attn_backend.py`
- `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py`
- `python/sglang/srt/model_executor/forward_context.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/models/deepseek_v2.py`
- `python/sglang/srt/state_capturer/indexer_topk.py`

## Audit Trail

- EXTRACTED: 521 (69%)
- INFERRED: 238 (31%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*