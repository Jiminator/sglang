# Community 73

> 84 nodes

## Key Concepts

- **Tensor** (16 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **utils.py** (16 connections) — `python/sglang/srt/layers/attention/utils.py`
- **CutlassMLABackend** (13 connections) — `python/sglang/srt/layers/attention/cutlass_mla_backend.py`
- **ForwardBatch** (11 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **.forward_extend()** (11 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **get_num_kv_index_blocks_flashmla()** (10 connections) — `python/sglang/srt/layers/attention/triton_ops/kv_indices.py`
- **RadixAttention** (10 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **.init_forward_metadata()** (9 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **.forward_decode()** (9 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **CutlassMLADecodeMetadata** (7 connections) — `python/sglang/srt/layers/attention/cutlass_mla_backend.py`
- **Tensor** (7 connections) — `python/sglang/srt/layers/attention/cutlass_mla_backend.py`
- **._run_decode_kernel()** (7 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **constexpr** (7 connections) — `python/sglang/srt/layers/attention/utils.py`
- **ForwardBatch** (6 connections) — `python/sglang/srt/layers/attention/cutlass_mla_backend.py`
- **._calc_padded_blocks()** (6 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **._create_block_kv_indices()** (6 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **._apply_cuda_graph_metadata()** (6 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **._run_prefill_kernel()** (6 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **forward_extend_vectorized_5d()** (5 connections) — `python/sglang/srt/layers/attention/aiter_utils.py`
- **Tensor** (5 connections) — `python/sglang/srt/layers/attention/aiter_utils.py`
- **.init_cuda_graph_state()** (5 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **.init_forward_metadata_out_graph()** (5 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **.init_forward_metadata()** (5 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **Tensor** (5 connections) — `python/sglang/srt/layers/attention/utils.py`
- **mla_quantize_and_rope_for_fp8()** (5 connections) — `python/sglang/srt/layers/attention/utils.py`
- *... and 59 more nodes in this community*

## Relationships

- [[Aiter Attention Backend]] (37 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (14 shared connections)
- [[DeepSeek MLA Attention & MoE]] (10 shared connections)
- [[Vision-Language Model Configs]] (10 shared connections)
- [[Community 43]] (4 shared connections)
- [[Disaggregation Utils & Cache Tests]] (2 shared connections)
- [[Community 67]] (1 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)
- [[Community 48]] (1 shared connections)
- [[Community 49]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/aiter_utils.py`
- `python/sglang/srt/layers/attention/attention_registry.py`
- `python/sglang/srt/layers/attention/cutlass_mla_backend.py`
- `python/sglang/srt/layers/attention/triton_ops/kv_indices.py`
- `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- `python/sglang/srt/layers/attention/utils.py`

## Audit Trail

- EXTRACTED: 267 (80%)
- INFERRED: 68 (20%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*