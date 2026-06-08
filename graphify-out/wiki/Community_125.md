# Community 125

> 52 nodes

## Key Concepts

- **LightningAttentionBackend** (18 connections) — `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- **BailingLinearMetadata** (11 connections) — `python/sglang/srt/layers/attention/linear/linear_metadata.py`
- **ForwardBatch** (10 connections) — `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- **lightning_attn.py** (9 connections) — `python/sglang/srt/layers/attention/linear/lightning_attn.py`
- **BailingLinearKernel** (8 connections) — `python/sglang/srt/layers/attention/linear/lightning_attn.py`
- **Tensor** (8 connections) — `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- **RadixAttention** (8 connections) — `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- **seg_la.py** (8 connections) — `python/sglang/srt/layers/attention/linear/seg_la.py`
- **ModelRunner** (7 connections) — `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- **SegLaMeta** (7 connections) — `python/sglang/srt/layers/attention/linear/seg_la.py`
- **.forward_extend()** (6 connections) — `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- **.forward_decode()** (6 connections) — `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- **constexpr** (6 connections) — `python/sglang/srt/layers/attention/linear/seg_la.py`
- **constexpr** (5 connections) — `python/sglang/srt/layers/attention/linear/lightning_attn.py`
- **._linear_attention_entry()** (5 connections) — `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- **.prepare_mixed()** (5 connections) — `python/sglang/srt/layers/attention/linear/linear_metadata.py`
- **linear_decode_forward_triton()** (4 connections) — `python/sglang/srt/layers/attention/linear/lightning_attn.py`
- **._decode_infer()** (4 connections) — `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- **.prepare_decode()** (4 connections) — `python/sglang/srt/layers/attention/linear/linear_metadata.py`
- **Tensor** (4 connections) — `python/sglang/srt/layers/attention/linear/linear_metadata.py`
- **lightning_attention()** (3 connections) — `python/sglang/srt/layers/attention/linear/lightning_attn.py`
- **_linear_attn_decode_kernel()** (3 connections) — `python/sglang/srt/layers/attention/linear/lightning_attn.py`
- **.jit_linear_forward_prefix()** (3 connections) — `python/sglang/srt/layers/attention/linear/lightning_attn.py`
- **.__init__()** (3 connections) — `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- **._prefill_and_mix_infer()** (3 connections) — `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- *... and 27 more nodes in this community*

## Relationships

- [[Community 86]] (8 shared connections)
- [[Vision-Language Model Configs]] (8 shared connections)
- [[DeepSeek MLA Attention & MoE]] (5 shared connections)
- [[Context-Parallel Attention]] (2 shared connections)
- [[Community 47]] (1 shared connections)
- [[Community 36]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/linear/lightning_attn.py`
- `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- `python/sglang/srt/layers/attention/linear/linear_metadata.py`
- `python/sglang/srt/layers/attention/linear/seg_la.py`

## Audit Trail

- EXTRACTED: 144 (71%)
- INFERRED: 59 (29%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*