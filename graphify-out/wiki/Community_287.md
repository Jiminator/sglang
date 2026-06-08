# Community 287

> 22 nodes

## Key Concepts

- **TorchFlexAttnBackend** (16 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **ForwardBatch** (7 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **Tensor** (6 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **RadixAttention** (6 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **ModelRunner** (5 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **torch_flex_backend.py** (4 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **._run_flex_forward_extend()** (4 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **._run_flex_forward_decode()** (4 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **.forward_extend()** (4 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **.forward_decode()** (4 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **.init_forward_metadata()** (3 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **create_flex_attention_backend()** (2 connections) — `python/sglang/srt/layers/attention/attention_registry.py`
- **.__init__()** (2 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **._causal_mask()** (1 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **._decode_mask()** (1 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **.support_triton()** (1 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **Init the metadata for a forward pass.** (1 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **Run the extend forward by using torch flex attention op.          Args:** (1 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **Run the decode forward by using torch flex attention op.          Args:** (1 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **# TODO: find a more elegant way to save memory** (1 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **# TODO: this loop process a sequence per iter, this is inefficient.** (1 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`
- **# TODO: this loop process a sequence per iter, this is inefficient.** (1 connections) — `python/sglang/srt/layers/attention/torch_flex_backend.py`

## Relationships

- [[Aiter Attention Backend]] (6 shared connections)
- [[Model Configs & Pooler]] (5 shared connections)
- [[DeepSeek MLA Attention & MoE]] (5 shared connections)
- [[Vision-Language Model Configs]] (5 shared connections)
- [[Community 67]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/attention_registry.py`
- `python/sglang/srt/layers/attention/torch_flex_backend.py`

## Audit Trail

- EXTRACTED: 54 (71%)
- INFERRED: 22 (29%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*