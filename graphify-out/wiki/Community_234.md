# Community 234

> 28 nodes

## Key Concepts

- **XGrammarGrammar** (20 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **Tensor** (12 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **demo_test()** (6 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **apply_token_bitmask_inplace_triton()** (5 connections) — `python/sglang/srt/constrained/triton_ops/bitmask_ops.py`
- **xgrammar_backend.py** (4 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **.accept_token()** (4 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **.apply_vocab_mask()** (4 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **.apply_vocab_mask()** (4 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **.set_token_filter()** (4 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **set_token_filter_torch()** (3 connections) — `python/sglang/srt/constrained/torch_ops/token_filter_torch_ops.py`
- **apply_token_bitmask_inplace_kernel()** (3 connections) — `python/sglang/srt/constrained/triton_ops/bitmask_ops.py`
- **.rollback()** (3 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **.is_terminated()** (3 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **.jump_and_retokenize()** (3 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **bitmask_ops.py** (2 connections) — `python/sglang/srt/constrained/triton_ops/bitmask_ops.py`
- **.allocate_vocab_mask()** (2 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **.fill_vocab_mask()** (2 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **.move_vocab_mask()** (2 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **.allocate_vocab_mask()** (2 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **.move_vocab_mask()** (2 connections) — `python/sglang/srt/constrained/xgrammar_backend.py`
- **token_filter_torch_ops.py** (1 connections) — `python/sglang/srt/constrained/torch_ops/token_filter_torch_ops.py`
- **Tensor** (1 connections) — `python/sglang/srt/constrained/torch_ops/token_filter_torch_ops.py`
- **constexpr** (1 connections) — `python/sglang/srt/constrained/triton_ops/bitmask_ops.py`
- **Tensor** (1 connections) — `python/sglang/srt/constrained/triton_ops/bitmask_ops.py`
- **Apply a bitmask to logits in-place using Triton. The bitmask is a 01 bitwise com** (1 connections) — `python/sglang/srt/constrained/triton_ops/bitmask_ops.py`
- *... and 3 more nodes in this community*

## Relationships

- [[Community 233]] (13 shared connections)
- [[Community 343]] (3 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (2 shared connections)
- [[Community 253]] (2 shared connections)
- [[Community 47]] (2 shared connections)
- [[Community 148]] (1 shared connections)
- [[Community 480]] (1 shared connections)

## Source Files

- `python/sglang/srt/constrained/torch_ops/token_filter_torch_ops.py`
- `python/sglang/srt/constrained/triton_ops/bitmask_ops.py`
- `python/sglang/srt/constrained/xgrammar_backend.py`

## Audit Trail

- EXTRACTED: 80 (82%)
- INFERRED: 18 (18%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*