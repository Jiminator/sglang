# Community 483

> 12 nodes

## Key Concepts

- **utils.py** (7 connections) — `python/sglang/srt/layers/rotary_embedding/utils.py`
- **Tensor** (5 connections) — `python/sglang/srt/layers/rotary_embedding/utils.py`
- **apply_rotary_pos_emb_native()** (5 connections) — `python/sglang/srt/layers/rotary_embedding/utils.py`
- **apply_rotary_pos_emb_npu()** (4 connections) — `python/sglang/srt/layers/rotary_embedding/utils.py`
- **rotate_neox()** (3 connections) — `python/sglang/srt/layers/rotary_embedding/utils.py`
- **apply_rotary_emb()** (3 connections) — `python/sglang/srt/layers/rotary_embedding/utils.py`
- **rotate_half()** (3 connections) — `python/sglang/srt/layers/rotary_embedding/utils.py`
- **rotate_gptj()** (2 connections) — `python/sglang/srt/layers/rotary_embedding/utils.py`
- **Primitive rotary embedding ops: _rotate_neox, _rotate_gptj, _apply_rotary_emb, a** (1 connections) — `python/sglang/srt/layers/rotary_embedding/utils.py`
- **Args:         x: [num_tokens, num_heads, head_size]         cos: [num_tokens, he** (1 connections) — `python/sglang/srt/layers/rotary_embedding/utils.py`
- **Rotates half the hidden dims of the input.** (1 connections) — `python/sglang/srt/layers/rotary_embedding/utils.py`
- **Ascend implementation equivalent to apply_rotary_pos_emb_native.      Args:** (1 connections) — `python/sglang/srt/layers/rotary_embedding/utils.py`

## Relationships

- [[Community 242]] (2 shared connections)

## Source Files

- `python/sglang/srt/layers/rotary_embedding/utils.py`

## Audit Trail

- EXTRACTED: 34 (94%)
- INFERRED: 2 (6%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*