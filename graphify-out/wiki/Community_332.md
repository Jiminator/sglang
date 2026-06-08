# Community 332

> 20 nodes

## Key Concepts

- **kv_b_lora_absorbed.py** (12 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **LoRABatchInfo** (7 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **step_a_q_fwd()** (7 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **step_b_q_fwd()** (7 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **step_a_v_fwd()** (7 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **step_b_v_fwd()** (7 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **_num_segments()** (6 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **_max_segment_len()** (6 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **_segment_grid_size()** (6 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **constexpr** (4 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **Tensor** (4 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **_step_a_q_kernel()** (2 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **_step_b_q_kernel()** (2 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **_step_a_v_kernel()** (2 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **_step_b_v_kernel()** (2 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **Triton kernels for absorbed-MLA ``kv_b_proj`` LoRA correction.  The absorbed-MLA** (1 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **Step A of the q-side correction.      Args:         q_nope: ``(S, H, qk_nope)``,** (1 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **Step B of the q-side correction, accumulating into ``base_output``.      Args:** (1 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **Step A of the v-side correction.      Args:         attn_output: ``(S, H, kv_lor** (1 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`
- **Step B of the v-side correction, accumulating into ``base_output``.      Args:** (1 connections) — `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`

## Audit Trail

- EXTRACTED: 86 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*