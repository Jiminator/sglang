# Community 205

> 34 nodes

## Key Concepts

- **get_pdl_launch_metadata()** (13 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kernel_utils.py`
- **kv_b_lora_absorbed.py** (12 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **step_a_q_fwd()** (8 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **step_b_q_fwd()** (8 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **step_a_v_fwd()** (8 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **step_b_v_fwd()** (8 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **LoRABatchInfo** (7 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **_num_segments()** (6 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **_max_segment_len()** (6 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **_segment_grid_size()** (6 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **sgemm_lora_a_fwd()** (5 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_a.py`
- **constexpr** (4 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **Tensor** (4 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **_resolve_token_positions()** (3 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kernel_utils.py`
- **sgemm_lora_a.py** (3 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_a.py`
- **_sgemm_lora_a_kernel()** (3 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_a.py`
- **kernel_utils.py** (2 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kernel_utils.py`
- **_step_a_q_kernel()** (2 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **_step_b_q_kernel()** (2 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **_step_a_v_kernel()** (2 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **_step_b_v_kernel()** (2 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- **_num_sms()** (2 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_a.py`
- **constexpr** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kernel_utils.py`
- **Return (ENABLE_PDL constexpr value, extra launch kwargs) for LoRA kernels.** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kernel_utils.py`
- **Map logical segment offsets to physical token positions.      When SORTED_BY_ADA** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kernel_utils.py`
- *... and 9 more nodes in this community*

## Relationships

- [[Community 182]] (3 shared connections)
- [[Community 880]] (1 shared connections)
- [[Community 881]] (1 shared connections)
- [[Community 882]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kernel_utils.py`
- `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/kv_b_lora_absorbed.py`
- `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_a.py`

## Audit Trail

- EXTRACTED: 112 (88%)
- INFERRED: 16 (12%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*