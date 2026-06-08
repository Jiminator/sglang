# Community 333

> 20 nodes

## Key Concepts

- **deepseek_mla_correction.py** (11 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **_get_state()** (7 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **_kv_b_two_stream_state()** (6 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **is_kv_b_lora_active()** (5 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **apply_q_correction()** (5 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **apply_v_correction()** (5 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **kv_b_lora_q_apply()** (5 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **kv_b_lora_v_apply()** (5 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **Tensor** (4 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **kv_b_lora_q_prepare()** (4 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **kv_b_lora_v_prepare()** (4 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **_ensure_step_kernels()** (2 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **LoRA correction for absorbed-MLA ``kv_b_proj``.  The absorbed-MLA path in ``Deep** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **Cheap precondition check used at call sites in the attention forward     to skip** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **LoRA correction for the absorbed ``q_nope @ w_kc`` path.      Computes ``q_nope_** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **LoRA correction for the absorbed ``attn_output @ w_vc`` path.      Computes ``at** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **Fork the q-correction A-step onto the side stream (``step_a_q`` reads only     `** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **Finish the q-correction: two-stream (rejoin + B-step) when ``handle`` is     set** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **Fork the v-correction A-step onto the side stream (``step_a_v`` reads only     `** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`
- **Finish the v-correction: two-stream (rejoin + B-step) when ``handle`` is     set** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`

## Relationships

- [[Community 363]] (4 shared connections)
- [[NCCL Symmetric Memory]] (2 shared connections)
- [[DeepSeek MLA Attention & MoE]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/trtllm_lora_temp/deepseek_mla_correction.py`

## Audit Trail

- EXTRACTED: 64 (90%)
- INFERRED: 7 (10%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*