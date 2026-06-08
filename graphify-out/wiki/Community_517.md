# Community 517

> 10 nodes

## Key Concepts

- **per_tensor_quant_mla_fp8()** (8 connections) — `python/sglang/srt/layers/quantization/fp8_kernel.py`
- **rocm_mla_decode_rope.py** (5 connections) — `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py`
- **.forward_absorb_fused_mla_rope_core()** (5 connections) — `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`
- **decode_attention_fwd_grouped_rope()** (4 connections) — `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py`
- **_fwd_grouped_kernel_stage1_rope()** (3 connections) — `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py`
- **tanh()** (2 connections) — `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py`
- **_decode_grouped_att_m_fwd_rope()** (2 connections) — `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py`
- **is_hip()** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py`
- **constexpr** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py`
- **This function quantizes input values to float8 values with tensor-wise quantizat** (1 connections) — `python/sglang/srt/layers/quantization/fp8_kernel.py`

## Relationships

- [[Community 2464]] (3 shared connections)
- [[Community 48]] (2 shared connections)
- [[Community 363]] (2 shared connections)
- [[Community 481]] (1 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)
- [[Community 49]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py`
- `python/sglang/srt/layers/quantization/fp8_kernel.py`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`

## Audit Trail

- EXTRACTED: 22 (69%)
- INFERRED: 10 (31%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*