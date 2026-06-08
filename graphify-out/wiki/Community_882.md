# Community 882

> 9 nodes

## Key Concepts

- **_sgemm_lora_b_cublas()** (5 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_b.py`
- **sgemm_lora_b_fwd()** (5 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_b.py`
- **sgemm_lora_b.py** (3 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_b.py`
- **_sgemm_lora_b_kernel()** (3 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_b.py`
- **Tensor** (2 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_b.py`
- **LoRABatchInfo** (2 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_b.py`
- **constexpr** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_b.py`
- **Single-adapter dense path: one cuBLAS addmm_ over the full output.      Mirrors** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_b.py`
- **Computes a segmented batched matrix multiplication for the LoRA B matrix     and** (1 connections) — `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_b.py`

## Relationships

- [[Community 205]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/trtllm_lora_temp/triton_ops/sgemm_lora_b.py`

## Audit Trail

- EXTRACTED: 22 (96%)
- INFERRED: 1 (4%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*