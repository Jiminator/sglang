# Community 482

> 12 nodes

## Key Concepts

- **trtllm_fp8_kv_kernel.py** (5 connections) — `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`
- **fused_fp8_set_kv_buffer()** (5 connections) — `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`
- **_process_kv_tensor()** (4 connections) — `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`
- **_fused_fp8_set_kv_buffer_kernel()** (4 connections) — `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`
- **_naive_fp8_set_kv_buffer()** (4 connections) — `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`
- **constexpr** (2 connections) — `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`
- **Tensor** (2 connections) — `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`
- **Fused FP8 quantization + paged KV cache write kernel for TRTLLM MHA backend.  Th** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`
- **Process a block of heads for a single K or V tensor.** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`
- **Fused FP8 quantization + paged KV cache write kernel.      Each program processe** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`
- **Python wrapper for the fused FP8 quantization + paged KV cache write kernel.** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`
- **Naive fallback implementation that mimics the original set_kv_buffer logic.** (1 connections) — `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`

## Relationships

- [[Aiter Attention Backend]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`

## Audit Trail

- EXTRACTED: 30 (97%)
- INFERRED: 1 (3%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*