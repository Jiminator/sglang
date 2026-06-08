# Community 388

> 16 nodes

## Key Concepts

- **mxfp4_moe_sm120_triton.py** (6 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **_mxfp4_slot_gemv_kernel()** (6 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **_dequant_fp4_lut()** (4 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **_mxfp4_gemm_kernel()** (4 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **mxfp4_moe_forward_triton()** (4 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **mxfp4_gemm_triton()** (3 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **constexpr** (2 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **Tensor** (2 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **int32** (1 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **int64** (1 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **SM120-optimized Triton MXFP4 MoE kernel — CUDA graph compatible.  Replaces the P** (1 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **Decode a 4-bit FP4 E2M1 nibble to float32 using arithmetic.** (1 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **Per-slot fused MXFP4 dequant + GEMV.      Grid: (num_slots, cdiv(N, BLOCK_N))** (1 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **Fused MXFP4 dequant + GEMM: C = A @ dequant(B_packed, B_scale).T** (1 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **Triton fused MXFP4 dequant + GEMM: output = A @ dequant(B).T      Kept for stand** (1 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`
- **SM120-optimized MXFP4 MoE forward — CUDA graph compatible.      Uses per-slot GE** (1 connections) — `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`

## Relationships

- [[Weight Loading & EPLB]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py`

## Audit Trail

- EXTRACTED: 38 (97%)
- INFERRED: 1 (3%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*