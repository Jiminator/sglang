# Community 424

> 15 nodes

## Key Concepts

- **nvfp4_gemm_swiglu_nvfp4_quant.py** (12 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **nvfp4_gemm_swiglu_nvfp4_quant()** (8 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **swizzle_blockscale_2d()** (5 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **cvt_sf_MKL_to_M32x4xrm_K4xrk_L()** (3 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **cvt_sf_M32x4xrm_K4xrk_L_to_MKL()** (3 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **_round_up()** (3 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **_get_compiled()** (3 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)x** (1 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **Convert scale factor tensor from mma specification M(32x4xrest_m)xK(4xrest_k)xL** (1 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **Standard CUTLASS block-scale 2D swizzle: pad to (128, 4) tiles then     permute** (1 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **NVFP4 GEMM fused with SwiGLU and NVFP4 output quantization.      Args:         a** (1 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **# TODO: Add 64 and 192 support** (1 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **# TODO: round up to 128, it is prepared for supporting N=64 or 192.** (1 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **# TODO: Currently we don't support m major output for Float4E2M1FN** (1 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- **# TODO: Add tile_n=64 and tile_n=192 support** (1 connections) — `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`

## Relationships

- [[Community 118]] (9 shared connections)
- [[Weight Loading & EPLB]] (1 shared connections)
- [[DeepSeek MLA Attention & MoE]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`

## Audit Trail

- EXTRACTED: 43 (96%)
- INFERRED: 2 (4%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*