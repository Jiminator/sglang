# Community 190

> 36 nodes

## Key Concepts

- **entrypoint.py** (12 connections) — `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
- **mhc.py** (12 connections) — `python/sglang/srt/layers/mhc.py`
- **mhc_fused_post_pre()** (11 connections) — `python/sglang/srt/layers/mhc.py`
- **Tensor** (10 connections) — `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
- **mhc_pre()** (9 connections) — `python/sglang/srt/layers/mhc.py`
- **.hc_pre()** (9 connections) — `python/sglang/srt/models/deepseek_v4.py`
- **Tensor** (7 connections) — `python/sglang/srt/layers/mhc.py`
- **grouped_gemm_nt_f8f8bf16_masked()** (6 connections) — `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
- **_sanity_check_input()** (6 connections) — `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
- **mhc_post()** (6 connections) — `python/sglang/srt/layers/mhc.py`
- **mhc_pre_gemm_sqrsum_tilelang()** (5 connections) — `python/sglang/srt/layers/mhc.py`
- **mhc_post_tilelang()** (5 connections) — `python/sglang/srt/layers/mhc.py`
- **.prewarm_mhc_token_counts()** (5 connections) — `python/sglang/srt/models/deepseek_v4.py`
- **tf32_hc_prenorm_gemm()** (4 connections) — `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
- **hc_split_sinkhorn()** (4 connections) — `python/sglang/srt/layers/mhc.py`
- **JITKernel** (4 connections) — `python/sglang/srt/layers/mhc.py`
- **mhc_pre_big_fuse_with_norm_tilelang()** (4 connections) — `python/sglang/srt/layers/mhc.py`
- **mhc_fused_post_pre_fma_tilelang()** (4 connections) — `python/sglang/srt/layers/mhc.py`
- **_ensure_cuda()** (3 connections) — `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
- **grouped_gemm_nt_f8f8bf16_contig()** (3 connections) — `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
- **gemm_nt_f8f8bf16()** (3 connections) — `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
- **mhc_pre_big_fuse_tilelang()** (3 connections) — `python/sglang/srt/layers/mhc.py`
- **mhc_pre_gemm_sqrsum_splitk_kernel()** (3 connections) — `python/sglang/srt/layers/mhc.py`
- **_compute_num_split_for_mhc_pre()** (3 connections) — `python/sglang/srt/layers/mhc.py`
- **Any** (2 connections) — `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
- *... and 11 more nodes in this community*

## Relationships

- [[Context-Parallel Attention]] (9 shared connections)
- [[CLI Arg Parsing & Deprecation]] (3 shared connections)
- [[Community 45]] (1 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (1 shared connections)
- [[DeepSeek MLA Attention & MoE]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/deep_gemm_wrapper/__init__.py`
- `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
- `python/sglang/srt/layers/mhc.py`
- `python/sglang/srt/models/deepseek_v4.py`

## Audit Trail

- EXTRACTED: 143 (89%)
- INFERRED: 18 (11%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*