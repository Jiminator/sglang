# Community 324

> 21 nodes

## Key Concepts

- **hf_transformers_patches.py** (16 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **apply_all()** (11 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **_patch_rope_parameters_validation()** (3 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **_patch_flash_attn_availability()** (3 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **_patch_removed_symbols()** (3 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **_patch_image_processor_kwargs()** (3 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **_patch_image_process_cuda_tensor()** (3 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **_patch_nemotron_h_pattern()** (3 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **_ensure_clean_up_tokenization_compat()** (3 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **_ensure_is_torch_fx_available_compat()** (3 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **patch_is_base_mistral_in_ci()** (3 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **Apply all transformers compatibility patches (idempotent).      Call this once a** (1 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **Fix rope_parameters validation for unregistered model types.      For unregister** (1 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **Prevent flash-attn-4 from masquerading as flash-attn-2.      flash-attn-4 regist** (1 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **Re-export symbols removed in transformers v5.4.0.      Remote model code (e.g. D** (1 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **Allow remote image processors that lack ``**kwargs`` in preprocess().      Trans** (1 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **Fix ``process_image()`` crashing on CUDA tensors.      Transformers v5.4's PIL i** (1 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **Fix ``_pattern_to_list()`` crashing on ``-`` in hybrid_override_pattern.      Ne** (1 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **Re-add ``clean_up_tokenization`` removed in transformers v5.      Remote-code to** (1 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **Re-add ``is_torch_fx_available`` removed in transformers v5.      Remote-code mo** (1 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **Patch transformers' _patch_mistral_regex to avoid HF API calls in CI.      trans** (1 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`

## Relationships

- [[Community 124]] (4 shared connections)
- [[Community 176]] (2 shared connections)

## Source Files

- `python/sglang/srt/utils/hf_transformers_patches.py`

## Audit Trail

- EXTRACTED: 64 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*