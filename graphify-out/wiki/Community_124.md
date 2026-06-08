# Community 124

> 52 nodes

## Key Concepts

- **common.py** (28 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **tokenizer.py** (24 connections) — `python/sglang/srt/utils/hf_transformers/tokenizer.py`
- **processor.py** (22 connections) — `python/sglang/srt/utils/hf_transformers/processor.py`
- **get_processor()** (22 connections) — `python/sglang/srt/utils/hf_transformers/processor.py`
- **__init__.py** (20 connections) — `python/sglang/srt/utils/hf_transformers/__init__.py`
- **get_tokenizer()** (19 connections) — `python/sglang/srt/utils/hf_transformers/tokenizer.py`
- **_apply_post_load_fixes()** (9 connections) — `python/sglang/srt/utils/hf_transformers/tokenizer.py`
- **download_from_hf()** (8 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **resolve_runai_obj_uri()** (8 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **_resolve_local_or_cached_file()** (8 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **.init_tokenizer()** (7 connections) — `python/sglang/srt/managers/scheduler.py`
- **get_tokenizer_from_processor()** (7 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **attach_additional_stop_token_ids()** (6 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **is_mistral_model()** (6 connections) — `python/sglang/srt/utils/hf_transformers/mistral_utils.py`
- **patch_mistral_common_tokenizer()** (6 connections) — `python/sglang/srt/utils/hf_transformers/mistral_utils.py`
- **_auto_tokenizer_from_pretrained()** (6 connections) — `python/sglang/srt/utils/hf_transformers/tokenizer.py`
- **_build_processor_manually()** (5 connections) — `python/sglang/srt/utils/hf_transformers/processor.py`
- **_resolve_tokenizers_backend()** (5 connections) — `python/sglang/srt/utils/hf_transformers/tokenizer.py`
- **_fix_v5_add_bos_eos_token()** (5 connections) — `python/sglang/srt/utils/hf_transformers/tokenizer.py`
- **_fix_special_tokens_pattern()** (5 connections) — `python/sglang/srt/utils/hf_transformers/tokenizer.py`
- **_ensure_fastokens_patched()** (5 connections) — `python/sglang/srt/utils/hf_transformers/tokenizer.py`
- **_fix_added_tokens_encoding()** (5 connections) — `python/sglang/srt/utils/hf_transformers/tokenizer.py`
- **normalize_rope_scaling_compat()** (5 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **get_context_length()** (4 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **get_sparse_attention_config()** (4 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- *... and 27 more nodes in this community*

## Relationships

- [[Community 176]] (27 shared connections)
- [[Community 346]] (12 shared connections)
- [[Community 415]] (5 shared connections)
- [[Community 77]] (4 shared connections)
- [[Community 324]] (4 shared connections)
- [[Community 290]] (4 shared connections)
- [[Community 47]] (4 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (3 shared connections)
- [[Community 132]] (2 shared connections)
- [[Community 32]] (2 shared connections)
- [[Context-Parallel Attention]] (2 shared connections)
- [[Model Config & Encode Server]] (1 shared connections)

## Source Files

- `python/sglang/srt/configs/model_config.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/utils/hf_transformers/__init__.py`
- `python/sglang/srt/utils/hf_transformers/common.py`
- `python/sglang/srt/utils/hf_transformers/mistral_utils.py`
- `python/sglang/srt/utils/hf_transformers/processor.py`
- `python/sglang/srt/utils/hf_transformers/tokenizer.py`
- `python/sglang/srt/utils/hf_transformers_patches.py`

## Audit Trail

- EXTRACTED: 266 (90%)
- INFERRED: 29 (10%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*