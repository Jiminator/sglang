# Community 176

> 38 nodes

## Key Concepts

- **config.py** (18 connections) — `python/sglang/srt/utils/hf_transformers/config.py`
- **get_config()** (12 connections) — `python/sglang/srt/utils/hf_transformers/config.py`
- **PretrainedConfig** (10 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **get_hf_text_config()** (10 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **check_gguf_file()** (9 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **.parse()** (8 connections) — `python/sglang/srt/utils/hf_transformers/config.py`
- **_resolve_tokenizer_name()** (8 connections) — `python/sglang/srt/utils/hf_transformers/tokenizer.py`
- **ModelConfigParserBase** (7 connections) — `python/sglang/srt/configs/model_config_parser_registry.py`
- **._handle_load_format()** (7 connections) — `python/sglang/srt/server_args.py`
- **is_remote_url()** (7 connections) — `python/sglang/srt/utils/common.py`
- **_override_v_head_dim_if_zero()** (7 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **_is_deepseek_ocr_model()** (6 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **_is_deepseek_ocr2_model()** (6 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **_ensure_gguf_version()** (6 connections) — `python/sglang/srt/utils/hf_transformers_patches.py`
- **model_config_parser_registry.py** (5 connections) — `python/sglang/srt/configs/model_config_parser_registry.py`
- **PathLike** (5 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **get_model_config_parser()** (4 connections) — `python/sglang/srt/configs/model_config_parser_registry.py`
- **_patch_text_config()** (4 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **_set_architectures()** (4 connections) — `python/sglang/srt/utils/hf_transformers/config.py`
- **_apply_deepseek_ocr_overrides()** (4 connections) — `python/sglang/srt/utils/hf_transformers/config.py`
- **HfModelConfigParser** (4 connections) — `python/sglang/srt/utils/hf_transformers/config.py`
- **MistralModelConfigParser** (4 connections) — `python/sglang/srt/utils/hf_transformers/config.py`
- **.parse()** (3 connections) — `python/sglang/srt/configs/model_config_parser_registry.py`
- **._is_mistral_native_format()** (3 connections) — `python/sglang/srt/server_args.py`
- **.parse()** (3 connections) — `python/sglang/srt/utils/hf_transformers/config.py`
- *... and 13 more nodes in this community*

## Relationships

- [[Community 124]] (27 shared connections)
- [[DeepSeek MLA Attention & MoE]] (9 shared connections)
- [[CLI Arg Parsing & Deprecation]] (8 shared connections)
- [[Community 346]] (4 shared connections)
- [[Community 112]] (3 shared connections)
- [[Aibrix KV Cache Storage]] (2 shared connections)
- [[Community 290]] (2 shared connections)
- [[Community 324]] (2 shared connections)
- [[Community 42]] (1 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (1 shared connections)
- [[Community 132]] (1 shared connections)
- [[Model Configs & Pooler]] (1 shared connections)

## Source Files

- `python/sglang/srt/configs/model_config_parser_registry.py`
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/utils/common.py`
- `python/sglang/srt/utils/hf_transformers/common.py`
- `python/sglang/srt/utils/hf_transformers/config.py`
- `python/sglang/srt/utils/hf_transformers/tokenizer.py`
- `python/sglang/srt/utils/hf_transformers_patches.py`

## Audit Trail

- EXTRACTED: 145 (80%)
- INFERRED: 36 (20%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*