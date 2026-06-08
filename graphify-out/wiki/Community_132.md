# Community 132

> 50 nodes

## Key Concepts

- **model_config.py** (31 connections) — `python/sglang/srt/configs/model_config.py`
- **.__init__()** (26 connections) — `python/sglang/srt/configs/model_config.py`
- **is_deepseek_dsa()** (24 connections) — `python/sglang/srt/configs/model_config.py`
- **is_deepseek_v4()** (9 connections) — `python/sglang/srt/configs/model_config.py`
- **PretrainedConfig** (9 connections) — `python/sglang/srt/configs/model_config.py`
- **get_dsa_index_head_dim()** (8 connections) — `python/sglang/srt/configs/model_config.py`
- **._compute_cell_size()** (8 connections) — `python/sglang/srt/model_executor/pool_configurator.py`
- **get_dsa_index_topk()** (6 connections) — `python/sglang/srt/configs/model_config.py`
- **validate_hisparse()** (5 connections) — `python/sglang/srt/arg_groups/hisparse_hook.py`
- **get_num_indexer_layers()** (5 connections) — `python/sglang/srt/configs/model_config.py`
- **._derive_hybrid_model()** (5 connections) — `python/sglang/srt/configs/model_config.py`
- **._derive_model_shapes()** (5 connections) — `python/sglang/srt/configs/model_config.py`
- **compute_mla_mscale_scaling()** (5 connections) — `python/sglang/srt/configs/model_config.py`
- **_hf_arch()** (4 connections) — `python/sglang/srt/configs/model_config.py`
- **get_dsa_index_n_heads()** (4 connections) — `python/sglang/srt/configs/model_config.py`
- **_get_and_verify_dtype()** (4 connections) — `python/sglang/srt/configs/model_config.py`
- **dtype** (4 connections) — `python/sglang/srt/configs/model_config.py`
- **_hf_attr()** (3 connections) — `python/sglang/srt/configs/model_config.py`
- **._detect_attention_sinks()** (3 connections) — `python/sglang/srt/configs/model_config.py`
- **._validate_quantize_and_serve_config()** (3 connections) — `python/sglang/srt/configs/model_config.py`
- **is_multimodal_chunked_prefill_supported()** (3 connections) — `python/sglang/srt/configs/model_config.py`
- **is_hybrid_swa_model()** (3 connections) — `python/sglang/srt/configs/model_config.py`
- **get_hybrid_layer_ids()** (3 connections) — `python/sglang/srt/configs/model_config.py`
- **._config_draft_model()** (2 connections) — `python/sglang/srt/configs/model_config.py`
- **._get_sliding_window_size()** (2 connections) — `python/sglang/srt/configs/model_config.py`
- *... and 25 more nodes in this community*

## Relationships

- [[Model Config & Encode Server]] (12 shared connections)
- [[DeepSeek MLA Attention & MoE]] (11 shared connections)
- [[CLI Arg Parsing & Deprecation]] (6 shared connections)
- [[Disaggregation Utils & Cache Tests]] (6 shared connections)
- [[Community 196]] (4 shared connections)
- [[Context-Parallel Attention]] (3 shared connections)
- [[Hybrid Attention Backend]] (2 shared connections)
- [[Grammar Manager & HiCache Clear]] (2 shared connections)
- [[Community 43]] (2 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (2 shared connections)
- [[Community 279]] (2 shared connections)
- [[Community 124]] (2 shared connections)

## Source Files

- `python/sglang/srt/arg_groups/hisparse_hook.py`
- `python/sglang/srt/configs/model_config.py`
- `python/sglang/srt/model_executor/pool_configurator.py`

## Audit Trail

- EXTRACTED: 174 (79%)
- INFERRED: 46 (21%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*