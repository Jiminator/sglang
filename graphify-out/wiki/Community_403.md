# Community 403

> 16 nodes

## Key Concepts

- **dflash_utils.py** (16 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **parse_dflash_draft_config()** (12 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **Any** (11 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **can_dflash_slice_qkv_weight()** (5 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **can_dflash_use_fused_qkv_proj()** (5 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **resolve_dflash_verify_mask_policy()** (4 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **is_dflash_sampling_verify_available()** (3 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **_cfg_get()** (3 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **_get_text_config()** (3 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **_get_dflash_config()** (3 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **_parse_optional_int()** (3 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **validate_dflash_request()** (3 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **Req** (3 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **Parse and validate DFLASH draft config fields from HF config/dict.** (1 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **Validate whether DFlash can slice KV weights from a fused QKV linear layer.** (1 connections) — `python/sglang/srt/speculative/dflash_utils.py`
- **Validate whether a QKV layer is eligible for DFlash fused KV materialization.** (1 connections) — `python/sglang/srt/speculative/dflash_utils.py`

## Relationships

- [[Community 4274]] (4 shared connections)
- [[CLI Arg Parsing & Deprecation]] (4 shared connections)
- [[Community 9590]] (3 shared connections)
- [[Compressed-Tensors Quant Linear]] (2 shared connections)
- [[HiCache Controller & Radix Tree]] (2 shared connections)
- [[Model Configs & Pooler]] (2 shared connections)
- [[Community 196]] (1 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (1 shared connections)
- [[Breakable CUDA Graph (TBO)]] (1 shared connections)
- [[Community 407]] (1 shared connections)
- [[Hybrid Attention Backend]] (1 shared connections)
- [[Community 32]] (1 shared connections)

## Source Files

- `python/sglang/srt/speculative/dflash_utils.py`

## Audit Trail

- EXTRACTED: 62 (81%)
- INFERRED: 15 (19%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*