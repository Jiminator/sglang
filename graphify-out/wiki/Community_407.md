# Community 407

> 15 nodes

## Key Concepts

- **speculative_hook.py** (10 connections) — `python/sglang/srt/arg_groups/speculative_hook.py`
- **handle_speculative_decoding()** (9 connections) — `python/sglang/srt/arg_groups/speculative_hook.py`
- **adaptive_unsupported_reason()** (4 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **_resolve_speculative_algorithm_alias()** (3 connections) — `python/sglang/srt/arg_groups/speculative_hook.py`
- **_handle_dflash()** (3 connections) — `python/sglang/srt/arg_groups/speculative_hook.py`
- **_handle_eagle_family()** (3 connections) — `python/sglang/srt/arg_groups/speculative_hook.py`
- **_maybe_disable_adaptive()** (3 connections) — `python/sglang/srt/arg_groups/speculative_hook.py`
- **_handle_frozen_kv_mtp()** (2 connections) — `python/sglang/srt/arg_groups/speculative_hook.py`
- **_handle_ngram()** (2 connections) — `python/sglang/srt/arg_groups/speculative_hook.py`
- **ServerArgs** (2 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **Resolve CLI speculative algorithm; NEXTN/EAGLE may become FROZEN_KV_MTP for Gemm** (1 connections) — `python/sglang/srt/arg_groups/speculative_hook.py`
- **# TODO: move the per-algorithm validation below into spec module hooks.** (1 connections) — `python/sglang/srt/arg_groups/speculative_hook.py`
- **# TODO: support dp attention for standalone speculative decoding** (1 connections) — `python/sglang/srt/arg_groups/speculative_hook.py`
- **# TODO: support dp attention for ngram speculative decoding** (1 connections) — `python/sglang/srt/arg_groups/speculative_hook.py`
- **Return why adaptive spec cannot run under the given server args, or None if supp** (1 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`

## Relationships

- [[CLI Arg Parsing & Deprecation]] (3 shared connections)
- [[Community 337]] (2 shared connections)
- [[Community 403]] (1 shared connections)

## Source Files

- `python/sglang/srt/arg_groups/speculative_hook.py`
- `python/sglang/srt/speculative/adaptive_spec_params.py`

## Audit Trail

- EXTRACTED: 39 (85%)
- INFERRED: 7 (15%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*