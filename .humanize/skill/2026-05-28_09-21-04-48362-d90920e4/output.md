AGREE:
- The two-tier split is mostly sound: smoke milestone first; loop4-compatible MVP only after AC-10/11/12 plus CUDA-graph and chunked-prefill evidence.
- AC-0 is the right blocker: `_write_token_labels` currently references `forward_batch` without accepting it.
- AC-11 is correctly gated on radix parity; `benchmark_compare.py` refuses `disable_radix_cache` mismatches.
- DEC-1/2/3/5/6 are real scope or deployment decisions, not resolvable from code alone.
- AC-10 requiring both label-capture and FP8 scale-stability fixtures before removing `--disable-radix-cache` is reasonable.

DISAGREE:
- AC-0 call-site coverage is incomplete. Besides extend/decode/TRT-LLM in `dsa_backend.py`, [forward_mha.py](/sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:484) also calls `_write_token_labels` in the MHA_ONE_SHOT path and must pass `forward_batch`.
- AC-4 overclaims validation: a schema-valid random mask with a valid hash is not distinguishable from a bad calibrated mask unless provenance metadata is enforced. Current loader rejects corrupt/out-of-range/NaN/Inf/all-zero-weight rows, not arbitrary random masks.
- AC-1.1 names `total_tokens` as if it exists inside `meta_info["double_sparsity"]`; current DS meta has `sparsity_rate`, `selected_tokens`, and `dense_fallback`.
- AC-1b tiering is inconsistent: it is Tier 2 in the goal/lower boundary, but the sequence puts it in M2 Smoke.
- Quality smoke omits the fourth existing gate: `first_8_tokens_divergence == 0`.

REQUIRED_CHANGES:
- Update AC-0 to require threading `forward_batch` through all production call sites, including MHA_ONE_SHOT, with regression coverage.
- Reconcile AC-1b: based on the candidate’s own tier definition, move it out of Tier 1 smoke sequencing.
- Reword AC-4 negative to something enforceable: require real calibration provenance metadata and reject known synthetic provenance, or move “not random/synthetic” to artifact review instead of loader validation.
- Fix AC-1.1 to compare `selected_tokens` against `prompt_tokens`/known seq length, or require `sparsity_rate > 0`.
- Add `first_8_tokens_divergence == 0` to the Tier 1 quality smoke criteria.

OPTIONAL_IMPROVEMENTS:
- State the AC-12 optional negative sensitivity runs explicitly: corrupt-mask and zero-signature servers.
- Clarify DEC-4: if a smoke comparator is run, radix parity is mandatory; the decision is only “radix-off baseline now” vs “defer comparator.”
- Define the exact smoke artifact labels/env overrides so smoke JSONLs cannot be confused with AC-11 artifacts.

UNRESOLVED:
- none
