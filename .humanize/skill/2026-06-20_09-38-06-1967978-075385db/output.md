AGREE:
- The phase ordering is good: reproduce and pin first, run cheap localization before building a large reference path, then only bisect optimizations if the ceiling is good.
- `_select_topk_indices` is the right selector seam, and `logical_to_physical` is the right place to instrument selection-to-attention handoff.
- H3 is a necessary hypothesis. Dense `seq_len < top_k` degradation is much more downstream-shaped than scorer-shaped.
- Elevating TP `head_agg="max"` semantics is correct: local per-rank max followed by cross-rank SUM is not global max.
- Retiring radix/topk and selector-width suspicions by selected-index equivalence, not GSM8K, is exactly right.

DISAGREE:
- Do not make materialized fp32 `K_label` the primary served reference. For GLM-scale `H x T x label_dim`, it is likely too expensive or memory-hostile. Use existing absorbed fp32 logical scoring as the raw-dot reference; materialize signatures only offline/blockwise for proof or cosine.
- `recall_oracle` cannot by itself exonerate the scorer. The Pensieve note says it is a NIAH recall diagnostic, not a generic selected-index equivalence checker. Treat it as corroborating evidence only.
- H3 is worded too narrowly around `flash_mla_sparse_fwd`. The current durable DS launcher uses `flashmla_kv` both phases unless lifted-budget is enabled. H3 must include `logical_to_physical -> transform_index_page_table_decode -> flashmla_kv` as well as the lifted `flash_mla_sparse_fwd` path.
- AC-3’s “DS genuinely active selected<total” conflicts with the dense no-op test. That criterion should apply to sparse arms only; dense is useful precisely because selection should select all live tokens.
- “Cosine wins” should not be classified as H1. Since cosine breaks the absorbed raw-dot identity and current config hard-locks `scorer_norm="off"`, a cosine win means the raw-dot/table-free design constraint is bad for GLM, closer to H0/H2 than a regression.

REQUIRED_CHANGES:
- Change Phase A reference to: `selector_impl="reference_absorbed_fp32"` using resident fp8 latent dequant, existing `absorbed_latent_score_logical`, full-width exact torch topk, fp32 reduce, and `--disable-cuda-graph`. Keep materialized `K_label` as offline/blockwise cross-check.
- Add a dense downstream control stronger than recall: forced logical `[0..seq_len-1]`, then assert physical slots equal `req_to_token[req_pool, 0:seq_len]`, no duplicates, no `-1`, all written, and output recovers or stays bad.
- For sparse H3, add a “native DSA selected slots through DS downstream path” or equivalent adapter/decode bypass if feasible. Forced recency is not a clean downstream control.
- Require selected-index equivalence artifacts for every cheap toggle: `score_reduce_dtype`, `head_agg`, width bucket, radix topk. GSM8K alone is too noisy for toggle conviction.
- If any GPU “fp32 reference” is used, explicitly disable TF32 or label it as GPU-fp32-with-TF32-risk. Otherwise the plan’s “exact fp32” claim is leaky.
- Decide DEC-7 now. The goal says the deliverable includes a reusable reference selector; if Phase 1 convicts H3, either still build the lighter absorbed reference or amend the deliverable.

OPTIONAL_IMPROVEMENTS:
- Add a TP=1 or offline all-head aggregation sanity check if model/resource constraints allow; it would isolate the local-max+SUM issue cleanly.
- Log per-layer failure localization for forced-all dense. If only later layers fail, that points toward slot-written/adapter state drift rather than the selector math.
- Include radix-on/off and serial/batched as separate axes in the evidence table, since graph/reuse path shape has already proven value-affecting.
- Define “length-cap-garbage-rate” concretely as invalid physical slots, unwritten slots, duplicates, and out-of-range lanes per layer/step.

UNRESOLVED:
- The numeric ceiling threshold: I’d use sparse within 5 points of DSA or absolute >0.90, but it needs owner confirmation.
- Whether cosine is in-scope as a served GSM8K arm or only an offline oracle. Served cosine may require substantial blockwise machinery.
- Whether BAD branch stops at verdict or includes mask-quality probes.
- Number of repeats for small GSM8K deltas. Large collapses need one run; <5 point deltas need index/recall corroboration or repeats.
