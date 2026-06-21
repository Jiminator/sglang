CORE_RISKS:
- The biggest false-ceiling risk is the reference selector being “pure” but not equivalent: K-noPE reconstruction must use the same q_noPE, W_UK slice, channel weights, latent quant/dequant semantics, layer id, TP head slice, and logical-position mapping as the served path.
- `head_agg="max"` plus cross-TP SUM is a red flag. If each rank computes local `max_h` and then all-reduces with SUM, that is not global max over heads. If the intended aggregate is mean/sum it is additive; if it is max, the reduction op or score definition is wrong.
- “Dense seq < top_k should equal plain attention” is only true if the selector emits every live logical token, logical-to-physical gather is correct, FlashMLA sparse decode is numerically equivalent to dense/native decode for an all-token set, and slot validity does not mask live tokens.
- Raw-dot and cosine should not be treated as two implementations of the same algorithm. Cosine breaks the absorbed-latent identity and requires materialized per-head signatures; if cosine wins, the verdict is “the table-free raw-dot constraint is bad,” not simply “a perf optimization regressed correctness.”
- The absorbed identity `score = agg_h(v_h · c_kv)` is raw-dot-only and key-side latent-only. A materialized `K_label` reference must prove it is the same score as `W_UK · c_kv`, not a subtly different score using pre-quant latent or post-rope channels.
- The current branch appears to claim exact deterministic radix top-k, not approximate top-k. Treat “radix approximation” as a hypothesis to verify by selected-index equivalence, not as a given.
- GSM8K accuracy alone can misattribute the bug. A bad logical-to-physical adapter, stale KV-slot validity, selector-width truncation, or sparse attention kernel mismatch can produce the same repetition garbage as a bad scorer.
- “Only selector differs” is not fully true unless DSA is also controlled for radix-cache state, batching/concurrency, CUDA graph path, and request scheduling. The serial 0.625 vs 0.700 gap is already evidence that path shape matters.

MISSING_REQUIREMENTS:
- Pin and record exact git SHA, model snapshot, mask `content_sha256`, server args, eval sample IDs/order, max tokens, concurrency, and whether CUDA graph was active for every arm.
- Run DSA with radix disabled, or otherwise prove radix-cache state is output-neutral for this exact harness and seed.
- Add a forced-all DS selector control: for `seq_len <= top_k`, emit `[0..seq_len-1]`; for sparse, optionally emit recency/all-known deterministic sets. If dense forced-all is still bad, the selector scorer is not the root cause.
- Assert per layer/step: `valid_lengths == min(seq_len, top_k)`, indices sorted ascending, no duplicates, no `-1` before `valid_lengths`, all physical slots are valid/written, and adapter error count is zero.
- Make `probe_ds_active.sh` per-arm and per-regime, not a single long-context smoke test. Dense should prove all-token selection; sparse should prove capped selection.
- Control the serial-vs-batched gap explicitly: run DSA, production DS, and reference DS under both serial and the eval harness’s normal batching.
- Require score/selection captures on a fixed subset of GSM8K prompts plus NIAH recall-oracle runs. Recall@2048 is corroboration, not an implementation-equivalence proof.
- Log selector width actually used at each decode step. The 24-shot prompt may start under W=5120 but cross it while generating to the cap.

TECHNICAL_GAPS:
- A pure-torch served reference is not just a config toggle. The current production path requires the graph-safe selector and resident fp8 latent; the eager fallback is intentionally fail-closed for production selection.
- `serve.sh ref` likely needs a new config field such as `selector_impl="reference"` plus validator support. `scorer_norm="cosine"` currently cannot pass config validation.
- Reconstructing K-noPE “in bf16/fp32” must be defined. If it dequantizes the resident fp8 latent to fp32, it matches served storage better; if it uses pre-quant bf16 latent, that data is not available for old tokens and is not what the model attends to.
- TP=8 makes “pure torch per-head” nontrivial. Full-head global scoring needs all-gathered per-head scores or an additive aggregate; local-head scoring followed by the wrong all-reduce can reproduce the production bug.
- Per-head selection is probably not directly serveable through the existing FlashMLA sparse path, which consumes a shared selected-token set. Treat per-head selection as an offline upper bound unless you plan a kernel/API change.
- Cosine needs a precise formula: normalize after mask gather? before/after channel weights? per head or shared? The Loop-7 definition must be reused exactly.
- The reference must use the same q_noPE tensor the model uses at decode, not q_lora, post-rope query, or a recomputed approximation.
- The sparse decode path must be verified independently: selected logical positions -> physical slots -> `flash_mla_sparse_fwd` output. Otherwise a downstream bug can be mistaken for H0.

ALTERNATIVE_DIRECTIONS:
- First run forced-all DS in the dense regime. If it does not recover DSA/plain accuracy, redirect to adapter/sparse-attention/slot-validity before building a full reference scorer.
- Before pure torch, try cheap existing toggles: `score_reduce_dtype="fp32"`, `selector_width_buckets=[]`, `head_agg="mean"`, CUDA graph disabled, and selection/score capture enabled.
- Use captured q_noPE + latent + mask from one live decode step to compute an offline fp32 materialized score and compare rank/top-k against production scores. This is faster and more diagnostic than full GSM8K.
- Add a TP aggregation micro-test: local max + SUM vs global max vs global mean/sum on captured per-head dots. This could expose a math bug immediately.
- Run NIAH/recall-oracle before GSM8K for each selector variant. If recall is perfect but GSM8K is bad, the scorer is likely exonerated.
- Use per-head selection only as an oracle ceiling: “would the paper-style per-head algorithm work if the backend could consume it?” Do not mix that with the serveable shared-set ceiling.

QUESTIONS_FOR_USER:
- What exact ceiling threshold should gate Phase B: long-context within 5 points of DSA, within confidence intervals, or a minimum absolute score such as `>0.90`?
- In the BAD branch, should the loop stop after forced-all and reference controls, or also spend time on mask probes such as recalibration, larger `label_dim`, and head aggregation alternatives?
- Should “naive” mean served reference through the existing sparse FlashMLA path, or offline score oracle plus a smaller served approximation first?
- Is cosine allowed to be considered a valid DS ceiling even though it cannot use the table-free absorbed identity?
- Is per-head selection a required research comparison, or should the loop restrict itself to shared-token-set selectors that can actually serve with current MLA kernels?
- Should DSA be rerun with `--disable-radix-cache` for a strict selector-only comparison?
- How many repeats are required before calling a 150/200-sample GSM8K delta real?

CANDIDATE_CRITERIA:
- AC-0: Current DSA, DSA-radix-off, and production DS are reproduced at a pinned SHA with fixed sample IDs; production DS regression and DS active metrics are recorded.
- AC-1: Forced-all DS passes dense invariants: every live logical token selected, no adapter errors, no invalid physical slots, and dense GSM8K/logit spot checks match DSA/plain within a stated tolerance.
- AC-2: Reference raw-dot scores match a materialized fp32 K-label implementation on captured decode steps, including TP aggregation semantics and selected-index equality at K=2048.
- AC-3: Reference served mode is explicit in config/scripts, states graph/eager behavior, uses the same FlashMLA sparse attention path, and supports raw-dot plus the precisely defined cosine variant.
- AC-4: Every measured arm reports GSM8K dense/sparse, serial/batched mode, selector width, reduce dtype/backend, head aggregation, selected/total summaries, and length-cap garbage rate.
- AC-5: BAD ceiling verdict is allowed only after forced-all sparse path, reference-score equivalence, and mask metadata checks pass.
- AC-6: GOOD ceiling Phase B changes exactly one variable per arm and reports both GSM8K delta and selected-index/score-rank mismatch against the reference.
- AC-7: Final writeup names H0/H1/H2 with numeric evidence and negative controls, and explicitly leaves fixes to a follow-up loop.
