# Ask Codex Input

## Question

# Candidate Plan v1 — Loop 13: Root-cause the DS-vs-DSA accuracy degradation

(This is a candidate for your reasonability review. Critique it. The ground truth from code inspection is included so you can check feasibility.)

## Ground truth established by code inspection (use this to judge feasibility)
- Selection seam: `DeepseekV2AttentionMLA._select_topk_indices` (deepseek_v2.py:2127) returns int32 [bs, max_top_k] logical positions ascending, -1 padded; downstream `logical_to_physical` (page_table_adapter.py) converts to physical slots, then `flash_mla_sparse_fwd` runs attention. This is the natural seam for a reference selector branch.
- `scorer_norm` is HARD-LOCKED to ("off",) in config.py (__post_init__) AND asserted in selection_kernel.py:529, with a comment: the absorbed-latent identity `score = max_h v_h·c_kv` holds only for the raw dot. So cosine cannot pass config validation today; it needs a materialized per-head signature the production path never builds.
- Existing pure-fp32 CPU reference functions ALREADY exist: `absorbed_latent_score` (absorbed_latent.py:166-191, fp32 einsum `score=agg_h(v_h·c_kv)`, head_agg max/mean), `absorbed_latent_score_logical` (paged), and `select_topk_sequence_order` (selection_kernel.py:321-381, exact torch top-k reference). There is NO production "naive/ref" selectable mode.
- Several draft "suspects" are CONFIG TOGGLES today: `score_reduce_dtype` ("bf16" default / "fp32"), `selector_width_buckets` ([5120] default / [] = full width), `head_agg` ("max"/"mean"), `recall_oracle` (bool, needs --disable-cuda-graph), `score_capture`/`selection_capture` (bool).
- The production radix top-k (topk_kernel.py:245-328) is EXACT (bit-identical to the torch reference), contradicting the draft's "approximate radix top-k" suspicion. selector_width_buckets are claimed bit-identical-selection prefix windows. Both should be VERIFIED by selected-index equivalence, not assumed culprits.
- head_agg="max" computes local per-rank max over TP-sharded heads (absorbed_latent_kernel.py:200-205) then SUM-reduces across TP (reduce_token_scores, selection_kernel.py:201-286). Local-max + cross-rank-SUM is NOT a global max over heads — a concrete math-semantics suspect.
- In fp32 (no fp8, no tf32 MMA), the absorbed identity is EXACT algebra, so fp32 absorbed_latent_score == fp32 materialized-K_label raw-dot score. The fp8 in-register dequant + tf32 MMA are the only approximations.

## Goal
Diagnosis loop (NOT a fix). Produce a root-cause VERDICT WITH EVIDENCE explaining why table-free DS on GLM-5.1-FP8 is far below the native DSA indexer (GSM8K dense 0.625 vs 0.970; sparse 0.000 vs 0.953). Localize to exactly one of:
- H0: channel-selection algorithm doesn't transfer to GLM MLA (ceiling bad even when exact/slow).
- H1: a loop6-12 perf optimization corrupts selection (ceiling good; one toggle regresses it).
- H2: the offline channel mask is bad for GLM-5.1 (sub-branch of H0).
- H3 (NEW): selection is fine; the regression is DOWNSTREAM of selection — the logical->physical adapter, KV-slot validity, or FlashMLA sparse-decode gather. Motivated by the dense regime: DS selects all ~763<2048 tokens (selection is a no-op there) yet still scores 0.625, the classic signature of a downstream bug, not a scorer bug.
Deliverable: verdict + the reusable reference selector built to find it. Fixing is a follow-up loop.

## Restructured approach: cheapest decisive experiment first

### Phase 0 — Pin & reproduce baselines (control the comparison)
- Pin and record for EVERY arm: git SHA, model snapshot, mask content_sha256, full server args, CUDA-graph on/off, eval sample IDs + order, max_tokens, concurrency, serial-vs-batched mode.
- Reproduce: DSA (native), production DS, and a DSA-radix-OFF control (to nail the draft's "radix is output-neutral at temp 0" claim instead of assuming it).
- Run all arms BOTH serial and under the eval harness's normal batching (the existing serial 0.625 vs batched 0.700 gap proves path shape matters).

### Phase 1 — Cheap localization controls BEFORE building a reference selector
Run these on a FIXED GSM8K subset, with --disable-cuda-graph where diagnostics require it:
- **Recall-oracle @2048** (existing `recall_oracle` config flag + selection_recall_oracle.py) in dense and sparse. If recall is ~1.0 but accuracy is bad => scorer exonerated => H3.
- **Forced-all DS control**: a selector that, for seq_len<=top_k, emits [0..seq_len-1] (and a deterministic recency/all-known set for sparse). If dense forced-all is STILL bad => selection is a no-op there => bug is DOWNSTREAM (H3). If it recovers => the scorer mis-selects even in dense (e.g. picks padding positions / page-rounding).
- **Offline score-equivalence + TP-aggregation micro-test**: from `score_capture`/`selection_capture` on captured decode steps, compute offline fp32 materialized-K_label score and compare rank/top-k@2048 against production scores; and explicitly test local-max+SUM vs global-max vs global-mean on captured per-head dots to expose the head_agg="max"+SUM semantics question.
- **Invariant assertions** per layer/step: valid_lengths == min(seq_len, top_k); indices sorted ascending; no duplicates; no -1 before valid_lengths; all physical slots valid/written; adapter error count == 0.
- **Verify the contradicted suspects cheaply**: confirm radix top-k == torch.topk and selector_width_buckets [5120] vs [] produce identical selected indices (selected-index equivalence), retiring or confirming suspects #4/#5 by inspection+equivalence rather than full GSM8K.
- **GATE 1**: If these controls localize to H3 (downstream) or already convict a specific toggle (e.g. head_agg SUM), record the verdict; the full reference-selector build (Phase A) becomes a confirmation, not the primary instrument.

### Phase A — Establish the accuracy ceiling (reference selector)
- Build a reference selector branched at `_select_topk_indices`, selectable via a new config field (e.g. `selector_impl="reference"`) + `serve.sh ref` mode, reusing the same guard and run_gsm8k.sh.
- Reference = materialize a real per-head K_label by dequantizing the RESIDENT fp8 latent to fp32 (same storage the model attends to — NOT pre-quant bf16, which isn't available for old tokens), gather mask channels, build Q_label from the SAME q_noPE the model uses at decode, score EXACTLY in fp32, exact full-width torch.topk, fp32 reduce.
- PROVE equivalence: fp32 materialized-K_label score == existing fp32 `absorbed_latent_score` on captured steps (selected-index equality @2048). Once proven, either is the trustworthy raw-dot ceiling.
- Provide BOTH raw-dot and a precisely-defined cosine variant (normalize after mask gather, per the Loop-7 definition, on the materialized signature). NOTE: cosine breaking the absorbed identity means "cosine wins" => verdict is "the table-free raw-dot CONSTRAINT is bad" (a design constraint, between H0 and H1), not simply "a perf opt regressed it."
- Measure GSM8K for DSA, naive-DS (raw), naive-DS (cosine), production DS, on the validated configs (5-shot/200 dense, 24-shot/150 sparse, temp 0, completion API), serial + batched.
- **Decision gate (numeric threshold, user-confirmed)**: ceiling GOOD if naive-DS (best of raw/cosine) sparse GSM8K is within N points of DSA AND does not collapse (>0); else BAD.
  - BAD => H0/H2 (algorithm/mask doesn't transfer); research problem; optionally cheap mask sanity via recall-oracle; STOP at verdict (no perf chasing).
  - GOOD => H1 => Phase B.

### Phase B — Bisect the optimization history (only if ceiling good)
- Walk forward from the reference, re-enabling ONE variable per arm, measuring GSM8K (dense+sparse) until accuracy drops. First drop = culprit (may be >1). Order by suspicion, but informed by Phase 1:
  1. head_agg shared-max+SUM semantics (elevated by the TP-agg micro-test),
  2. raw-dot vs cosine scorer (scorer_norm lock),
  3. fp8 absorbed-latent scoring vs materialized fp32 K_label,
  4. bf16 vs fp32 score-reduce (score_reduce_dtype toggle),
  5. approximate radix top-k vs exact (likely already retired in Phase 1),
  6. selector-width ladder/W=5120 (likely already retired in Phase 1).
- Prefer config/flag toggles (score_reduce_dtype, head_agg, selector_width_buckets, recall_oracle); fall back to git-stepping loop6->loop12 commits where no toggle exists.
- Corroborate EVERY GSM8K delta with recall@2048 + selected-index/score-rank mismatch vs the reference (GSM8K n=150 stderr ~4 points, so deltas <~5 points need oracle corroboration, not just the eval number).

## Proposed Acceptance Criteria (draft — refine into AC-X with positive/negative tests)
- AC-1: Pinned reproduction of DSA, DSA-radix-off, production DS with the regression reproduced; full per-arm metadata recorded.
- AC-2: Phase-1 cheap controls executed and recorded (recall-oracle, forced-all, TP-agg micro-test, invariants, radix/width equivalence), with an explicit H3 fork.
- AC-3: Reference selector serves GLM-5.1-FP8 (materialized fp32 K_label from resident-fp8 dequant, exact full-width topk, fp32 reduce, raw+cosine), proven equal to fp32 absorbed reference on captured steps, DS genuinely active (selected<total, dense_fallback==0).
- AC-4: GSM8K for DSA / naive-raw / naive-cosine / production on validated configs, serial+batched, with selector width / reduce dtype / head_agg / selected-vs-total / length-cap-garbage-rate reported.
- AC-5: Decision gate recorded with explicit numeric threshold; loop branches accordingly.
- AC-6 (conditional, ceiling good): culprit toggle identified by single-variable bisection with the GSM8K cost and commit(s), each delta corroborated by recall/selected-index.
- AC-7: Root-cause writeup in development/loop13/ with evidence table, verdict (H0/H1/H2/H3), recommendation (research vs targeted fix), explicitly NOT a fix.

## Open decisions to confirm with the user
- DEC-1 (how naive): materialized-fp32-K_label reference (unimpeachable + cosine-enabler) reusing the existing fp32 absorbed reference as a cross-check — vs a fully standalone pure-torch selector. Recommend materialized-K_label + absorbed cross-check.
- DEC-2 (ceiling threshold): exact N for "sparse within N points of DSA" (e.g. within 5) and/or an absolute floor (e.g. >0.90).
- DEC-3 (BAD-branch scope): stop at verdict, or also probe mask quality (recalibrate / label_dim / per-head) this loop.
- DEC-4 (per-head): treat per-head selection as an OFFLINE ORACLE upper bound only (FlashMLA consumes a shared set), not a served GSM8K arm — confirm.
- DEC-5 (cosine's verdict category): is cosine a legitimate DS ceiling even though it breaks the table-free identity? If cosine is the ceiling-maker, is that H0-design or H1-regression?
- DEC-6 (significance): how many GSM8K repeats before a 150/200 delta is "real"? (Recommend: large gaps single-run; Phase-B small deltas need oracle corroboration.)
- DEC-7 (is reference build mandatory): if Phase-1 cheap controls already convict H3 or a specific toggle, is the full reference-selector build still required for the deliverable, or does it become optional confirmation?

## Your task
Review this candidate plan for reasonability. Output EXACTLY these sections:
AGREE:
DISAGREE:
REQUIRED_CHANGES:
OPTIONAL_IMPROVEMENTS:
UNRESOLVED:

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-20_09-38-06
- Tool: codex
