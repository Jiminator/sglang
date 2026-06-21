# Round 10 Contract

Round 9 was ADVANCED. Codex's three remaining gaps are all CPU-only evidence-package
consistency/completeness — the explicit "fix these quickly, then move to GPU/instrumentation". This
round finishes the package consistency so AC-1/AC-4/AC-6 generated surfaces are non-contradictory and
complete; after this, the only remaining work is the GPU/instrumentation close-out items.

## Mainline Objective (exactly one)
**Make the generated evidence package fully consistent and complete:** one head-aggregation
classification across every generated surface, a full *effective* per-arm DS config (launch JSON +
runtime defaults), and the stale raw head-agg rows moved under a superseded key — each guarded
fail-closed so it cannot recur.

## Target ACs
- **AC-6** (primary): one head-agg classification across matrix + table + findings.
- **AC-1 / AC-4** (secondary): effective per-arm DS config (defaults expanded), guarded.

## Blocking Side Issues (the three CPU gaps — they ARE the mainline)
- **Head-agg classification contradiction.** R9 reclassified matrix leg 1 to MEASURED (cross-TP
  second-order), but `build_ledger.py:277` still generates `evidence_table.md` with "head_agg
  NOT-a-differing-variable (max on both paths; AC-2.2 covers cross-TP)", and `findings.md:162` repeats
  it. Update both to the matrix's classification (within-rank head_agg=max matched; cross-TP SUM vs
  reference-local is a measured second-order ≤1.3pp difference); regenerate `evidence_table.md`; extend
  the AC-2.2 guard to scan the table + findings for the stale "NOT-a-differing-variable" head-agg wording.
- **`ds_config` is the literal launch JSON, not the effective runtime config.** AC-4 needs selector
  width, score-reduce dtype, head-agg, selector_impl, etc. for every arm. Emit `effective_ds_config` =
  the full resolved `DoubleSparsityConfig` (launch overrides + defaults:
  `selector_width_buckets:[5120]`, `selector_width_overflow_policy:"full_fallback"`,
  `score_reduce_dtype:"bf16"`, `selector_impl:"production"`, `recall_oracle/selection_capture/
  latent_capture/score_capture:false`, `forced_all_dense_control:false`, `reference_include_current:
  false`, `enable_lifted_budget_decode:false`, `lifted_budget_top_k:0`) per DS arm. Strengthen the
  assertion to require the AC-4-relevant effective keys, not only the launch keys.
- **`cheap_controls.json` still has top-level stale raw rows.** `n_score_groups:78` + the top-level
  `head_agg_test` array (rows contain `served_sum_matches_post_reduce`) + the old
  `selected_index_equivalence`/`join`/`n_selection_records` are still active-looking. Move them under the
  `superseded_*` keys; extend the guard so row-level `served_sum_matches`/stale fields are allowed ONLY
  under a `superseded_*` section.

## Queued Side Issues (documented, OUT OF SCOPE — GPU/instrumentation, next rounds)
- AC-2.1 forced-all physical-slot assertions + AC-4 garbage counters (guarded `logical_to_physical`
  adapter instrumentation + GPU run; shared physical-slot boundary).
- AC-3.1 captured-row materialized fp32 `K_label` equality (resident-latent capture + offline compute).
- AC-2.4 recall-oracle@2048 (NIAH-only; GPU).
- AC-4 remaining: DSA-radix serial + production DS sparse serial cells; selected-vs-total gaps.
- AC-8 final writeup (after the above).
- Plan-term comment cleanup; reference-mode fail-closed.

## Concrete Success Criteria
1. `evidence_table.md` and `findings.md` describe head aggregation with the SAME classification as
   `ac6_bisection_matrix.json` (measured, cross-TP second-order ≤1.3pp); no surface says head_agg
   "NOT-a-differing-variable". The AC-2.2 guard scans the table + findings and fails-closed on the stale
   wording (verified it fires).
2. Every DS arm JSON has `effective_ds_config` with all 20 `DoubleSparsityConfig` fields resolved
   (launch overrides + defaults); the ledger assertion requires the AC-4-relevant effective keys
   (`selector_width_buckets`, `score_reduce_dtype`, `selector_impl`, `head_agg`, `scorer_norm`) and fires
   when one is dropped. `evidence_table.md` shows selector width + score-reduce dtype + head-agg per DS arm.
3. `cheap_controls.json` top level no longer carries `n_score_groups:78` / the raw `head_agg_test`
   array / the old `selected_index_equivalence` as active fields — they live under `superseded_*`; the
   guard forbids row-level `served_sum_matches` outside a `superseded_*` section (verified it fires). A
   repo-wide scan shows no `served_sum_matches`/`PRELIMINARY`/`NOT-a-differing-variable`(head-agg) in any
   active surface.
4. Tests pass; provenance consistent. Commit; round-10-summary with BitLesson Delta + Goal Tracker Update
   Request. CPU-only this round; no selection/adapter fix; no exit by lying / editing loop state /
   cancel-rlcr-loop.
