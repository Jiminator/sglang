# Round 2 Summary — Loop 13 (DS-vs-DSA accuracy diagnosis)

## Objective (round-2-contract.md): make the evidence reproducible/captured + fix correctness
No verdict change. The GOOD-ceiling two-regression verdict (dense = H3 current-slot exclusion;
sparse = the raw-dot `scorer_norm="off"` lock) stands and is unchanged.

## Work Completed
- **DSA-baseline consistency (blocking fix).** Gate/writeup now use the measured **batched** DSA
  0.975/0.973 (was the plan's prior-session 0.953). Sparse gap = 3.3 pp; **GATE stays GOOD**.
  `run_meta.json` SHA corrected (baselines @180f6dd6d, current @HEAD).
- **Verdict softened (blocking fix).** ROOT_CAUSE/gate now label the sparse attribution
  **reference-ceiling** and mark the production-path one-variable bisection **pending**.
- **AC-4 per-arm JSON ledger.** `build_ledger.py` generates `evidence/meta/arms/*.json` for all 8
  arms (config, full server args, scores read from the `.out` files, DS selected-vs-total by
  regime) and regenerates `evidence_table.md`. Fields needing harness instrumentation (per-example
  sample IDs/order; per-step length-cap garbage counters) are listed as `fields_not_instrumented`
  — honest, not faked.
- **AC-3.1 materialized-K equality.** `test_reference_selectors.py::test_materialized_raw_equals_absorbed_raw`
  + `evidence/ac3_1_materialized_k.json`: the materialized fp32 `K_label` score is selection-equal to
  the absorbed raw-dot (max |Δ| 4.8e-6, bit-identical top-k). The identity is exact algebra
  (input-independent), so the synthetic-row proof is conclusive.
- **AC-2 capture pipeline.** `ds_capture` → 1872 score + 104 selection `.pt` dumps →
  `analyze_captures.py` → `cheap_controls.json`, end-to-end. (See limitation below.)
- **Code-comment cleanup.** Removed plan-workflow terms (`AC-*`, `H3`) from my new code/harness
  comments (pre-existing comments untouched). 5/5 CPU tests still pass.

## Files Changed
- Code: `absorbed_latent.py`, `deepseek_v2.py` (comment cleanup only this round; the `normalize`
  control + reference selectors were committed in Round 1).
- Harness/evidence: `serve.sh` (comment cleanup), `build_ledger.py` (new), `evidence/meta/arms/*.json`
  (8 arms), `evidence_table.md` (regenerated), `evidence/{gate_ac5.md, ROOT_CAUSE.md, cheap_controls.json,
  ac3_1_materialized_k.json, meta/run_meta.json}`.
Commit `ac479aeb3` (+ the earlier consistency edits); tree clean; one TP=8 server at a time; GPUs idle.

## Validation
- `python3 development/loop13/test_reference_selectors.py` → 5/5 pass (incl. the AC-3.1 materialized-K
  equality test).
- `build_ledger.py` regenerated the table data-driven from the `.out` files (scores match the
  committed runs). `cheap_controls.json` produced from real `ds_capture` `.pt` dumps.
- Gate re-checked with the consistent measured baseline: dense 0.950 (2.5 pp), sparse 0.940 (3.3 pp)
  → GOOD.

## Remaining Items (honest — do not over-claim)
- **AC-2.2/2.3 numbers are PRELIMINARY.** `analyze_captures.py`'s selected-index-equivalence is a
  cross-record cartesian comparison (not aligned by `(req_pool_index, layer, decode_step)`), and the
  head-agg `pre_reduce_scores` semantics is unconfirmed (served-SUM ≠ post-reduce). Annotated
  PRELIMINARY in `cheap_controls.json`; secondary corroboration only — the verdict does not depend on them.
- **AC-2.1 forced-all physical-slot assertion JSON** + **AC-4 per-step garbage counters** + sample
  IDs/order: not built — require `logical_to_physical`/adapter capture instrumentation in the serving path.
- **AC-6 production-path bisection** incomplete: reference-ceiling cliff (cosine 0.940 vs raw-dot
  0.013 + materialized-raw proof) names the sparse candidate and the opts are bounded second-order
  (production raw-dot 0.000 ≈ exact raw-dot 0.013), but the one-variable production arms are not run.

## Goal Tracker Update Request

### Requested Changes / scope decisions needed:
1. **AC-6 "production-style cosine" conflicts with the no-fix constraint.** Cosine is NOT servable on
   the production graph-safe path without implementing the materialized per-head signature there —
   which IS the recommended fix. The plan says "no fix this loop." So either (a) accept the
   reference-ceiling AC-6 attribution (cliff cosine 0.940 vs raw-dot 0.013 + materialized-raw proof +
   opts-second-order) as the diagnosis-loop result, or (b) authorize landing the production cosine
   path (a fix-adjacent change) to complete the production-path bisection. **Recommend (a).**
2. **AC-2.1 forced-all assertions + AC-4 garbage counters + sample-ID/order** require building
   `logical_to_physical`/adapter capture instrumentation + harness changes. Confirm whether this is in
   scope for a diagnosis loop whose verdict already does not depend on it (it would CORROBORATE the
   already-decisive forced-all GSM8K recovery 0.620→0.950), or defer to a follow-up.
3. **AC-2.2/2.3 captured cheap-controls** need a per-`(req,layer,step)`-aligned analyzer + confirmation
   of `pre_reduce_scores` semantics before their numbers are load-bearing.

### Justification:
The Ultimate Goal is a root-cause verdict with live evidence. That verdict is delivered and
Codex-accepted in substance (GOOD ceiling; dense=H3, sparse=raw-dot scorer lock), and the decisive
controls (forced-all/anchor dense recovery; faithful raw-dot vs faithful cosine sparse;
materialized-raw selection-equality) are live GSM8K + proven. The open items are CORROBORATING rigor
whose two largest pieces (production-style cosine; adapter garbage-counter instrumentation) either
conflict with the "no fix" constraint or require building serving instrumentation beyond a diagnosis
loop. Surfacing this for an explicit scope decision rather than silently deferring.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: Round 2 was evidence-completeness + correctness consolidation; the load-bearing lessons
  (BL-20260620-ds-current-slot-exclusion, BL-20260620-ds-rawdot-scorer-lock) were captured in
  Rounds 0–1 and are unchanged.
