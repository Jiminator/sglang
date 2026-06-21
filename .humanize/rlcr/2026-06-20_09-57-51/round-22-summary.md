# Round 22 Summary

Mainline: **AC-8 — final root-cause writeup regenerated from the complete evidence package + a fail-closed
self-check.** Codex's R21-review marked AC-1/2/3/4/5 MET; AC-8 was the sole remaining item. CPU-only (all
evidence is committed); no GPU. The diagnosis loop is now complete — verdict + recommendation, **no fix
landed**.

## Work Completed
1. **`ROOT_CAUSE.md` regenerated** from the final package (no stale "Round 1/7" labels):
   - Per-arm table carries BOTH **batched AND serial** (R21) for the core arms + **artifact-backed
     selected-vs-total** (dense 334/334, sparse 2048/3692, dense_fallback 0).
   - Cites every corroboration artifact directly: `forced_all_assertions.json` (AC-2.1 — H3 measured on the
     `_ds_slot_written` bitmap, 61776/61776, current slot the only unwritten lane), `ac2_4_recall_oracle.json`
     (AC-2.4 — sparse recall@2048 = 0.41 scorer-driven / dense 1.0), `ac3_1_materialized_k_selected_index_equality.json`
     (AC-3.1 — captured-row raw-dot == materialized fp32 `K_label` top-2048, 96/96 both regimes),
     `ac4_garbage_counters*.json` (clean adapter path on every served arm), `ac4_selected_vs_total.json`,
     `evidence_table.md`, `ac6_bisection_matrix.json`, `gate_ac5.md`.
   - **Ranked verdict:** dense = **H3** (current decode slot excluded from its own selection — `_slot_written`
     not restored); sparse = the **raw-dot `scorer_norm="off"` lock** (Loop-11 `01e3ff238` dropped the Loop-7
     cosine scorer) **interacting** with H3 (the AC-6 2×2 — sparse recovery to ≈0.94 needs BOTH). GOOD gate;
     not H0/H2; AC-7 moot. Explicit **"no selection/adapter fix is landed — recommendation only"** boundary.
2. **`ac8_selfcheck.py`** (new, fail-closed) — refuses AC-8 unless every AC-4 core arm has non-blank
   batched+serial cells, `run_meta` records the selected-vs-total + recall-oracle + captured-materialized-K
   summaries (with the correct artifact paths), and `ROOT_CAUSE.md` cites each required artifact + contains
   the serial table + the H3/scorer verdict + the GOOD gate + the no-fix boundary.
3. **`findings.md` reconciled** — the early AC-1 table now shows production DS serial **0.013** (not blank)
   and a DSA-radix-off serial row, matching the R21 section (Codex R21-review #7).

## Verification (the self-check fires)
- `ac8_selfcheck.py` PASSES on the final package (verdict "AC-8 PACKAGE COMPLETE", exit 0).
- It ABORTS on each negative, then restored: a removed required citation in `ROOT_CAUSE.md`, a blanked core
  serial cell, and a wrong `run_meta.selected_vs_total.artifact`.

## Files Changed (committed `762330437`)
- `development/loop13/ROOT_CAUSE.md` (regenerated), `development/loop13/ac8_selfcheck.py` (new),
  `development/loop13/evidence/findings.md` (AC-1 reconcile), `evidence/evidence_table.md` + `evidence/meta/*`
  (head-sha refresh — generator blob unchanged).

## Validation
- `ac8_selfcheck.py` exit 0; full CPU suite — `ac3_1_materialized_k_equality`, `ac4_garbage_counters`,
  `ac2_1_forced_all_assertions`, `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `verify_ac2_3 .sglang_ds_scorecap_sparse`
  (AC-2.3 artifact unchanged), `test_reference_selectors` (5/5) — **all exit 0**.
- `build_ledger.py` → provenance consistent. No GPU this round. No `.humanize`/`.pt` committed. No
  selection/adapter **fix**.

## Remaining Items
- **None for the diagnosis deliverable.** All ACs are MET (AC-1/2/3/4/5/8) or justified-moot (AC-7); AC-6 is
  packaged into the AC-8 verdict. Queued cleanup (plan-term comments in retained diagnostics; reference-mode
  CUDA-graph safety outside loop13; the `ac4_garbage_counters.py --arm <non-prod>` default-dir footgun)
  remains non-blocking and is out of scope for the diagnosis loop.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: Capstone packaging round — regenerate the writeup from the final committed artifacts (not interim
  prose) + a fail-closed self-check that asserts the evidence is complete AND cited before the diagnosis can
  stand. This is an application of the already-captured `reconcile-generated-surfaces` + fail-closed-gate
  lessons (write from the authoritative surfaces; gate refuses an incomplete package), not a new technique.

## Goal Tracker Update Request

### Requested Changes (already applied to the mutable section):
- Plan Version → 27 (Round 22); added the Round-22 evolution row.
- task13 (adversarial verify) + task14 (root-cause writeup) → done (R22).
- **AC-8 CLOSED**; the diagnosis loop is complete — all ACs MET or justified-moot.

### Justification:
Codex's R21-review named AC-8 the sole remaining mainline gap and `ROOT_CAUSE.md` as pre-R21. The writeup is
now regenerated from the complete, fail-closed evidence package — serial+batched table, artifact-backed
selected-vs-total, and direct citations to every corroboration artifact — with a self-check that refuses an
incomplete or uncited package. The ranked verdict (dense H3 + sparse raw-dot scorer lock interacting with
current-slot, GOOD gate, not H0/H2, AC-7 moot) is tied to numeric live evidence, and the diagnosis-loop
boundary (no fix landed, recommendation only) is preserved. The loop's deliverable is complete.
