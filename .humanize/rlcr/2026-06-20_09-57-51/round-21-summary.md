# Round 21 Summary

Mainline: **AC-4 evidence-table close-out** — fill the missing strict serial GSM8K cells and replace the
static selected-vs-total literals with an artifact-backed, fail-closed probe. Diagnostic/evidence only; no
selection/adapter fix.

## Work Completed
1. **Serial GSM8K cells** (THREADS=1, one TP=8 server at a time, no PYTHONPATH, completion API, teardown to
   0 MiB between arms) — filled for every AC-4 core arm from real `.out`:

   | arm | dense_b | sparse_b | dense_serial | sparse_serial |
   |---|---|---|---|---|
   | dsa | 0.975 | 0.973 | 0.965 | 0.947 |
   | dsa_noradix | 0.960 | 0.940 | **0.965** | **0.973** |
   | production_ds | 0.620 | 0.000 | 0.655 | **0.013** |
   | ref_faithful | 0.950 | 0.013 | **0.965** | **0.013** |
   | ref_cosine | 0.940 | 0.940 | **0.965** | **0.947** |

   **Serial ≈ batched** everywhere → the regression is NOT batch-dependent. The serial cells corroborate the
   verdict from a second mode: production_ds **dense** serial 0.655 stays collapsed while the
   current-slot-INCLUDED reference arms get dense ~0.965 (the dense gap tracks current-slot exclusion, H3),
   and **sparse** stays scorer-driven (rawdot 0.013 vs cosine 0.947).
2. **`ac4_selected_vs_total_probe.py`** (new, fail-closed) — probes the live server's
   `meta_info["double_sparsity"]` per arm per regime (a dense <top_k + a sparse >top_k `/generate`), asserts
   dense `selected==total`, sparse `selected<total`, `dense_fallback==0`, and atomically updates
   `evidence/ac4_selected_vs_total.json`. Result: production_ds / ref_faithful / ref_cosine each dense
   334/334, sparse 2048/3692, dense_fallback 0 — DS genuinely active (keeps all in dense, prunes in sparse).
3. **`build_ledger.py`** — wired the new serial `.out` labels; **replaced the static `ds={...}` literals**
   for the core DS arms with values loaded from the artifact via `validate_selected_vs_total_artifact()`
   (fail-closed on the DS-active invariants); added a guard that **rejects a BLANK serial cell** for any AC-4
   core arm with a wired label; records `selected_vs_total` provenance in `run_meta`. `findings.md` records
   the result.

## Verification (the guards fire)
All 6 negatives make `build_ledger.py` ABORT, then restore → provenance consistent:
- selected-vs-total: sparse `selected==total` (no pruning), `dense_fallback!=0`, missing core arm, dense
  `selected!=total`, and a missing artifact.
- a hidden serial `.out` (blank cell) for a core arm.

## Files Changed (committed `cc9865440`)
- `development/loop13/ac4_selected_vs_total_probe.py` (new), `development/loop13/build_ledger.py` (serial
  labels + artifact-backed selected-vs-total + validate gate + blank-serial guard + run_meta provenance),
  `development/loop13/evidence/ac4_selected_vs_total.json` (new), `development/loop13/evidence/evidence_table.md`
  + `evidence/meta/*` (regenerated — serial cells filled, selected-vs-total from the artifact),
  `development/loop13/evidence/findings.md`. (Serial `.out` run logs are gitignored, per the established
  convention — the derived scores are committed in the per-arm JSONs/table.)

## Validation
- CPU suite, explicit args: `ac3_1_materialized_k_equality`, `ac4_garbage_counters`,
  `ac2_1_forced_all_assertions`, `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `verify_ac2_3 .sglang_ds_scorecap_sparse`
  (committed AC-2.3 artifact unchanged), `test_reference_selectors` (5/5) — **all exit 0**.
- `build_ledger.py` → provenance consistent. One TP=8 server at a time (4 serial boots + 3 probe boots),
  each torn down to 0 MiB; no PYTHONPATH; no `.out`/`.humanize` raw artifacts committed. No selection/adapter
  **fix**.

## Remaining Items (for COMPLETE)
- **AC-8** final root-cause writeup — the LAST item. Regenerate `ROOT_CAUSE.md` from the final evidence
  package (per-arm serial+batched table, AC-2.1/AC-2.4/AC-3.1/AC-4/AC-6 artifacts, the H0/H1/H2/H3 verdict,
  the "diagnosis loop, no fix" scope, the recommendation), and add a self-check that refuses AC-8 while the
  AC-4 core serial cells or the selected-vs-total artifact are absent.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: Mechanical evidence-table close-out — serial GSM8K runs via the existing guarded harness +
  an artifact-backed `meta_info["double_sparsity"]` probe + the now-standard fail-closed ledger gates
  (validate-before-render, reject-blank-cell). No new reusable technique beyond the already-captured
  fail-closed-artifact + ledger-tracks-serve-mode lessons; the one operational note (the sparse probe prompt
  must exceed top_k or selected==total trivially — caught + fixed by the fail-closed dense<total invariant)
  is a parameter detail, not a lesson.

## Goal Tracker Update Request

### Requested Changes (already applied to the mutable section):
- Plan Version → 25 (Round 21); added the Round-21 evolution row.
- task1 (AC-1 serial) → done; task9 (AC-4) → AC-4 CLOSED for the core arms (batched + serial + garbage +
  artifact-backed selected-vs-total, all guarded).
- Marked the broad "evidence package lacks selected-vs-total / serial cells" blocker RESOLVED.

### Justification:
Codex's R20-review named the blank serial cells + static selected-vs-total as the AC-4 gap. Both are now
filled from real runs and provenance-backed with fail-closed gates (validate-before-render + reject-blank-
cell, all 6 negatives verified to abort). The serial cells also corroborate the verdict from a second mode
(serial≈batched; dense gap = current-slot, sparse = scorer). The ONLY remaining loop item is the AC-8 final
root-cause writeup.
