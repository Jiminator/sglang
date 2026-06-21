# Round 22 Contract

Round 21 was ADVANCED; Codex now marks AC-1/AC-2/AC-3/AC-4/AC-5 as MET. The SOLE remaining mainline gap is
**AC-8** — the final root-cause writeup regenerated from the complete evidence package, plus a self-check
gate. This is the capstone round (CPU-only; all evidence is already committed). No GPU.

## Mainline Objective (exactly one)
**Regenerate `development/loop13/ROOT_CAUSE.md` from the FINAL evidence package and add a fail-closed AC-8
self-check** (`ac8_selfcheck.py`) that refuses AC-8 completion unless the evidence is complete and the
writeup cites it.

## Target ACs
- **AC-8** (primary): per-arm GSM8K evidence table (serial+batched) + recall-oracle/selected-index
  corroboration + the H0/H1/H2/H3 verdict + a research-vs-fix recommendation — explicitly NOT a fix.

## Blocking Side Issues (these ARE the mainline)
- `ROOT_CAUSE.md` is a pre-R21 writeup: headline "Round 1" / scope "Round 7", batched-only per-arm table,
  and it does not cite the R14-R21 artifacts (forced_all_assertions, ac2_4_recall_oracle,
  ac3_1_materialized_k_selected_index_equality, ac4_selected_vs_total, ac4_garbage_counters*). It must be
  regenerated from the final package.
- `findings.md` AC-1 table still shows production DS serial sparse blank even though the R21 section has
  0.013 (stale-surface reconcile, Codex #7).

## Queued Side Issues (documented, OUT OF SCOPE this round)
- Plan-term comment cleanup (`AC-*`/`H3`) in retained diagnostics; reference selector CUDA-graph safety
  outside loop13; `ac4_garbage_counters.py --arm <non-prod>` default CAPDIR footgun.

## Approach
1. **Rewrite `ROOT_CAUSE.md`** from the final evidence package: refresh the headline/scope (no stale
   round labels); the per-arm table includes BOTH batched AND serial (R21) for the core arms +
   selected-vs-total (artifact-backed) + the garbage-counter and selector-behavior summary; cite the
   corroboration artifacts directly — `forced_all_assertions.json` (AC-2.1, H3 on the `_ds_slot_written`
   bitmap), `ac2_4_recall_oracle.json` (AC-2.4, sparse recall@2048=0.41 scorer-driven, dense 1.0),
   `ac3_1_materialized_k_selected_index_equality.json` (AC-3.1, captured-row raw-dot==materialized-K
   96/96 both regimes), `ac4_garbage_counters*.json` (clean adapter path across all arms),
   `ac4_selected_vs_total.json`, `evidence_table.md`, `ac6_bisection_matrix.json`. Keep the verdict (two
   regressions: dense=H3 current-slot exclusion; sparse=raw-dot scorer lock interacting with current-slot),
   the GOOD gate, AC-7 moot, and the "no fix landed, recommendation only" boundary.
2. **`ac8_selfcheck.py`** (fail-closed): assert the AC-4 core serial cells exist in `evidence_table.md` /
   per-arm JSONs, `run_meta.selected_vs_total.artifact == "evidence/ac4_selected_vs_total.json"`, the
   recall-oracle + materialized-K run_meta summaries are present, and `ROOT_CAUSE.md` contains the final
   serial table + cites each required artifact. Nonzero exit on any miss.
3. **Reconcile `findings.md`** AC-1 table: production DS serial sparse 0.013 (not blank).

## Concrete Success Criteria
1. `ROOT_CAUSE.md` regenerated from the final package: no stale "Round 1/7" labels; the per-arm table has
   serial+batched for the core arms + selected-vs-total; cites forced_all_assertions / ac2_4_recall_oracle /
   ac3_1_materialized_k_selected_index_equality / ac4_selected_vs_total / ac4_garbage_counters /
   evidence_table / ac6_bisection_matrix; names the H3+scorer verdict tied to numbers; GOOD gate; AC-7 moot;
   no-fix boundary + recommendation.
2. `ac8_selfcheck.py` exits 0 against the final package and FAILS (exit nonzero) on a deliberately-broken
   writeup (missing artifact citation) or a missing AC-4 serial cell (verified, then restored).
3. `findings.md` AC-1 table reconciled (no blank production DS serial sparse).
4. Full CPU validation suite passes; `build_ledger.py` provenance consistent. Commit; round-22-summary with
   BitLesson Delta + Goal Tracker Update Request. No selection/adapter FIX. No exit by lying / editing loop
   state / cancel-rlcr-loop.
