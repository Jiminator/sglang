# Round 16 Summary

Mainline: **repair the R15 AC-4 production scored garbage artifact** (it was the WRONG dataset) and make the
forced-all capture impossible to accept as scored evidence again. Diagnostic/evidence-integrity only; no
selection/adapter fix.

## Root Cause of the R15 Regression
My R15 "validation suite" ran `ac4_garbage_counters.py` with **NO ARGS**. Its `DEFAULT_DIR` was the
forced-all capture `.sglang_ds_forcedall`, so that no-arg run **overwrote** the correct scored artifact with
forced-all dense-only data (61776 rows, `current_slot_unwritten=61776`, source `.sglang_ds_forcedall`) and I
committed it. The reducer **failed open** on the missing sparse regime, so the wrong file exited 0. This is
the exact no-arg-reducer-over-wrong-default-dir mistake I had just caught for `verify_ac2_3` the same round
but missed here (for ac4 the wrong default dir existed and the reducer didn't require both regimes).

The raw scored capture `.sglang_ds_garbage` (79248 `.pt` = 41808 dense + 37440 sparse) was **intact on
disk**, so this was repaired entirely offline — no GPU re-run.

## Work Completed (the 15-review required plan, all three items)
1. **`ac4_garbage_counters.py`** — `DEFAULT_DIR` → `evidence/.sglang_ds_garbage` (the scored capture, NOT
   the forced-all control); the report now stamps `source_dir_basename`; the reducer **fails closed**
   (exit 2) unless BOTH `dense` and `sparse` regimes are present with rows>0, and **does NOT write the JSON**
   in that case — so a wrong-dir / no-arg run can never clobber the canonical artifact (the verify_ac2_3
   lesson, now enforced here too).
2. **`evidence/ac4_garbage_counters.json`** — regenerated from `.sglang_ds_garbage`: **41808 dense + 37440
   sparse** scored rows, real garbage 0 in both regimes, `current_slot_unwritten=0` in both, source basename
   `.sglang_ds_garbage`.
3. **`build_ledger.py`** — new `validate_scored_garbage_artifact()` LOADS the JSON and asserts
   `arm==production_ds`, `source_dir_basename==".sglang_ds_garbage"`, both regimes rows>0,
   `real_garbage_total==0` both, and `current_slot_unwritten==0` both (the footer/findings prose claims the
   current slot is excluded from the scored selection — guard it) **before** attaching
   `garbage_counters_artifact` to production_ds; it records the validated dense/sparse summary on the arm.

`findings.md` prose already stated the correct 41808/37440 numbers (only the JSON file had been wrong), so it
needed no change; `evidence_table.md` + `evidence/meta/arms/*.json` + `run_meta.json` were regenerated.

## Verification (the guards actually fire)
- Forced-all dir → reducer **exit 2**, and the canonical JSON is **untouched** (verified before/after source
  unchanged).
- No-arg run now defaults to `.sglang_ds_garbage` → correct 41808/37440 clean artifact, exit 0.
- A deliberately-injected forced-all-style artifact (`source_dir_basename=.sglang_ds_forcedall`, dense-only)
  → `build_ledger.py` **aborts** with `AssertionError` (exit 1); restored the good artifact, ledger then
  regenerates `provenance consistent`.

## Files Changed (committed `3238c78dc`)
- `development/loop13/ac4_garbage_counters.py` (default dir + both-regimes fail-closed + source stamp),
  `development/loop13/build_ledger.py` (`validate_scored_garbage_artifact()` + wiring),
  `development/loop13/evidence/ac4_garbage_counters.json` (regenerated, correct scored data),
  `development/loop13/evidence/evidence_table.md` + `evidence/meta/arms/*.json` + `evidence/meta/run_meta.json`
  (regenerated: generator-blob bump + production_ds `garbage_counters_validated`).

## Validation
- Full CPU suite, run with **explicit args** (no blind no-arg reducer runs this time): `ac4_garbage_counters`
  (→`.sglang_ds_garbage`, CLEAN), `ac2_1_forced_all_assertions` (→`.sglang_ds_forcedall`, 61776/61776 PASS),
  `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`, `ac6_score_reduce_corrob`, `ac2_2_head_agg`,
  `ac4_sample_ids`, `verify_ac2_3 .sglang_ds_scorecap_sparse` (4992 pruning rows; committed AC-2.3 artifact
  unchanged), `test_reference_selectors` (5/5) — **all exit 0**.
- `py_compile` clean; `build_ledger.py` → provenance consistent; production_ds carries the validated summary.
- No `.pt`/`.humanize` committed. One-server rule moot (offline repair; no server launched). No selection/
  adapter **fix**.

## Remaining Items (for AC-8 COMPLETE)
- **AC-4** garbage counters on the REFERENCE arms (`ref_faithful`/`ref_cosine`) — needs a fresh GPU capture.
- **AC-2.4** recall-oracle@2048 (NIAH-only).
- **AC-3.1** captured decode-row materialized fp32 `K_label` selected-index equality (latent-VALUE capture).
- **AC-4** serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial) +
  selected-vs-total gaps.
- **AC-8** final root-cause writeup.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260621-forced-include-vs-scored-exclude-complementary-h3
- Notes: My R15 lesson already warned (for verify_ac2_3) "if the ephemeral capture dir is absent, do NOT
  re-run the reducer over a stale/partial dir." R15 then violated the FLIP side of the same rule: I ran the
  scored reducer with NO ARGS and its DEFAULT_DIR pointed at the WRONG (forced-all) capture, silently
  overwriting the good artifact, which then failed OPEN on the missing sparse regime. Strengthened the
  lesson: (1) a reducer's DEFAULT_DIR must point at the dir whose data matches the artifact's CLAIMED
  identity (scored reducer → scored dir), never a sibling control dir; (2) a regime/structure check must
  fail CLOSED and must NOT write the artifact when it fails, so a wrong-dir run can't clobber the canonical
  one; (3) the downstream consumer (the ledger) must LOAD and re-validate the artifact's self-described
  provenance (source basename, both regimes, counters) before trusting it — never wire a path by name alone.
  A "validation suite" that blindly runs every reducer with no args is itself a footgun: run reducers with
  explicit args, or make their defaults safe AND fail-closed.

## Goal Tracker Update Request

### Requested Changes (already applied to the mutable section):
- Plan Version → 18 (Round 16); added the Round-16 plan-evolution row.
- task9 → partial (R16): production_ds scored garbage counters now VALID + guarded; reference arms / serial /
  selected-vs-total remain.
- Marked the 15-review blocking issue ("R15 production scored garbage artifact generated from the forced-all
  capture") **RESOLVED (R16)** with the verification evidence.

### Justification:
The committed evidence is now the exact state claimed (production scored, dense+sparse, real garbage 0,
current slot excluded), tied to the actual `.sglang_ds_garbage` scored runtime path, and the fail-open hole
that let the forced-all artifact through is closed at both the reducer (write-side) and the ledger
(read-side). This restores AC-4/AC-8 ledger trustworthiness. The remaining AC-4 reference-arm garbage / serial
cells / selected-vs-total, plus AC-2.4 / AC-3.1 / AC-8, are the active close-out sequence — not deferrals.
