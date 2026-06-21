# Round 16 Contract

Round 15 REGRESSED: the committed `evidence/ac4_garbage_counters.json` is the WRONG dataset. My R15
"validation suite" ran `ac4_garbage_counters.py` with **no args**, whose `DEFAULT_DIR` is the forced-all
capture `.sglang_ds_forcedall`; that no-arg run OVERWROTE the correct scored artifact with forced-all
dense-only data (61776 rows, `current_slot_unwritten=61776`, source `.sglang_ds_forcedall`), and I then
committed it. The reducer failed OPEN on the missing sparse regime, so the wrong file exited 0. This is the
same no-arg-reducer-over-wrong-default-dir mistake I caught for `verify_ac2_3` but missed here.

The raw scored capture `.sglang_ds_garbage` (79248 .pt = 41808 dense + 37440 sparse) is **intact on disk**,
so this is repairable OFFLINE — no GPU re-run.

## Mainline Objective (exactly one)
**Repair the AC-4 production scored garbage artifact and make the forced-all capture impossible to accept as
scored evidence again.** Fix the reducer default + add a both-regimes fail-closed check + stamp the actual
source basename, regenerate the committed JSON from `.sglang_ds_garbage`, and add a `build_ledger.py` guard
that LOADS and VALIDATES the artifact (source basename, dense+sparse rows>0, real garbage 0 both,
current_slot_unwritten 0 both) before wiring it onto `production_ds`.

## Target ACs
- **AC-4** (primary): valid, committed, reproducible production_ds scored length-cap garbage counters
  (dense + sparse), tied to the actual `.sglang_ds_garbage` scored runtime path.

## Blocking Side Issues (these ARE the mainline)
- (15-review blocker, goal-tracker line 138) `ac4_garbage_counters.py` default points at `.sglang_ds_forcedall`
  and fails open on a missing regime; `build_ledger.py` wires the artifact without loading/validating it.

## Queued Side Issues (documented, OUT OF SCOPE this round)
- AC-4 garbage counters on the REFERENCE arms (`ref_faithful`, `ref_cosine`) — needs a fresh GPU capture.
- AC-2.4 recall-oracle@2048 (NIAH-only).
- AC-3.1 captured decode-row materialized fp32 `K_label` selected-index equality (needs latent-VALUE capture).
- AC-4 serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial) +
  selected-vs-total gaps.
- AC-8 final root-cause writeup.
- `serve.sh` usage/help text omits `ds_garbage` and other newer modes; plan-term comment cleanup.

## Approach
1. `ac4_garbage_counters.py`: change `DEFAULT_DIR` to `evidence/.sglang_ds_garbage`; add
   `source_dir_basename` to the report (set from the actual capture dir); fail closed unless BOTH `dense`
   and `sparse` regimes are present with rows>0 (this scored reducer requires both regimes).
2. Regenerate `evidence/ac4_garbage_counters.json` from `.sglang_ds_garbage`; confirm 41808 dense + 37440
   sparse, real garbage 0 both, `current_slot_unwritten=0` both, source basename `.sglang_ds_garbage`.
3. `build_ledger.py`: before attaching `garbage_counters_artifact` to `production_ds`, LOAD the JSON and
   assert `arm == "production_ds"`, `source_dir_basename == ".sglang_ds_garbage"`, dense+sparse rows>0,
   `real_garbage_total == 0` both regimes, and `current_slot_unwritten == 0` both regimes (the footer prose
   claims the current slot is excluded from the scored selection — guard it). Fail closed; verify it fires.
4. Regenerate `findings.md` numbers, `evidence_table.md`, and `evidence/meta/arms/*.json` only AFTER the
   JSON is correct.

## Concrete Success Criteria
1. `ac4_garbage_counters.py` defaults to `.sglang_ds_garbage`, stamps `source_dir_basename`, and fails
   closed (exit 2) unless both dense AND sparse regimes are present with rows>0 (verified: a no-arg or
   forced-all-dir run now exits 2, not 0).
2. Committed `evidence/ac4_garbage_counters.json`: source basename `.sglang_ds_garbage`, dense 41808 +
   sparse 37440 rows, real garbage 0 both, `current_slot_unwritten=0` both.
3. `build_ledger.py` loads + validates the production_ds artifact before wiring it; a deliberately-wrong
   source/regime/count makes ledger generation fail (verified the guard fires, then reverted).
4. `findings.md`/`evidence_table.md`/arm JSONs reflect the corrected scored result; full CPU validation
   suite passes; provenance consistent. Commit; round-16-summary with BitLesson Delta + Goal Tracker
   Update Request. No selection/adapter FIX (diagnostic/guarded only). No exit by lying / editing loop
   state / cancel-rlcr-loop.
