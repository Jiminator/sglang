# Round 17 Summary

Mainline: **AC-4 length-cap garbage counters on the served REFERENCE arms (`ref_faithful` + `ref_cosine`)**
— the last AC-4 garbage-counter gap (Codex R16-review item #3). Completes garbage counters across ALL
primary served DS arms. Diagnostic/guarded instrumentation only; no selection/adapter fix.

## Feasibility (verified before spending GPU)
The reference selector path (`reference_rawdot`/`reference_cosine`, deepseek_v2.py:2443) produces
`selected_indices` and falls through to the common `logical_to_physical` adapter (2693) and the
`forced_all_assert` hook (2722), which is gated **only** on `forced_all_assert` — not on
`forced_all_dense_control` or `selector_impl`. So serving a reference arm with `forced_all_assert:true`
dumps its real scored selection exactly as `ds_garbage` did for production. (Confirmed empirically: the
first capture produced 79248 `.pt` records with all required fields.)

## Work Completed
1. **`serve.sh`** — add `ref_faithful_garbage` (ref_faithful config + `forced_all_assert`, eager) and
   `ref_cosine_garbage` (ref_cosine config + `forced_all_assert`, eager). Mode-error string updated with all
   current modes.
2. **`ac4_garbage_counters.py`** — add `--arm NAME` (default `production_ds`): per-arm output
   (`ac4_garbage_counters.json` for production_ds, `..._{arm}.json` otherwise); the `arm` field, `ac`/`source`
   strings and verdict are now arm-generic. The both-regimes fail-closed + no-real-garbage checks are
   unchanged (arm-agnostic); the current-slot count is only reported.
3. **GPU capture** — one TP=8 server at a time (`ref_faithful_garbage` then `ref_cosine_garbage`), each a
   small dense (5-shot/4ex/16tok) + sparse (24-shot/4ex/16tok) capture into a per-arm dir, each torn down to
   0 MiB. Reduced each with `--arm`.
4. **`build_ledger.py`** — generalize `validate_scored_garbage_artifact()` → `validate_garbage_artifact(arm)`
   over a `GARBAGE_ARTIFACTS` table (arm → relative path, expected `source_dir_basename`, current-slot
   expectation). production_ds: assert current_slot_unwritten==0 (EXCLUDED); reference arms: assert >0
   (INCLUDED). Loads + validates each artifact before wiring `garbage_counters_artifact` +
   `garbage_counters_validated` onto production_ds / ref_faithful / ref_cosine. `NOT_INSTRUMENTED` + footer +
   `findings.md` updated (no primary served DS arm remains for garbage counters).

## Result (CLEAN — production-excludes vs reference-includes contrast)
| arm | dense rows | sparse rows | real garbage (both) | current_slot_unwritten |
|---|---|---|---|---|
| production_ds (R15/R16) | 41808 | 37440 | **0** | **0** (current slot EXCLUDED) |
| ref_faithful (R17) | 41808 | 37440 | **0** | 41808 / 37440 (= rows; INCLUDED) |
| ref_cosine (R17) | 41808 | 37440 | **0** | 41808 / 37440 (= rows; INCLUDED) |

Across ALL served DS arms the adapter + selected-index path is provably clean (0 duplicate / live-`-1` /
out-of-range / adapter-error / NON-current unwritten, dense AND sparse). The ONLY moving part is whether the
current decode slot is in the selection: production EXCLUDES it (the H3 dense regression), the faithful
references INCLUDE it (`reference_include_current=true`, the recovery). H3 pinned from both sides, on the
real served selection of every arm.

## Files Changed (committed `082510939`)
- `development/loop13/serve.sh` (+2 modes), `development/loop13/ac4_garbage_counters.py` (`--arm`),
  `development/loop13/build_ledger.py` (generalized validator + wiring + footer/NOT_INSTRUMENTED),
  `development/loop13/evidence/ac4_garbage_counters_ref_faithful.json` + `_ref_cosine.json` (new),
  `development/loop13/evidence/ac4_garbage_counters.json` (arm-generic strings; data identical),
  `development/loop13/evidence/findings.md` (reference-arm section), `evidence/evidence_table.md` +
  `evidence/meta/*` (regenerated), `.gitignore` (+2 raw capture dirs).

## Validation
- CPU suite, explicit args (no blind no-arg reducer runs): `ac4_garbage_counters` production + `--arm
  ref_faithful` + `--arm ref_cosine` (all CLEAN), `ac2_1_forced_all_assertions`, `ac6_bisection_matrix`,
  `ac6_corrob_ref_cosine_noinc`, `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`,
  `verify_ac2_3 .sglang_ds_scorecap_sparse` (committed AC-2.3 artifact unchanged), `test_reference_selectors`
  (5/5) — **all exit 0**.
- `build_ledger.py` → provenance consistent; verified the generalized guard ABORTS ledger generation on an
  injected current==0 reference artifact (reference must be >0), then restored.
- One TP=8 server at a time, each torn down to 0 MiB. No `.pt`/`.humanize` committed. No selection/adapter
  **fix**.

## Remaining Items (for AC-8 COMPLETE)
- **AC-2.4** recall-oracle@2048 (NIAH-only).
- **AC-3.1** captured decode-row materialized fp32 `K_label` selected-index equality (latent-VALUE capture).
- **AC-4** serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial) +
  selected-vs-total gaps.
- **AC-8** final root-cause writeup.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260621-forced-include-vs-scored-exclude-complementary-h3
- Notes: The same forced_all_assert capture now generalizes to EVERY served selector variant by toggling one
  config flag — the hook is gated only on `forced_all_assert`, and the reference path falls through to the
  same adapter+hook, so no new instrumentation was needed for the reference arms. Generalizing the reducer
  by `--arm` (per-arm output) and the ledger guard by a per-arm (source_dir_basename, current-slot
  expectation) table makes the provenance check arm-aware: production EXCLUDES the current slot
  (current_slot_unwritten must be 0), the references INCLUDE it (must be >0) — the guard asserts the arm's
  EXPECTED current-slot behavior, turning a previously production-only invariant into a positive/negative
  cross-arm control. Same-shaped artifact, three arms, one validated provenance contract.

## Goal Tracker Update Request

### Requested Changes (already applied to the mutable section):
- Plan Version → 19 (Round 17); added a 16-review row + the Round-17 evolution row.
- task9 → partial (R17): AC-4 garbage counters DONE for ALL primary served DS arms; only serial cells +
  selected-vs-total remain.
- Updated the broad evidence-package blocker: reference-arm garbage now resolved (R17); only selected-vs-total
  + serial cells remain there.

### Justification:
The R16-review named reference-arm garbage counters as the next close-out item. This round produced them on
the real served reference selection by reusing the thrice-verified `forced_all_assert` instrumentation (no
new production code), with a fail-closed, arm-aware ledger guard. AC-4 garbage counters are now complete and
guarded across production + forced-all control + both reference arms, with the production-excludes /
reference-includes current-slot contrast establishing the clean adapter path and pinning H3 from both sides.
Remaining close-out (AC-2.4, AC-3.1, AC-4 serial/selected-vs-total, AC-8) is the active sequence toward
COMPLETE — not deferrals.
