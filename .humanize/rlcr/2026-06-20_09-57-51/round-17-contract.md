# Round 17 Contract

Round 16 was ADVANCED — the R15 artifact regression is repaired and Codex found no new R16 blocker. The
loop remains NOT_COMPLETE only on original-plan close-out: AC-2.4 recall-oracle, AC-3.1 captured
materialized-K, AC-4 REFERENCE-arm garbage counters + serial cells + selected-vs-total, AC-8 writeup.

This round takes the next decisive, lowest-NEW-code item that reuses the now-thrice-verified
(`R14/R15/R16`) `forced_all_assert` instrumentation: **AC-4 length-cap garbage counters on the REFERENCE
arms** (Codex required-plan item #3). Confirmed feasible: the reference selector path
(`reference_rawdot`/`reference_cosine`, deepseek_v2.py:2443) falls through to `logical_to_physical`
(2693) and the `forced_all_assert` hook (2722), which is gated ONLY on `forced_all_assert` — not on
`forced_all_dense_control` or `selector_impl`. So serving the reference arms with `forced_all_assert:true`
dumps their real scored selection exactly as `ds_garbage` did for production.

## Mainline Objective (exactly one)
**Produce AC-4 length-cap garbage counters for the served REFERENCE arms (`ref_faithful` and `ref_cosine`),
dense + sparse**, by serving each with `forced_all_assert:true` (eager), reducing with the repaired
`ac4_garbage_counters.py` (parameterized by arm), and wiring per-arm validated artifacts into the ledger.
This completes AC-4 garbage counters across ALL served DS arms (production + forced-all control already
done; references are the last).

## Target ACs
- **AC-4** (primary): per-arm length-cap garbage-rate (duplicate / live-`-1` / unwritten / out-of-range
  physical slots + adapter error_count) for `ref_faithful` AND `ref_cosine`, dense AND sparse.

## Blocking Side Issues (these ARE the mainline)
- `ac4_garbage_counters.py` hardcodes `arm=production_ds`, the output filename, and (via the ledger guard)
  `current_slot_unwritten==0`. The reference arms INCLUDE the current slot (`reference_include_current=true`),
  so the reducer must be parameterized by arm (per-arm output) and the ledger validator generalized so a
  current-slot-INCLUDED arm is validated correctly (production excludes → count 0; reference includes →
  count > 0; BOTH must have zero real/non-current garbage).

## Queued Side Issues (documented, OUT OF SCOPE this round)
- AC-2.4 recall-oracle@2048 (NIAH-only).
- AC-3.1 captured decode-row materialized fp32 `K_label` selected-index equality (needs latent-VALUE capture).
- AC-4 serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial) +
  selected-vs-total gaps.
- AC-8 final root-cause writeup.
- `serve.sh` usage/help text completeness for all newer modes; plan-term comment cleanup.

## Approach
1. `serve.sh`: add `ref_faithful_garbage` (ref_faithful config + `"forced_all_assert": true`, eager) and
   `ref_cosine_garbage` (ref_cosine config + `"forced_all_assert": true`, eager). Update the mode-error string.
2. `ac4_garbage_counters.py`: add `--arm NAME` (default `production_ds`); set the report `arm` from it and
   write per-arm output (`evidence/ac4_garbage_counters.json` for production_ds; `..._{arm}.json` otherwise).
   Keep the both-regimes fail-closed + no-real-garbage checks (arm-agnostic). The current-slot count is only
   reported, not asserted, in the reducer.
3. GPU: one TP=8 server at a time. Boot `ref_faithful_garbage`, set `SGLANG_DS_FORCEDALL_ASSERT_DIR`=
   `.sglang_ds_ref_faithful_garbage`, drive a SMALL dense (5-shot) + sparse (24-shot) capture, teardown to
   0 MiB. Repeat for `ref_cosine_garbage` → `.sglang_ds_ref_cosine_garbage`. Reduce each with `--arm`.
4. `build_ledger.py`: generalize `validate_scored_garbage_artifact()` → validate (arm, source_basename,
   both regimes rows>0, real_garbage_total==0 both) for any arm, with an `expect_current_excluded` flag
   (production_ds: assert current==0; reference arms: assert current>0 — the positive control that they
   include the current slot). Wire `garbage_counters_artifact` + `garbage_counters_validated` onto
   `ref_faithful` and `ref_cosine`. Update `findings.md`, `evidence_table.md`, `NOT_INSTRUMENTED` (no served
   DS arms remain for garbage counters). `.gitignore` the two new raw capture dirs.

## Concrete Success Criteria
1. `serve.sh ref_faithful_garbage` and `ref_cosine_garbage` exist (eager, ref config + forced_all_assert).
   One TP=8 server at a time, each torn down to 0 MiB. No `.pt`/`.humanize` committed.
2. `evidence/ac4_garbage_counters_ref_faithful.json` and `..._ref_cosine.json` each record, per regime
   (dense+sparse), the AC-4 garbage counters on real reference scored rows; real (non-current) garbage is a
   number (0 expected); `source_dir_basename` set; fail-closed (verified exit 2 on empty / single-regime).
3. `build_ledger.py` loads + validates BOTH reference artifacts before wiring them onto `ref_faithful`/
   `ref_cosine` (arm, source basename, both regimes, real garbage 0, current-slot-INCLUDED >0); a wrong
   source/regime/count makes ledger generation fail (verified the guard fires, then reverted). `findings.md`
   records the reference-arm garbage result; `NOT_INSTRUMENTED` no longer lists reference-arm garbage.
4. Tests pass; provenance consistent. Commit; round-17-summary with BitLesson Delta + Goal Tracker Update
   Request. No selection/adapter FIX (diagnostic/guarded instrumentation only). No exit by lying / editing
   loop state / cancel-rlcr-loop.
