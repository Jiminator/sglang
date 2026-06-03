# Round 12 Summary

## Mainline objective (met)
Closed blocking side issue **#L**: made the AC-12 NIAH gate **artifact-safe** so a server rejection
still produces a durable per-gate JSON, regenerated the missing `ac12_niah_65536_*.json` on
hardware, and corrected `ac12_analysis.md` + `evidence_bundle.md`. task14/task15 evidence is now
complete. The AC-12 verdict is unchanged — **HARD FAIL, loop4-compatible MVP not complete** (not
reclassified as directional; DEC-7 is AC-11-only).

## Work completed

### #L — AC-12 NIAH gate is artifact-safe (mainline)
`_run_niah()` called `_generate()` directly, so when DS rejected the ~70K-token 64K prompt with
HTTP 400 (prompt exceeds the mem-0.6 KV pool of 53,056 tokens) the `urllib.error.HTTPError` escaped
before `_niah_assert()` reached `_record_artifact()` — leaving no `ac12_niah_65536_*.json`. Fix:
- **`_GenAttempt`** (`text, ok, http_status, error, body`) + **`_generate_attempt()`** wrap
  `_generate` so a 4xx/5xx (`HTTPError`) or transport failure (`URLError`) is captured, not raised.
- **`_run_niah()`** collects DSA-then-DS attempts, summarizes served counts + the first error per
  side, and computes recall over `num_prompts` (an unservable prompt is a miss). `_NIAHRunResult`
  gains `dsa_served`/`ds_served`/`dsa_error`/`ds_error`.
- **`_niah_assert()`** ALWAYS records the per-length artifact (served counts, per-side error,
  `verdict`) **before** asserting, and the failure message distinguishes a recall miss from an
  admission failure. 4K/16K/MMLU behavior is unchanged when no server error occurs.
- New registered regression **`test_niah_64k_ds_rejection_records_failure_artifact`**: patches
  `_generate` so DSA returns the needle and DS raises `HTTPError(400)`; asserts the gate fails
  cleanly (1 failure, 0 errors) and records exactly one `niah_65536` artifact with `verdict=FAIL`,
  `ds_served=0`, and the DS error body.

### Hardware 64K rerun (mainline)
Reran the AC-12 64K gate on the same locked two-node operating point (DS radix-on via fixture,
mem 0.6, node 0; DSA radix-on, mem 0.85, node 1, bound `0.0.0.0`, reached cross-node). Durable
artifact `ac12_results/ac12_niah_65536_20260529T093912Z.json`:

```
dsa_served=20  dsa_hits=20  dsa_recall_pct=100.0
ds_served=0    ds_recall_pct=0.0   delta_pct=100.0   verdict=FAIL
ds_error="HTTP 400 ... Input length (69970 tokens) exceeds the maximum allowed length (53050 tokens)"
```

The test fails cleanly (assertion, not an uncaught exception). `ac12_analysis.md` +
`evidence_bundle.md` now reference all four per-gate JSONs and phrase the DSA 20/20 result from the
artifact (no longer overstating coverage).

### Plan-term reword (queued #4, folded in)
Reworded the Round-11-added plan-specific terms (`AC-12`/`AC-Q`/`BL-...`/"Option B") in the
`test_double_sparsity_v32.py` `_generate`/NIAH comments and both `serve_*.sh` HOST-knob comments
into behavior-based wording. Only the pre-existing file-header "Locked Option B operating point
(plan §13/DEC-1)" lines remain (predate Round 11).

## Files changed
- `test/manual/test_double_sparsity_v32.py` — `_GenAttempt`/`_generate_attempt`,
  error-aware `_run_niah`, record-before-assert `_niah_assert`; comment reword.
- `test/registered/unit/manual/test_ac12_helpers.py` — new DS-rejection regression.
- `development/serve_native_nsa.sh`, `development/serve_double_sparsity.sh` — HOST-comment reword.
- `runs/20260528_dsv32_mvp/` — `ac12_results/ac12_niah_65536_*.json` (new), updated
  `ac12_results/ac12_pytest_summary.txt`, `ac12_analysis.md`, `evidence_bundle.md`.
- Commits `d2f48bbd4` (#L code + regression + reword), `cc50bae38` (64K artifact + doc updates).
  Both pushed.

## Validation
- **408 CPU tests pass** (`test_ac11_comparator` + `test_double_sparsity_unit` +
  `test_dsv32_quality_smoke_sequential` + `test_option_b_scripts` + `test_ac12_helpers`); +1 from
  the new #L regression.
- Hardware: 64K gate reran to a clean failure (`1 failed`), producing the durable
  `ac12_niah_65536_*.json`. Both servers shut down afterward; both nodes' GPUs freed; the
  pre-existing port-30000 router was already down at round start and was not touched.

## Remaining Items
- **No mainline work remains** — all plan tasks (task1–task15) are executed and AC-12 evidence is
  now complete. AC-12 is a recorded hard failure; per the Ultimate Goal the deliverable is a
  TIER-1 smoke milestone with TIER-2 loop4 quality not met (recorded, not a build-break).
- **Queued (out of scope, documented):** (a) comparator per-side `mem_fraction_static` validation
  hole — tighten when the comparator is next touched; (b) AC-11 directional performance follow-up
  (TokenLabelTable / KV-budget — the same lever bounds AC-12 64K admission); (c) stale
  `calibrate.py` `--tp 1` recipe docstring (doc-accuracy only); (d) pre-existing "Option B" header
  lines in `serve_*.sh` (predate Round 11; reword if those headers are next edited).

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-gate-record-artifact-before-raise
- Notes: Added the reusable lesson behind #L — a paired hardware quality/eval gate that records its
  per-case artifact only AFTER a raising measurement call loses all durable evidence when the
  server legitimately rejects a case (here: DS HTTP 400 because the 64K prompt exceeds the mem-0.6
  KV pool). Fix: an error-aware attempt wrapper that captures HTTP/URL errors instead of raising,
  count an unservable case as a recorded miss, and ALWAYS `_record_artifact(...)` before the
  assertion — a server rejection is a recordable hard FAIL, never a silent skip or an uncaught
  error — plus a CPU regression that drives the rejection path. Validated by the new regression
  (408 CPU pass) and the durable hardware 64K artifact.

## Goal Tracker Update Request

### Requested Changes:
- Move **task14 / AC-12** to Completed and Verified — evidence is now complete (all four per-gate
  JSONs present including the durable 64K admission artifact); AC-12 remains a recorded HARD FAIL.
- Move **task15 / evidence bundle** to Completed and Verified — `evidence_bundle.md` now references
  all four AC-12 JSONs and no longer overstates 64K coverage.
- Confirm **#L RESOLVED** (artifact-safe NIAH path + regression + durable hardware 64K artifact +
  analysis/bundle correction).
- Note the **plan-term reintroduction** queued item as addressed for the Round-11 additions (only
  pre-existing header lines remain).

### Justification:
Round 12 fixed exactly the blocking gap Codex raised: the AC-12 64K subgate now produces a durable
per-gate JSON (`ac12_niah_65536_*.json`) recording the DSA 20/20 reference and the DS HTTP-400
admission failure with `verdict=FAIL`, and a registered regression proves the failure path records
evidence and fails cleanly. The AC-12 verdict is unchanged (HARD FAIL; loop4-MVP not complete) — no
immutable AC or threshold was modified, and the quality failure was not reclassified as directional.
All plan tasks are now executed with complete evidence; the remaining items are explicitly-queued,
non-blocking cleanups.
