# Round 12 Contract

## Mainline Objective (exactly one)
**Complete AC-12's evidence: make the NIAH gate artifact-safe so a server rejection still produces
a durable per-gate JSON, then regenerate the missing `ac12_niah_65536_*.json` on hardware and
correct `ac12_analysis.md` + `evidence_bundle.md`.** This closes blocking side issue **#L** and
makes task14/task15 verifiable-complete. The AC-12 verdict itself is unchanged (HARD FAIL — DS
NIAH long-context recall is top_k-bounded and DS cannot admit 64K at mem 0.6); this round only
makes the 64K failure's evidence durable, as the Round-11 contract required.

## Target AC
- **AC-12** (completing evidence; verdict stays HARD FAIL, not reclassified — DEC-7 is AC-11-only).

## Blocking Side Issue (drives the code change)
- **#L: AC-12 NIAH 64K failure path loses the per-gate artifact.** `_run_niah()` calls
  `_generate()` directly; when DS rejects the 64K prompt with HTTP 400 the `urllib.error.HTTPError`
  escapes before `_niah_assert()` reaches `_record_artifact()`, so no `ac12_niah_65536_*.json`
  exists. Fix per Codex's directive:
  1. Add an error-aware attempt shape (`_GenAttempt`: `text, ok, http_status, error, body`).
  2. `_run_niah()` collects attempts (DSA then DS), summarizes served counts + first error per
     side, computes recall over `num_prompts` (an unservable prompt counts as a miss), and returns
     the served/error details. Server errors are NOT silently converted to a pass.
  3. `_niah_assert()` ALWAYS records the artifact before asserting; the 64K payload includes
     `length_tokens=65536`, `num_prompts`, DSA served/hits, DS served/hits, the DS HTTP
     status/message/body, threshold, and `verdict=FAIL`. Failure message distinguishes a recall
     miss from an admission failure.
  4. Add a registered CPU regression (`test_ac12_helpers.py`): patch `_generate` so DSA returns the
     needle and DS raises `urllib.error.HTTPError`; assert `_niah_assert(65536)` fails cleanly
     (failure, not error) and records exactly one `niah_65536` artifact with the DS error details.
  5. Rerun the CPU suite; then rerun at least the AC-12 64K gate on hardware (same locked DS/DSA
     operating point) and copy the new `ac12_niah_65536_*.json` into `ac12_results/`.
  6. Update `ac12_analysis.md` + `evidence_bundle.md` to reference the 64K JSON; phrase the DSA
     result from what the artifact actually records.

## In-scope cleanup (folded in because these files are touched this round)
- Reword plan-specific terms (`AC-12`, `AC-Q`, `BL-...`, "Option B") that Round 11 added to the
  comments/docstrings of `test/manual/test_double_sparsity_v32.py` and the two `serve_*.sh` HOST
  knobs into behavior-based wording (Codex queued #4; "next time those files are touched"). Pure
  comment edits, zero behavior change; the launcher contract test asserts on flags, not comment
  text.

## Queued (explicitly OUT of scope this round)
- Comparator per-side `mem_fraction_static` validation hole — fix when the comparator is next
  touched (AC-12 does not touch the comparator).
- AC-11 directional performance follow-up (TokenLabelTable / KV-budget) — performance/design work;
  not a prerequisite to recording the AC-12 hard failure.
- Stale `calibrate.py` `--tp 1` recipe docstring — doc-accuracy only; mask+SHA are authoritative.

## Success Criteria
1. `_run_niah`/`_niah_assert` capture server rejections; a DS HTTP 400 yields a recorded
   `niah_<L>` artifact with `verdict=FAIL` + DS error details + served counts, and the test still
   fails (no uncaught exception).
2. New registered CPU regression passes and proves exactly one `niah_65536` artifact with error
   details is recorded on the DS-rejects path; full CPU suite stays green.
3. Hardware rerun of the 64K gate produces `runs/20260528_dsv32_mvp/ac12_results/ac12_niah_65536_*.json`
   capturing DSA served/hits + the DS admission error (HTTP 400, KV pool < prompt length).
4. `ac12_analysis.md` + `evidence_bundle.md` reference the 64K JSON; bundle no longer overstates
   `ac12_results/` coverage; AC-12 stays a hard failure and loop4-MVP stays "not complete."
5. 4K/16K/MMLU behavior unchanged (the error-aware path is transparent when no server error).
6. Commit (NO AI authorship) + push. Goal-tracker mutable section updated.

## Out-of-Scope Confirmation
No immutable AC or threshold changed. AC-12 is NOT reclassified as directional. The comparator is
not modified. DS performance is not tuned.
