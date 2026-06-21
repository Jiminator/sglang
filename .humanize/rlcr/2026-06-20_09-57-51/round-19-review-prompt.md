# FULL GOAL ALIGNMENT CHECK - Round 19

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 19 Summary

Mainline: **repair the AC-2.4 recall-oracle fail-closed contract** (both R18-review blockers). The R18
recall numbers were correct, but the producer wrote the canonical JSON before validating, `no_active_trial`
markers weren't fatal, the ledger checks were too weak, and `serve.sh` didn't encode the oracle-dir/CWD
agreement the TP worker depends on. Diagnostic/evidence-integrity only; no selection/adapter fix.

## Work Completed
1. **Producer (`niah_recall_oracle.py`)** — factored reduce+validate into `_reduce_validate_write()`:
   builds the report IN MEMORY, runs the FULL contract, and writes the canonical artifact ONLY via atomic
   `.tmp` → `os.replace` AFTER every check passes — so a partial/failed run leaves the canonical artifact
   UNTOUCHED. Removed the `span_out_of_range`/`exception` whitelist: ANY non-zero failure marker (incl.
   `no_active_trial` and any future name) is fatal. Per-regime invariants: `trials_issued == num`,
   `trials_with_records == trials_issued`, `oracle_records > 0`, `recall_at_2048_records == oracle_records`,
   `selected_contains_needle_records == oracle_records`, `recall_at_2048 == selected_contains_needle_rate`,
   and every issued trial has a non-null server `prompt_tokens`.
2. **Serve harness (`serve.sh`)** — added `LAUNCH_CWD` (default caller `$PWD`, so existing modes are
   unchanged); `ds_recall_oracle` sets `LAUNCH_CWD=$EVID` and the server launches via `( cd "$LAUNCH_CWD" &&
   exec python3 -m sglang.launch_server … )` (LOG/PIDFILE absolute; `exec` keeps `$!` == the server PID for
   teardown). So the TP worker's CWD-default oracle dir (`cwd/.sglang_ds_oracle`) = `$EVID/.sglang_ds_oracle`
   = the driver's default `--oracle-dir`, from ANY caller CWD. Also refreshed the stale usage header.
3. **Consumer (`build_ledger.py`)** — `validate_recall_oracle_artifact()` now asserts the FULL success
   contract before recording `run_meta.recall_oracle_corroboration`: exact `{dense,sparse}` regimes,
   `index_topk==2048`, `source_oracle_dir_basename==".sglang_ds_oracle"`, ZERO failure markers, and per
   regime `trials_with_records==trials_issued>0`, `oracle_records==recall_at_2048_records==
   selected_contains_needle_records>0`, `recall==selected_rate`, non-null prompt-token sample.
4. **Regenerated the committed artifact** via a fresh GPU sweep with the hardened driver, invoked through
   the fixed `serve.sh` **from the repo root** (no manual cd / env export) — proving the LAUNCH_CWD fix:
   records landed in the driver's default oracle dir, driver rc=0, identical numbers (dense 1.0, sparse
   0.4103) now with the full contract fields. One TP=8 server, eager, torn down to 0 MiB.
5. **Nits** — removed the duplicate `.sglang_ds_oracle/` `.gitignore` line; refreshed the serve.sh usage
   header to list all current modes.

## Verification (the guards actually fire)
- **Producer (offline, synthetic sinks)**: clean run → writes (exit 0); a `no_active_trial` marker / a
  missing trial / a null prompt_tokens → **exit 2 and NO canonical write** (atomic-write-after-validate).
- **Consumer (ledger negatives)**: a partial (`trials_with_records < issued`), a failure-marker, a
  missing-regime, a wrong-source-dir, a `recall != selected_rate`, and a null prompt-token-sample artifact
  each make `build_ledger.py` **abort** (AssertionError, exit 1); restored → provenance consistent.
- **Harness**: serve.sh launched from the repo root → worker oracle dir correct → driver got records.

## Files Changed (committed `8a179067d`)
- `development/loop13/niah_recall_oracle.py` (atomic write + all-markers-fatal + count invariants),
  `development/loop13/serve.sh` (LAUNCH_CWD + usage header), `development/loop13/build_ledger.py`
  (strengthened validator), `development/loop13/evidence/ac2_4_recall_oracle.json` (regenerated, full
  contract), `evidence/evidence_table.md` + `evidence/meta/*` (regenerated), `.gitignore` (dedup).

## Validation
- CPU suite, explicit args: `ac4_garbage_counters` (production + ref_faithful + ref_cosine),
  `ac2_1_forced_all_assertions`, `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `verify_ac2_3 .sglang_ds_scorecap_sparse`
  (committed AC-2.3 artifact unchanged), `test_reference_selectors` (5/5) — **all exit 0**.
- `build_ledger.py` → provenance consistent; `bash -n serve.sh` clean. No sink/`.pt`/`.humanize` raw
  artifacts committed. One TP=8 server, eager, torn down to 0 MiB. No selection/adapter **fix**.

## Remaining Items (for AC-8 COMPLETE)
- **AC-3.1** captured decode-row materialized fp32 `K_label` selected-index equality (synthetic proof
  insufficient — needs latent/scales/query capture extension).
- **AC-4** serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial) +
  selected-vs-total gaps.
- **AC-8** final root-cause writeup.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260621-niah-recall-oracle-fail-closed-span-self-verify
- Notes: A fail-closed PRODUCER is only fail-closed if it (1) writes the canonical artifact LAST, via atomic
  rename, ONLY after every check passes — writing-then-checking still leaves a canonical artifact on
  failure (the exact R15 forced-all-vs-scored class, now repeated for the oracle); (2) treats EVERY failure
  marker as fatal, not a hand-picked whitelist — the server records `no_active_trial` as a fail-closed
  marker too, so `sum(failure_markers.values()) == 0` is the right gate, not `span_out_of_range/exception`
  only; and (3) asserts COUNT invariants (issued==with_records, oracle==recall==selected counts,
  recall==selected_rate, non-null prompt-tokens) so a partial run can't pass. The CONSUMER (ledger) must be
  an INDEPENDENT gate asserting the same invariants — never trust a nearby JSON by presence alone. And a
  cross-process diagnostic whose worker resolves a dir from CWD must bake that CWD into the serve harness
  (LAUNCH_CWD via a `( cd … && exec … )` subshell, LOG/PIDFILE absolute), not merely document it, and prove
  it by launching from an arbitrary CWD. Watch the defaultdict footgun: `sum(d[k] for k in WHITELIST)`
  inserts those keys — sum over `d.values()` instead.

## Goal Tracker Update Request

### Requested Changes (already applied to the mutable section):
- Plan Version → 22 (Round 19); added the Round-19 evolution row.
- task4 → done (AC-2.3 R5; **AC-2.4 R18 measured + R19 fail-closed hardened**).
- Marked BOTH R18-review blockers **RESOLVED (R19)** with verification evidence (producer atomic
  write/all-markers-fatal/count invariants; ledger independent gate; serve.sh LAUNCH_CWD proven from repo
  root).

### Justification:
Codex flagged the same evidence-integrity class as R15 (a partial/failure-marker run could leave a canonical
JSON the ledger accepts) plus an unenforced oracle-dir/CWD requirement. The repair makes the producer never
publish on failure, treats every marker as fatal, makes the ledger an independent gate, and bakes the CWD
agreement into serve.sh (proven by a repo-root re-run). The committed artifact is unchanged in substance
(dense 1.0, sparse 0.4103) but now carries the full contract fields. Remaining close-out (AC-3.1, AC-4
serial/selected-vs-total, AC-8) is the active sequence toward COMPLETE — not deferrals.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
8f88e1aef [loop12] R1 evidence: corrected conc-64 perf (1 group, 256/256)
4706b2138 [loop12] document double-sparsity v2 performance numbers
480fd70ba [loop12] add double-sparsity v2 run-and-evaluate runbook
2babc5afa [loop12] add gsm8k evidence + refresh perf verdict numbers
180f6dd6d [loop13] add diagnosis-loop plan, draft, and guarded harness
fc6ac20a7 [loop13] diagnostic DS selectors: fp32 raw-dot reference + forced-all dense control
29ec137bf [loop13] harness: ref/ds_capture/ds_forced_all serve modes + AC-1 baseline evidence
16caf4f5b [loop13] reference selector: gather-then-dequant (full-pool dequant was intractable)
5a1da871d [loop13] verdict: DS dense degradation = current-decode-slot exclusion (H3)
fea920c06 [loop13] reference selector: served cosine + faithful/leak-free ceiling
62ad64346 [loop13] Round 1 verdict FLIP: ceiling GOOD; sparse = raw-dot scorer_norm=off lock
ac479aeb3 [loop13] Round 2: per-arm evidence ledger, baseline consistency, captures, cleanup
29ed825fa [loop13] Round 3: ledger SHA provenance, capture row-identity, exact-join analyzer
393966c02 [loop13] Round 4: AC-2.3 RESOLVED on real captured rows; fail-closed analyzer; ledger provenance
c7b66f04b [loop13] Round 5 (drift recovery): pruning-valid AC-2.3 + AC-6 scorer×current-slot 2×2
8b55dfba3 [loop13] Round 6: AC-6 corroboration + complete per-leg bisection matrix
8281361e7 [loop13] Round 7: measure the score-reduce leg + dense current-slot corroboration
4d874b89e [loop13] gitignore transient DS capture scratch dirs
752752f6d [loop13] Round 8: fix ds_reduce_fp32 metadata; settle AC-2.2 + AC-4 sample IDs
5d48cbd0d [loop13] Round 9: reconcile evidence package — full DS configs + AC-2.2 consistency
75158e505 [loop13] Round 10: finish evidence-package consistency (head-agg + effective DS config)
482ff8083 [loop13] Round 11: AC-4 selector-behavior surface (reference arms != production knobs)
d11e752b8 [loop13] Round 12: render forced-all dense override in the selector-behavior surface
e62112335 [loop13] Round 13: AC-2.1 forced-all dense physical-slot assertions (H3 downstream control)
08caeda27 [loop13] Round 14: repair AC-2.1 — _ds_slot_written + per-step + true KV range (H3 on the bitmap)
e0f28d547 [loop13][R15] AC-4 length-cap garbage counters on the production SCORED DS arm
3238c78dc [loop13][R16] Repair AC-4 production scored garbage artifact + fail-closed provenance guards
082510939 [loop13][R17] AC-4 length-cap garbage counters on the REFERENCE arms (ref_faithful + ref_cosine)
4a16c082a [loop13][R18] AC-2.4 NIAH recall-oracle@2048 corroboration (production DS scorer)
8a179067d [loop13][R19] Harden the AC-2.4 recall-oracle fail-closed contract (producer + consumer + harness)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-18-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-18-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-17-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-17-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-16-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-16-review-result.md


Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Goal Tracker Audit (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/goal-tracker.md and verify:

### 1.1 Acceptance Criteria Status
For EACH Acceptance Criterion in the IMMUTABLE SECTION:
| AC | Status | Evidence (if MET) | Blocker (if NOT MET) | Justification (if DEFERRED) |
|----|--------|-------------------|---------------------|----------------------------|
| AC-1 | MET / PARTIAL / NOT MET / DEFERRED | ... | ... | ... |
| ... | ... | ... | ... | ... |

### 1.2 Forgotten Items Detection
Compare the original plan (@development/loop13/plan.md) with the current goal-tracker:
- Are there tasks that are neither in "Active", "Completed", nor "Deferred"?
- Are there tasks marked "complete" in summaries but not verified?
- List any forgotten items found.

### 1.3 Deferred Items Audit
For each item in "Explicitly Deferred":
- Is the deferral justification still valid?
- Should it be un-deferred based on current progress?
- Does it contradict the Ultimate Goal?

### 1.4 Goal Completion Summary
```
Acceptance Criteria: X/Y met (Z deferred)
Active Tasks: N remaining
Estimated remaining rounds: ?
Critical blockers: [list if any]
```

## Part 2: Mainline Drift Audit (MANDATORY)

Determine whether the recent rounds are still serving the original plan:
- Is the current round's mainline objective clear and singular?
- Has Claude been advancing mainline ACs, or mostly clearing side issues?
- Which findings are true **blocking side issues** versus merely **queued side issues**?

Include a short drift summary:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
Blocking Side Issues: N
Queued Side Issues: N
```

The `Mainline Progress Verdict` line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

## Part 3: Implementation Review

- Conduct a deep critical review of the implementation
- Verify Claude's claims match reality
- Identify any gaps, bugs, or incomplete work
- Reference @docs for design documents

## Part 4: ## Goal Tracker Update Requests (YOUR RESPONSIBILITY)

Claude should normally keep the **mutable section** of `goal-tracker.md` up to date directly. If Claude's summary contains a "Goal Tracker Update Request" section, or if you detect tracker drift during review, YOU must:

1. **Evaluate the tracker state**: Is the mutable section still aligned with the Ultimate Goal and current AC progress?
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/goal-tracker.md yourself with the requested changes:
   - Move tasks between Active/Completed/Deferred sections as appropriate
   - Add entries to "Plan Evolution Log" with round number and justification
   - Add new issues to "Blocking Side Issues" or "Queued Side Issues" as appropriate
   - **NEVER modify the IMMUTABLE SECTION** (Ultimate Goal and Acceptance Criteria)
3. **If you reject a requested tracker change**: Include in your review why it was rejected

Common update requests you should handle:
- Task completion: Move from "Active Tasks" to "Completed and Verified"
- New blocking issues: Add to "Blocking Side Issues"
- New queued issues: Add to "Queued Side Issues"
- Plan changes: Add to "Plan Evolution Log" with your assessment
- Deferrals: Only allow with strong justification; add to "Explicitly Deferred"

## Part 5: Progress Stagnation Check (MANDATORY for Full Alignment Rounds)

To implement the original plan at @development/loop13/plan.md, we have completed **20 iterations** (Round 0 to Round 19).

The project's `.humanize/rlcr/2026-06-20_09-57-51/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-06-20_09-57-51/round-18-review-result.md` (previous round)
- `@.humanize/rlcr/2026-06-20_09-57-51/round-17-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-06-20_09-57-51/round-18-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-19-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-19-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
