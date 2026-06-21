# Code Review - Round 17

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-17-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
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
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
3058bdc35 [loop12] add gen-plan output + draft; record pensieve doctor state
aaefdaf1e [loop12] R0 evidence: calibrate/boot/perf scripts + verdicts + BASE
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-16-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-16-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-15-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-15-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-14-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-14-review-result.md


Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Implementation Review

- Your task is to conduct a deep critical review, focusing on finding implementation issues and identifying gaps between "plan-design" and actual implementation.
- Relevant top-level guidance documents, phased implementation plans, and other important documentation and implementation references are located under @docs.
- If Claude planned to defer any tasks to future phases in its summary, DO NOT follow its lead. Instead, you should force Claude to complete ALL tasks as planned.
  - Such deferred tasks are considered incomplete work and should be flagged in your review comments, requiring Claude to address them.
  - If Claude planned to defer any tasks, please explore the codebase in-depth and draft a detailed implementation plan. This plan should be included in your review comments for Claude to follow.
  - Your review should be meticulous and skeptical. Look for any discrepancies, missing features, incomplete implementations.
- If Claude does not plan to defer any tasks, but honestly admits that some tasks are still pending (not yet completed), you should also include those pending tasks in your review.
  - Your review should elaborate on those unfinished tasks, explore the codebase, and draft an implementation plan.
  - A good engineering implementation plan should be **singular, directive, and definitive**, rather than discussing multiple possible implementation options.
  - The implementation plan should be **unambiguous**, internally consistent, and coherent from beginning to end, so that **Claude can execute the work accurately and without error**.

## Part 2: Goal Alignment Check (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/goal-tracker.md and verify:

1. **Acceptance Criteria Progress**: For each AC, is progress being made? Are any ACs being ignored?
2. **Forgotten Items**: Are there tasks from the original plan that are not tracked in Active/Completed/Deferred?
3. **Deferred Items**: Are deferrals justified? Do they block any ACs?
4. **Plan Evolution**: If Claude modified the plan, is the justification valid?

Include a brief Goal Alignment Summary in your review:
```
ACs: X/Y addressed | Forgotten items: N | Unjustified deferrals: N
```

## Part 3: Required Finding Classification

You MUST classify your findings into these lanes:
- **Mainline Gaps**: plan-derived work or AC progress that is missing, incomplete, or regressing
- **Blocking Side Issues**: bugs or implementation issues that block the current mainline objective from succeeding safely
- **Queued Side Issues**: valid non-blocking follow-up issues that should be documented but must NOT take over the next round

Also include a one-line verdict:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
```

This verdict line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

If Claude mostly worked on queued side issues and failed to advance the mainline, say so explicitly.

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

## Part 5: Output Requirements

- In short, your review comments can include: problems/findings/blockers; claims that don't match reality; implementation plans for deferred work (to be implemented now); implementation plans for unfinished work; goal alignment issues.
- Your output should be structured so Claude can tell which items are mainline gaps, blocking side issues, and queued side issues.
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-17-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
