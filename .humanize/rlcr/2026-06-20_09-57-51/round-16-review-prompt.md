# Code Review - Round 16

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-16-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
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
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
9ab62e6ad [loop11b] R3 review fixes: DS abort test rename + comparator report verdict/labels + green test suite
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-15-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-15-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-14-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-14-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-13-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-13-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-16-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
