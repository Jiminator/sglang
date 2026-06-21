# Code Review - Round 0

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@.humanize/rlcr/2026-06-21_19-48-53/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-21_19-48-53/round-0-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Skip-impl code-review summary — loop13 Double-Sparsity diagnosis

This is a `--skip-impl` review of an already-complete body of work: the 22-round Loop-13 **diagnosis loop**
(commits `fc6ac20a7 … 762330437`, base `loop13-base` = `180f6dd6d`). The prior RLCR loop's implementation
phase already returned **COMPLETE** (8/8 ACs); this session re-runs the code-review phase against a valid
ancestor base (the original `main` base was a disjoint single-commit history, so `git merge-base` errored —
fixed by reviewing against `loop13-base`). No new code was written this round; the tree is clean.

## What Was Implemented
The diff under review (`loop13-base..HEAD`, 58 files, +13132/-10) is the Loop-13 diagnosis harness, evidence
package, and verdict — **diagnostic instrumentation only, no selection/adapter fix landed**.
- **`development/loop13/`** — guarded serve modes (`serve.sh`) + `run_gsm8k.sh`; the per-AC reducers
  (`ac2_1_*`, `ac2_2_*`, `verify_ac2_3.py`, `ac4_garbage_counters.py`, `niah_recall_oracle.py`,
  `ac3_1_materialized_k_equality.py`, `ac4_selected_vs_total_probe.py`, `ac6_*`, `ac8_selfcheck.py`);
  `build_ledger.py` (provenance-consistent ledger with per-artifact fail-closed gates); the deliverable
  `ROOT_CAUSE.md` + committed `evidence/`.
- **Production code** (`python/sglang/srt/layers/attention/double_sparsity/*` + `models/deepseek_v2.py`) —
  config-borne, **default-off** diagnostic additions only: the reference selectors (`selector_impl ∈
  {reference_rawdot, reference_cosine}`, `reference_include_current`) and capture flags (`forced_all_assert`,
  `recall_oracle`, `materialized_k_capture`, `score_reduce_dtype`). Every capture hook is inside the existing
  `not torch.cuda.is_current_stream_capturing()` guard, host-side copy only, mutates nothing in the selected
  set, eager-only → the production decode path is **byte-identical when the flags are off** (the 5
  reference-selector unit tests pass; flags default `False` in all 4 config places).

## Files Changed
58 files vs `loop13-base`; net new diagnostic code + evidence under `development/loop13/`, plus the guarded
default-off DS instrumentation in `python/sglang/srt/layers/attention/double_sparsity/` and
`python/sglang/srt/models/deepseek_v2.py`. No production behavior change when the diagnostic flags are off.

## Validation
- `python3 development/loop13/ac8_selfcheck.py` → "AC-8 PACKAGE COMPLETE".
- `python3 development/loop13/build_ledger.py` → provenance consistent.
- Full CPU reducer suite + the 5 reference-selector unit tests pass; AC-2.3 artifact unchanged.
- Raw capture dirs (`.sglang_ds_*`) and GSM8K `.out` logs are gitignored (derived scores committed in the
  per-arm JSONs/table).

## Verdict delivered (`development/loop13/ROOT_CAUSE.md`)
Two regressions, not an algorithm/mask failure: dense 0.620 = **H3** (current decode slot excluded from its
own selection — `_slot_written` not restored), measured on the `_ds_slot_written` bitmap (61776/61776);
sparse 0.000 = the raw-dot `scorer_norm="off"` lock (Loop-11 dropped the cosine scorer), interacting with H3
(AC-6 2×2 — sparse recovery to ≈0.94 needs both). GOOD gate; not H0/H2; AC-7 moot. Recommendation only.

## Remaining Items
None blocking. Queued non-blocking cleanup (out of scope for the diagnosis loop): plan-workflow terms
(`AC-*`/`H3`) in retained diagnostic comments; reference-selector CUDA-graph safety if these modes are ever
exposed outside `development/loop13`; the `ac4_garbage_counters.py --arm <non-prod>` default-CAPDIR ergonomics
(the ledger already rejects a wrong-source artifact).

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Code-review pass over already-committed diagnosis work; no new implementation, so no new lesson. The
one operational note (an RLCR code-review phase needs a base that is a real ancestor of HEAD — a disjoint base
branch makes `codex review`'s `git merge-base` error) is loop tooling, not a project lesson.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
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
e67f1b5f3 [loop13][R20] AC-3.1 CAPTURED-row materialized fp32 K_label selected-index equality
cc9865440 [loop13][R21] AC-4 evidence-table close-out: strict serial cells + artifact-backed selected-vs-total
762330437 [loop13][R22] AC-8 final root-cause writeup + fail-closed self-check (diagnosis complete)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
(first round, no prior history)

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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-21_19-48-53/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-21_19-48-53/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-21_19-48-53/round-0-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
