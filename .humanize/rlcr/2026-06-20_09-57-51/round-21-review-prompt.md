# Code Review - Round 21

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-21-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 21 Summary

Mainline: **AC-4 evidence-table close-out** — fill the missing strict serial GSM8K cells and replace the
static selected-vs-total literals with an artifact-backed, fail-closed probe. Diagnostic/evidence only; no
selection/adapter fix.

## Work Completed
1. **Serial GSM8K cells** (THREADS=1, one TP=8 server at a time, no PYTHONPATH, completion API, teardown to
   0 MiB between arms) — filled for every AC-4 core arm from real `.out`:

   | arm | dense_b | sparse_b | dense_serial | sparse_serial |
   |---|---|---|---|---|
   | dsa | 0.975 | 0.973 | 0.965 | 0.947 |
   | dsa_noradix | 0.960 | 0.940 | **0.965** | **0.973** |
   | production_ds | 0.620 | 0.000 | 0.655 | **0.013** |
   | ref_faithful | 0.950 | 0.013 | **0.965** | **0.013** |
   | ref_cosine | 0.940 | 0.940 | **0.965** | **0.947** |

   **Serial ≈ batched** everywhere → the regression is NOT batch-dependent. The serial cells corroborate the
   verdict from a second mode: production_ds **dense** serial 0.655 stays collapsed while the
   current-slot-INCLUDED reference arms get dense ~0.965 (the dense gap tracks current-slot exclusion, H3),
   and **sparse** stays scorer-driven (rawdot 0.013 vs cosine 0.947).
2. **`ac4_selected_vs_total_probe.py`** (new, fail-closed) — probes the live server's
   `meta_info["double_sparsity"]` per arm per regime (a dense <top_k + a sparse >top_k `/generate`), asserts
   dense `selected==total`, sparse `selected<total`, `dense_fallback==0`, and atomically updates
   `evidence/ac4_selected_vs_total.json`. Result: production_ds / ref_faithful / ref_cosine each dense
   334/334, sparse 2048/3692, dense_fallback 0 — DS genuinely active (keeps all in dense, prunes in sparse).
3. **`build_ledger.py`** — wired the new serial `.out` labels; **replaced the static `ds={...}` literals**
   for the core DS arms with values loaded from the artifact via `validate_selected_vs_total_artifact()`
   (fail-closed on the DS-active invariants); added a guard that **rejects a BLANK serial cell** for any AC-4
   core arm with a wired label; records `selected_vs_total` provenance in `run_meta`. `findings.md` records
   the result.

## Verification (the guards fire)
All 6 negatives make `build_ledger.py` ABORT, then restore → provenance consistent:
- selected-vs-total: sparse `selected==total` (no pruning), `dense_fallback!=0`, missing core arm, dense
  `selected!=total`, and a missing artifact.
- a hidden serial `.out` (blank cell) for a core arm.

## Files Changed (committed `cc9865440`)
- `development/loop13/ac4_selected_vs_total_probe.py` (new), `development/loop13/build_ledger.py` (serial
  labels + artifact-backed selected-vs-total + validate gate + blank-serial guard + run_meta provenance),
  `development/loop13/evidence/ac4_selected_vs_total.json` (new), `development/loop13/evidence/evidence_table.md`
  + `evidence/meta/*` (regenerated — serial cells filled, selected-vs-total from the artifact),
  `development/loop13/evidence/findings.md`. (Serial `.out` run logs are gitignored, per the established
  convention — the derived scores are committed in the per-arm JSONs/table.)

## Validation
- CPU suite, explicit args: `ac3_1_materialized_k_equality`, `ac4_garbage_counters`,
  `ac2_1_forced_all_assertions`, `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `verify_ac2_3 .sglang_ds_scorecap_sparse`
  (committed AC-2.3 artifact unchanged), `test_reference_selectors` (5/5) — **all exit 0**.
- `build_ledger.py` → provenance consistent. One TP=8 server at a time (4 serial boots + 3 probe boots),
  each torn down to 0 MiB; no PYTHONPATH; no `.out`/`.humanize` raw artifacts committed. No selection/adapter
  **fix**.

## Remaining Items (for COMPLETE)
- **AC-8** final root-cause writeup — the LAST item. Regenerate `ROOT_CAUSE.md` from the final evidence
  package (per-arm serial+batched table, AC-2.1/AC-2.4/AC-3.1/AC-4/AC-6 artifacts, the H0/H1/H2/H3 verdict,
  the "diagnosis loop, no fix" scope, the recommendation), and add a self-check that refuses AC-8 while the
  AC-4 core serial cells or the selected-vs-total artifact are absent.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: Mechanical evidence-table close-out — serial GSM8K runs via the existing guarded harness +
  an artifact-backed `meta_info["double_sparsity"]` probe + the now-standard fail-closed ledger gates
  (validate-before-render, reject-blank-cell). No new reusable technique beyond the already-captured
  fail-closed-artifact + ledger-tracks-serve-mode lessons; the one operational note (the sparse probe prompt
  must exceed top_k or selected==total trivially — caught + fixed by the fail-closed dense<total invariant)
  is a parameter detail, not a lesson.

## Goal Tracker Update Request

### Requested Changes (already applied to the mutable section):
- Plan Version → 25 (Round 21); added the Round-21 evolution row.
- task1 (AC-1 serial) → done; task9 (AC-4) → AC-4 CLOSED for the core arms (batched + serial + garbage +
  artifact-backed selected-vs-total, all guarded).
- Marked the broad "evidence package lacks selected-vs-total / serial cells" blocker RESOLVED.

### Justification:
Codex's R20-review named the blank serial cells + static selected-vs-total as the AC-4 gap. Both are now
filled from real runs and provenance-backed with fail-closed gates (validate-before-render + reject-blank-
cell, all 6 negatives verified to abort). The serial cells also corroborate the verdict from a second mode
(serial≈batched; dense gap = current-slot, sparse = scorer). The ONLY remaining loop item is the AC-8 final
root-cause writeup.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
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
e67f1b5f3 [loop13][R20] AC-3.1 CAPTURED-row materialized fp32 K_label selected-index equality
cc9865440 [loop13][R21] AC-4 evidence-table close-out: strict serial cells + artifact-backed selected-vs-total
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-20-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-20-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-19-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-19-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-18-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-18-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-21-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
