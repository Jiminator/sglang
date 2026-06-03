# Code Review - Round 3

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-3-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 3 Summary — Loop 7

## Mainline objective (round-3-contract.md)
Close the AC-3 non-learned selector-variant correctness so task10/task11 can be marked complete: fix the anchor over-budget bug and replace the TP test with a real TP=8, full-matrix, production-logical-path test.

## Outcome: ACHIEVED

## Work Completed
1. **Anchor over-budget bug fixed (Blocking).** `_force_include_anchor` clamped the anchor budget only to `seq_len`, not to the selected count — so with `top_k=3, seq_len=8` a recency `budget=5` forced `[3,4,5]` instead of the most-recent `[5,6,7]`. Now it computes `effective_budget = min(anchor_budget, len(selected))` BEFORE generating anchor positions, so recency/global/strided produce the correct effective set. Verified: `recency b5 → [5,6,7]`, `strided b5 → [0,4,7]`, `recency b10 → [5,6,7]`, `global b5 → [0,1,2]`. Regression tests added for recency/strided `budget > top_k` and `budget ≥ seq_len`.
2. **TP determinism upgraded to the required TP=8 full-matrix, production-path test.** Replaced the partial TP=2 / 5-combo / direct-`compute_token_scores` test with a real **8-rank gloo** test that drives the **production logical path** (`retrieve_topk_via_labels` with `req_pool_indices`/`req_to_token`/`seq_lens` + config threading: head-sharded per-rank scoring → SUM all-reduce → deterministic top-K → anchor force-include). It covers the **full 24-combo matrix** `scorer_norm{off,cosine,hybrid} × head_agg{max,mean} × anchor_mode{off,recency,global,strided}` and asserts identical per-rank `selected_indices`/`valid_lengths`. Fail-fast (`DoubleSparsityTPMisconfigured`/`DoubleSparsityRebindError`) preserved.

## Files Changed
`selection_kernel.py` (anchor effective-budget clamp), `test_scorer_variants.py` (over-budget regressions), `test_ds_scorer_tp_determinism.py` (rewritten TP=8 logical-path full-matrix). Commit `fc8871372`.

## Validation
- **323 DS unit tests pass** + the **TP=8 24-combo gloo matrix** test (21 s, all ranks identical for every combo).
- Anchor reproduction verified against the Round-2 review counterexamples.

## Remaining Items (queued, justified)
- **Graph-safe Triton scorer/head/anchor port + full AC-3 measurement matrix** (task #13): heavy kernel + GPU work (MMLU, dense-DS, N≥50, DSA same-node, eager-vs-graph perf, per-variant attribution). The variants are *correct* and *safe* (Round-2 startup guard); the production port + binding matrix is the next round. AC-3 *measurement* closure depends on it.
- **Oracle fail-closed + 64K re-run** (task #12, AC-1/AC-2): the most-deferred item; gets its own focused round next (config-borne activation so it records on TP workers + fail-closed semantics + re-run).
- **Tier-2.A / AC-4** (task13–17), **M4 consolidation / AC-6 perf + final decision record** (task19–20): sequenced after the selector + oracle are binding.
- **Plan-marker code/comment cleanup**: pre-merge; queued.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: no new reusable engineering pitfall (the anchor bug and TP-test-coverage gap are round-specific defects, captured in the commit + this summary).

## Goal Tracker Update Request

### Requested Changes
- **Resolve Blocking Side Issue "anchor force-include mishandles `anchor_budget > top_k`"**: fixed (`effective_budget = min(anchor_budget, len(selected))` before anchor generation; reproduction verified; regression tests added).
- **task10 (anchor-budget ablation)** → **implemented + unit-tested**: full `anchor_mode {off,recency,global,strided}` with the over-budget correctness fix; per-variant *measurement* still part of the AC-3 matrix (task #13).
- **task11 (TP determinism)** → **implemented + tested**: TP=8 gloo test through the production logical path over the full `scorer_norm × head_agg × anchor_mode` matrix (24 combos), cross-rank-identical. (CPU gloo TP=8 equality; an 8×H200 hardware TP=8 artifact can be added during the task #13 measurement run.)
- **Keep Active**: task #13 (graph-safe Triton port + full AC-3 measurement matrix) and task #12 (oracle fail-closed + 64K) as the next rounds' mainline; AC-4 (task13–17) and M4 (task19–20) sequenced after.

### Justification
Round 3 closed the two AC-3 selector-variant correctness gaps the Round-2 review rejected (the anchor over-budget bug and the partial TP test), so the variants are now correct and have TP=8 full-matrix cross-rank determinism evidence. The remaining work is the heavy graph-safe Triton port + binding measurement matrix and the oracle fail-closed hardening, sequenced as the next rounds' mainline to converge the loop on binding closure.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
9ca1f5133 [Sparsity] Loop-7 recall R&D: plan + QA (active plan gitignored)
9914a3004 [Sparsity] Loop-7 M0: selection-recall oracle diagnostic math
8074cb1cf [Sparsity] Loop-7 M0: oracle sink + AC-1.1 force + flag-gated hook
c6ffcdea6 [Sparsity] Loop-7 M0: DS served-recall baseline at mem 0.7 (N=20)
78f6b5d17 [Sparsity] Loop-7 M0: oracle budget-vs-scorer evidence (A-vs-B decider)
a1e2c72dc [Sparsity] Loop-7 M0: A-vs-B decision (Codex-adjudicated)
599d7cc99 [Sparsity] Loop-7 M1: flag-gated cosine scorer (Tier-2.B candidate)
e2674f4f4 [Sparsity] Loop-7 M1: cosine scorer MEASURED — 16K recall 5%->40%
c5a829def [Sparsity] Loop-7: oracle trial-file read fresh; gitignore transient artifacts
273622705 [Sparsity] Loop-7 R1: length-conditional hybrid scorer (best of both regimes)
72c704edf [Sparsity] Loop-7 R2: scorer variants correct + production-safe
fc8871372 [Sparsity] Loop-7 R3: fix anchor over-budget + TP=8 logical-path matrix
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-2-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-2-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-1-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-1-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-0-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-0-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-3-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
