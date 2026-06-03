# Code Review - Round 2

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-2-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 2 Summary — Loop 7

## Mainline objective (round-2-contract.md)
Make the AC-3 non-learned scorer variants **correct and production-safe** (fix the Round-1 review's correctness holes; no measurement claims).

## Outcome: ACHIEVED

All four correctness items + the blocking issue are fixed and tested.

## Work Completed
1. **Production-path graph-capture safety (Blocking)** — Round 1's guard only covered `capture_decode_step`, not the real CUDA-graph runner (where a non-default scorer would run the eager selector *inside* capture). `validate_double_sparsity` now **fails fast at server init** when `ds_scorer_is_default(config) == False` and CUDA graph is enabled. **Verified on a real boot**: with `scorer_norm=cosine` + graphs enabled it errors *before* "Load weight begin" with `"...not yet supported under CUDA graph capture... Re-run with --disable-cuda-graph"`. (The `capture_decode_step` guard is kept as defense-in-depth.)
2. **Physical-path hybrid mis-application** — `compute_token_scores` now **rejects** `scorer_norm="hybrid"` (it has no per-request `seq_len`) instead of silently degrading to cosine (the exact moderate-context regression hybrid avoids).
3. **Anchor ablation completed** — `anchor_mode {off, recency, global, strided}` config field with a single deterministic generator `_anchor_positions` (recency = most-recent, global = earliest, strided = evenly spaced over `[0, seq_len)`), budget-clamp / short-seq / dedup / ascending-order handling; replaces the recency-only impl.
4. **TP cross-rank determinism** — a real 2-rank **gloo multiprocess** test (`test_ds_scorer_tp_determinism.py`) parameterized over `scorer_norm × head_agg × anchor_mode`: each rank holds a head-shard, computes per-rank scores, all-reduces, runs the shared top-K + anchor, and asserts identical per-rank `selected_indices`/`valid_lengths`. The `DoubleSparsityTPMisconfigured`/`DoubleSparsityRebindError` fail-fast guards are kept in the matrix.
5. **Launcher knobs** — `serve_double_sparsity.sh` exposes `HEAD_AGG`, `ANCHOR_MODE`, `ANCHOR_BUDGET`, `SCORER_NORM_HYBRID_THRESHOLD` in `DS_CONFIG` (needed for the AC-3 matrix).

## Files Changed
`config.py` (anchor_mode), `selection_kernel.py` (physical-hybrid reject, anchor modes, `ds_scorer_is_default`), `selector.py`, `validator.py` (startup guard), `cuda_graph.py` (message), `serve_double_sparsity.sh` (knobs), updated `test_scorer_variants.py`, new `test_ds_scorer_tp_determinism.py`. Commit `72c704edf`.

## Validation
- **301 DS unit tests + 6 gloo TP determinism tests pass** (anchor generators all modes, recency/strided/off/dedup force-include, physical-hybrid reject, default-guard, TP cross-rank equality for 5 flag combos, startup-guard reject + eager-pass).
- **Real boot** confirms the startup guard fires before model load.

## Remaining Items (queued, justified)
- **Graph-safe Triton hybrid/cosine/head/anchor port + full AC-3 measurement matrix** (task #13): the definitive production fix (so the variants run under CUDA graph) + binding non-regression matrix (MMLU re-anchor, dense-DS/within-budget parity, N≥50 16K, DSA same-node, eager-vs-graph perf, per-variant attribution, threshold sweep). Heavy kernel + GPU work; next round. **AC-3 closure depends on this.**
- **Oracle fail-closed + 64K re-run** (task #12, AC-1): M0 diagnostic hardening; next round.
- **Tier-2.A / AC-4** (task13–17), **M4 consolidation / AC-6 perf** (task19–20): separate milestones.
- **Plan-marker code/comment cleanup**: pre-merge cleanup, queued.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: no new reusable engineering pitfall surfaced. Reinforced the prior config-borne-flag lesson (the production guard reads `selector.config`, not env). The "guard the real runner path, not just the helper" point is captured in the round summary + the validator comment.

## Goal Tracker Update Request

### Requested Changes
- **Resolve Blocking Side Issue "graph-capture fix does not cover the production CUDA-graph runner"**: fixed — startup guard in `validate_double_sparsity` rejects a non-default scorer + CUDA graph before model load (real-boot verified); `capture_decode_step` guard kept as defense-in-depth.
- **task10 (anchor-budget)** → move to **implemented + unit-tested**: full `anchor_mode {off,recency,global,strided}` ablation with deterministic generators + tests (recency/global/strided, budget>top_k, short-seq, dedup, ordering). Per-variant *measurement* deferred to the AC-3 matrix (task #13).
- **task11 (TP determinism)** → move to **implemented + tested**: parameterized gloo multiprocess test over scorer_norm × head_agg × anchor_mode with cross-rank equality + fail-fast guard coverage.
- **task8 (AC-3 scorer)**: physical-path hybrid mis-application fixed (now rejected); remaining for AC-3 closure is the graph-safe Triton port + matrix (task #13).
- **Keep Active**: task #13 (graph-safe port + matrix) and task #12 (oracle fail-closed) as the next round's mainline; AC-4 (task13–17 plan) and M4 (task19–20) queued.

### Justification
Round 2 closed the correctness/safety gaps the Round-1 review found (production graph guard, physical-hybrid, full anchor ablation, TP determinism) so the AC-3 selector variants are trustworthy and cannot be silently misused under production graph capture. The remaining AC-3 work (graph-safe port + binding matrix) and the oracle fail-closed (AC-1) are the next round's mainline, converging the loop toward binding closure.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-2-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
