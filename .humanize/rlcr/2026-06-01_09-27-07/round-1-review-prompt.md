# Code Review - Round 1

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-1-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 1 Summary — Loop 7

## Mainline objective (round-1-contract.md)
Land the production-shaped AC-3 Tier-2.B selector — primarily a **length-conditional hybrid scorer** — and measure it to show it recovers 4K while keeping the 16K gain; fix the two scorer-related blocking issues.

## Outcome: ACHIEVED (measured)

The length-conditional **hybrid scorer is the best of both regimes**:

| length | raw (prod) | uniform cosine | **hybrid** | path |
|--------|-----------|----------------|-----------|------|
| 4K | 75% [.51,.91] | 25% [.09,.49] | **85% [.62,.97]** | raw (≤8K) |
| 16K | 5% [.00,.25] | 40% [.19,.64] | **40% [.19,.64]** | cosine (>8K) |

The hybrid **recovers 4K (85%, cosine's regression gone)** AND **keeps 16K (40%)** — it is the per-length max. Measured 8×H200, N=20, via the new `LOOP7_MEASUREMENT` op-point mode. `development/loop7/m1_hybrid_finding.md`.

## Work Completed
- **`config.py`**: `scorer_norm` extended to `{off, cosine, hybrid}` + `scorer_norm_hybrid_threshold` (8192); independent `head_agg` (`max|mean`) and `anchor_budget` (int) config fields; all validated. Default = byte-identical.
- **`selection_kernel.py`**: hybrid (per-request `seq_len` raw/cosine) + head-aggregation in both score paths; `_force_include_recency_anchor` (anchor force-include); `ds_scorer_is_default` guard; threaded through `retrieve_topk_via_labels`.
- **`selector.py`**: passes all config variant values to the scorer.
- **`models/deepseek_v2.py`**: routes decode to the eager logical scorer for ANY non-default scorer.
- **`cuda_graph.py` (Blocking B1)**: `capture_decode_step` FAILS FAST for a non-default scorer instead of silently raw-scoring under capture.
- **`serve_double_sparsity.sh` (Blocking B2)**: `LOOP7_MEASUREMENT=1` pins int8/mem 0.7 and logs the effective `double_sparsity_config`.

## Files Changed
`config.py`, `selection_kernel.py`, `selector.py`, `deepseek_v2.py`, `cuda_graph.py`, `serve_double_sparsity.sh`, new `test_scorer_variants.py`, `development/loop7/m1_hybrid_finding.md` + `recall_hybrid.json`. Commit `273622705`.

## Validation
- **5 new variant unit tests** + **308 existing DS tests pass** (hybrid picks raw≤thr / cosine>thr, off==raw byte-identical, head_agg max≠mean, anchor force-include, default-guard).
- Live 8×H200 hybrid NIAH 4K/16K (N=20) — numbers above.

## Remaining Items (queued, justified)
- **Oracle fail-closed + 64K oracle re-run** (task #12, Codex gap #1): M0 diagnostic hardening; does not block the served-recall hybrid measurement. High priority next round.
- **Graph-safe Triton hybrid port + full AC-3 matrix** (task #13): MMLU re-anchor, dense-DS/within-budget parity, N≥50 binding 16K, DSA same-node reference, TP=8 cross-rank determinism per flag, hybrid-threshold sweep. **AC-3 closure depends on this — not claimed yet.**
- Tier-2.A / AC-4, M4 consolidation / AC-6 perf, plan-marker cleanup: separate milestones, queued.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: applied the prior round's lesson (BL-20260602-ds-flag-must-be-config-borne-not-env) — all new variants are DS-config fields. No new reusable engineering pitfall surfaced.

## Goal Tracker Update Request

### Requested Changes
- **task8 (AC-3 scorer)** → near-complete: the landable hybrid is implemented + measured (4K 85% / 16K 40%, best of both); remaining is the graph-safe port + matrix (task #13).
- **task9 (head-aggregation)** + **task10 (anchor-budget)** → mark **implemented + unit-tested** (config-gated, default byte-identical); per-variant measurement deferred to the AC-3 matrix (task #13).
- **Resolve Blocking Side Issue "graph-capture ignores scorer_norm"** → fixed (`cuda_graph.capture_decode_step` fail-fast + deepseek_v2 routing via `ds_scorer_is_default`).
- **Resolve Blocking Side Issue "serve op-point mismatch"** → fixed (`LOOP7_MEASUREMENT=1`).
- **Add Active task #13** (graph-safe Triton hybrid port + full AC-3 matrix) for AC-3 closure; **keep Active task #12** (oracle fail-closed + 64K) for AC-1/AC-2 binding.

### Justification
The round delivered the evidence-backed AC-3 landed candidate (hybrid, best-of-both measured) and fixed both scorer Blocking Side Issues, advancing AC-3/AC-6. The remaining AC-3 items (graph-safe port, full matrix) and the oracle fail-closed (AC-1) are scoped as the next round's mainline so the loop converges on binding closure rather than partial claims.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-1-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
