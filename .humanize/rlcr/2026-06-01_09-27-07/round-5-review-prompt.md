# Code Review - Round 5

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-5-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 5 Summary — Loop 7

## Mainline objective (round-5-contract.md)
Produce the **binding DS-vs-DSA same-node served-recall uplift matrix (AC-2)**:
measure the missing DSA same-node NIAH reference at the Loop-7 op-point, assemble
the DS-default vs DS-hybrid(Tier-2.B) vs DSA matrix at 4K/16K/64K (N=20) with
Clopper–Pearson CIs + an up-front materiality rule, and record within-budget
(≤2048) parity + a dense reference.

## Outcome: ACHIEVED — binding AC-2 recall matrix; Tier-2.B gives a material 16K uplift.

## Work completed (mainline)
1. **DSA same-node reference measured** (`niah_dsa_reference.json`,
   `serve_native_nsa.sh` at mem 0.7, N=20): 1024w/4K/16K/64K **all 100%**, 0
   admission failures — the recall ceiling, and the DSA-same-node artifact whose
   absence AC-2's negative tests reject.
2. **DS-hybrid (Tier-2.B) measured** fresh (`niah_ds_hybrid.json`, eager
   scorer_norm=hybrid, int8/mem0.7, N=20), incl. the previously-missing 64K. DS
   engagement verified live via the `double_sparsity` meta (16K: sparsity 0.88,
   selected 2048, dense_fallback 0 — sparse, not error-contained dense).
3. **Consolidated DS-vs-DSA matrix** (`niah_recall_matrix.py`,
   `ds_vs_dsa_recall_matrix.json`, `m2_recall_matrix_finding.md`) with per-cell
   recall + N + **Clopper–Pearson 95% CI** and the up-front materiality rule (a
   variant uplift is material only when its recall point exceeds the DS-default
   baseline CI). DS-default cited as the plan's AC-1 served baseline
   (`ds_niah_baseline_mem07.json`, N=20, same node/op-point).

### Matrix (N=20, served recall % [95% CP CI])
| len | DSA | DS-default | DS-hybrid | uplift | material |
|-----|-----|-----------|-----------|--------|----------|
| 1024w (≤budget) | 100 [83,100] | 100 [83,100] | 100 [83,100] | 0 | parity ✓ |
| 4K  | 100 [83,100] | 75 [51,91] | 85 [62,97] | +10pp | NO (within CI) |
| 16K | 100 [83,100] | 5 [0,25] | **40 [19,64]** | **+35pp** | **YES** (>CI hi 24.9) |
| 64K | 100 [83,100] | 5 [0,25] | 0 [0,17] | −5pp | NO (1-needle floor noise) |

**Finding:** Tier-2.B (hybrid scorer) delivers a **material 16K recall uplift**
(5%→40%, the long-context goal regime) with **within-budget parity** preserved
and **no material 4K change**; 64K stays scorer-limited (both ~0%, sampling
noise). A recorded, characterized, non-regressing AC-2 result, consistent with
the M0 oracle (16K budget-partial, 64K scorer-limited).

## Work completed (queued, bundled cheap)
4. **Fixed the R4-Review analyzer artifact** (Codex gap #1): `analyze_oracle.py`
   now emits separate `uplift_4096_minus_2048` / `uplift_8192_minus_2048` fields
   and a three-way verdict (budget-limited / **budget-partial** / scorer-limited);
   regenerated `oracle_budget_vs_scorer_r4.json` so 16K reads budget-partial,
   matching `m0_oracle_finding_r4.md`.

## Files changed
`analyze_oracle.py`, `oracle_budget_vs_scorer_r4.json` (regenerated),
`niah_recall_matrix.py` (new), `niah_dsa_reference.json` (new),
`niah_ds_hybrid.json` (new), `ds_vs_dsa_recall_matrix.json` (new),
`m2_recall_matrix_finding.md` (new). Commit `9f76ad659` (pushed). No production
code changed this round.

## Validation
- DSA/DS-hybrid NIAH measured on 8×H200 TP=8 at mem 0.7, N=20, served-vs-admission
  separated; CIs via scipy Clopper–Pearson.
- DS unit tests unaffected (no production change); ran the oracle + scorer suites
  (33 pass) as a sanity check.

## Remaining items (queued, justified)
- **Graph-safe Triton scorer port** (AC-3 "landed path"): the eager hybrid scorer
  is ~8× slower per request (e.g. 64K trial ~207 s vs DSA ~184 s for 20). Porting
  scorer_norm/head_agg into the graph-safe Triton kernel with bit-exact
  eager-vs-graph equality is the production-viability keystone — research-grade,
  next round's candidate mainline (or a documented disposition per the plan).
- **MMLU ≤1.0pp re-anchor + N≥50 binding 16K** (AC-3): long evals; next.
- **AC-4 lifted-budget** (task13–17), **AC-6 perf/consolidation + final
  decision record** (task19–20): sequenced after AC-3.
- Plan-marker / stale-comment cleanup: pre-merge.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-niah-recall-materiality-floor-noise
- Notes: at N=20, a 1-needle move at the recall floor (default 1/20=5% vs hybrid
  0/20=0%) trips a naive "point outside the baseline CI" materiality test as a
  "material regression" — a degenerate-CI artifact. The plan's rule is
  directional ("uplift exceeds the baseline CI" = upward); report below-CI
  separately and never claim a sub-2-needle floor move as material. Captured so
  the final AC-2 decision record doesn't over-claim a 64K regression.

## Goal Tracker Update Request
- **task6** (AC-1,AC-2): DSA same-node comparison artifact DONE (R5); within-budget
  parity recorded. Remaining: MMLU re-anchor + dense/default stride microbench.
- **task12** (AC-2,AC-3): binding DS-vs-DSA served-recall matrix + CIs + 64K-hybrid
  + within-budget parity DONE (R5). Remaining: N≥50 16K, MMLU, graph-safe port + perf.
- **Resolve queued side issue** "R4 analyzer artifact mislabel" (fixed R5).
- **Keep Active**: task13 (graph-safe Triton port + perf) as the next mainline
  candidate; MMLU/N≥50; AC-4 / M4 sequenced after.
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
bf2ce9b2b [Sparsity] Loop-7 R4: oracle fail-closed + config-borne + 64K binding re-run
9f76ad659 [Sparsity] Loop-7 R5: binding DS-vs-DSA same-node served-recall matrix (AC-2)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-4-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-4-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-3-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-3-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-2-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-2-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-5-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
