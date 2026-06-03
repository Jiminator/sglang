# Code Review - Round 7

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-7-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 7 Summary — Loop 7

## Mainline objective (round-7-contract.md)
Produce the **binding AC-3 graph-mode non-regression matrix** for the landed
Tier-2.B hybrid scorer: N≥50 16K served recall (DS-default vs DS-hybrid vs DSA,
all under CUDA graph, same session) + a durable dense-DS within-budget parity
artifact + an MMLU ≤1.0pp re-anchor (DSA vs DS-hybrid).

## Outcome: ACHIEVED — AC-3 non-regression SATISFIED for the hybrid scorer.

## Work completed
1. **Binding graph-mode recall matrix, N=50, 95% Clopper–Pearson CI**
   (`ds_vs_dsa_recall_matrix_graph_n50.json`, `niah_{dsa,default,hybrid}_graph_n50.json`):
   - 1024w **dense-DS / within-budget** (≤2048 tok): DSA/default/hybrid all **100%**.
   - 4K: hybrid **80% == default 80%** (≤8192 ⇒ raw regime; no regression); DSA 100%.
   - 16K: default **6% [1.3,16.5] → hybrid 38% [24.7,52.8] = +32 pp, MATERIAL**
     (the R6 N=20 graph read 25% — a low draw; N=50 binds it at 38%, ≈ eager 40%).
2. **MMLU 5-shot re-anchor, N=200, same questions (deterministic seed), graph-mode**
   (`mmlu_{dsa,default,hybrid}_graph.json`): DSA **89.0%** / default 88.5% /
   hybrid **88.5%** → hybrid **−0.5 pp vs DSA (≤1.0 pp gate PASSED)**; 0 pp vs
   default (MMLU is within-budget ⇒ hybrid uses its raw regime = default).
3. **Fast MMLU runner** (`mmlu_5shot.py`): 5-shot "Answer:" prompt +
   `max_new_tokens=4` (single-letter extraction) — avoids the reasoning model's
   2048-token chains that made `run_eval` (default `max_tokens=2048`) glacial
   (~minutes/question); ~0.25 s/question.
4. **R5 evidence-label cleanup (Codex queued #2)**: DSA JSONs relabeled
   `DSA native-NSA (no double-sparsity)`; `niah_ds_baseline.py` gained `--op-point`;
   `niah_recall_matrix.py` materiality_rule reworded directional ("variant point
   exceeds the DS-default baseline CI high"); matrices regenerated.

## Validation
- All recall + MMLU measured **under CUDA graph** (the production path, per
  `BL-20260602-eager-vs-graph-recall-differs-despite-identical-scorer`), 8×H200
  TP=8, int8/mem0.7, same session. DS engagement implied by the hybrid-vs-default
  16K recall gap (38% vs 6%).
- No production code changed; ran `test_scorer_variants.py` (20 pass) as a sanity.

## AC-3 verdict
**Non-regression SATISFIED** for the hybrid scorer: material long-context (16K)
uplift (6%→38%) + MMLU within 0.5pp of re-anchored DSA + dense-DS/within-budget
parity + no 4K regression + (R3/R6) TP=8 determinism & bit-identical eager-vs-graph
selection. The long-context gap to DSA's 100% is reduced (16K 38%), not closed —
a recorded, characterized result (64K remains scorer-limited per the oracle).

## Files changed
`m4_ac3_nonregression_finding.md` (new), `mmlu_5shot.py` (new),
`ds_vs_dsa_recall_matrix_graph_n50.json` + `niah_*_graph_n50.json` +
`mmlu_*_graph.json` (new data), `niah_ds_baseline.py` (--op-point),
`niah_recall_matrix.py` (directional wording), `niah_dsa_reference.json` +
`ds_vs_dsa_recall_matrix.json` (relabeled/regenerated). Commit `9a37590ec` (pushed).

## Remaining items (queued, justified) — task #16 + others
- **AC-6 graph-vs-eager scorer perf delta** (conc-1/16 TTFT, decode-TPS/req, mem).
- **anchor_mode graph-safe port** (still eager-only).
- **AC-4 lifted-budget** (task13–17): the oracle gate justifies bounded Tier-2.A.
- **AC-1 task4 alloc-detector under graph replay + dense/default oracle-stride
  artifact** (Codex gap #2): contained AC-1 closure.
- **AC-6 consolidation + final strategic-gate supersession decision record** (task20).

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-mmlu-quality-gate-on-reasoning-model
- Notes: a reasoning model (DSv3.2) under the standard `run_eval` MMLU sampler
  (`max_tokens=2048`) generates long chains-of-thought per question → the quality
  gate runs at minutes/question and never finishes. Use a 5-shot "Answer:" prompt
  + `max_new_tokens=4` + leading-letter parse (the AC-12 method) for a fast,
  paired DS-vs-DSA accuracy gate; deterministic example seed gives identical
  questions across servers for an exact paired delta.

## Goal Tracker Update Request
- **task12** (AC-2,AC-3): binding graph-mode recall+MMLU+dense matrix DONE (R7) —
  AC-3 non-regression satisfied. Remaining = perf + anchor port (task #16).
- **Resolve queued side issue** "R5 evidence labels" (fixed R7).
- **Keep Active**: task #16 (AC-6 perf + anchor port + final decision record),
  AC-4 (task13–17), AC-1 task4.
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
cb02b6673 [Sparsity] Loop-7 R6: port Tier-2.B scorer to the graph-safe Triton path (AC-3)
9a37590ec [Sparsity] Loop-7 R7: binding AC-3 non-regression matrix (graph-mode N=50 + MMLU)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-6-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-6-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-5-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-5-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-4-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-4-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-7-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
