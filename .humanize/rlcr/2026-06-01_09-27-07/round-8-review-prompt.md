# Code Review - Round 8

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-8-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 8 Summary — Loop 7

## Mainline objective (round-8-contract.md)
**Close AC-1** (PARTIAL → MET): produce the two durable evidence artifacts the
plan requires (Codex R7 gap #1) — oracle-off byte-identical + zero-alloc under
CUDA-graph replay, and the dense/default oracle-stride reference.

## Outcome: ACHIEVED — AC-1 is MET.

## Work completed
1. **`oracle_off_graph_replay_alloc.json`** (+ `oracle_off_replay_alloc.py`):
   with `recall_oracle=false` (default), the production graph-safe DS selector is
   captured under CUDA graph and replayed 120 steps —
   `replay_indices/lengths_byte_identical_to_eager: true` (eager & replay share
   `selected_indices` sha `87426fc4`), `replay_allocation_delta_bytes: 0`,
   `replay_zero_new_allocations: true`, **verdict PASS**. The "zero hot-path cost"
   claim is **demonstrated, not asserted**.
   - **CI backing**: new GPU test
     `test_double_sparsity_unit.py::test_oracle_off_replay_byte_identical_and_zero_alloc`
     (asserts `sel.config.recall_oracle is False` + byte-identical replay + zero
     alloc), beside the pre-existing `test_cuda_graph_100_step_replay_matches_eager`
     and `test_cuda_graph_replay_zero_allocations`.
2. **`oracle_stride_reference.json`** (+ `oracle_stride_reference.py`): the
   oracle's emitted `stride` field is **1 for all 14,640 R4 success records**
   (hook hardcodes `stride=1` — dense sampling of every needle token, no
   subsample) ⇒ **`default_equals_stride1: true`**, proven from records; plus the
   **dense-DS within-budget** reference (1024w ≤2048 tok ⇒ DS selects densely ⇒
   default & hybrid both 100%) next to the default-stride beyond-budget served
   recall (4K 80%, 16K default 6% / hybrid 38%) + per-length score-only recall@K.
3. **Bundled queued cleanups** (Codex queued #1/#2): `niah_recall_matrix.py`
   module docstring made directional ("exceeds the baseline CI high"); the three
   MMLU artifacts enriched with `op_point` / `graph_mode` / `example_seed` /
   `data_dir` metadata (+ the runner now emits it).

## AC-1 verdict: MET
Oracle records the required per-trial fields on the live all-reduced score tensor,
fail-closed, dedicated sink (R1–R4); **oracle-off byte-identical + zero-alloc
under graph replay — demonstrated (R8)**; separated served-vs-admission baseline
at mem 0.7 with the **stride=1 dense reference (R8)**; AC-1.1 post-topK replacement
(R1/task5). All sub-criteria have committed evidence.

## Validation
- `oracle_off_replay_alloc.py` → PASS (byte-identical + 0 alloc bytes), 8×H200.
- **326 DS unit tests pass** including the new oracle-off replay test + the
  existing 100-step replay + zero-alloc tests.

## Files changed
`oracle_off_replay_alloc.py` (new), `oracle_off_graph_replay_alloc.json` (new),
`oracle_stride_reference.py` (new), `oracle_stride_reference.json` (new),
`m5_ac1_closure_finding.md` (new), `test_double_sparsity_unit.py` (new GPU test),
`niah_recall_matrix.py` (docstring), `mmlu_5shot.py` + `mmlu_{dsa,default,hybrid}_graph.json`
(metadata). Commit `f05cb730e` (pushed). No production runtime code changed.

## Remaining items (queued, justified)
- **AC-3 anchor_mode graph-safe port** + **AC-6 graph-vs-eager perf delta** (task #16).
- **AC-4 lifted-budget** (task13–17): the oracle gate justifies bounded Tier-2.A.
- **AC-6 consolidation + final strategic-gate supersession decision record** (task20).

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: this round assembled durable AC-1 artifacts from existing mechanisms
  (the graph-replay alloc detector + the oracle stride field) — no new reusable
  engineering pitfall surfaced.

## Goal Tracker Update Request
- **task4** (AC-1): oracle-off zero-hot-path DEMONSTRATED (R8) → done.
- **task6** (AC-1,AC-2): dense/stride reference DONE (R8); DSA/MMLU already done →
  done.
- **AC-1 → MET.**
- **Keep Active**: task #16 (anchor port + AC-6 perf + decision record), AC-4
  (task13–17), task20.
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
f05cb730e [Sparsity] Loop-7 R8: close AC-1 (oracle-off zero-hot-path + stride reference)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-7-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-7-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-6-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-6-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-5-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-5-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-8-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
