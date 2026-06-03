Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop7/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop7/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-19-contract.md

Your round contract must contain:
- Exactly one **mainline objective**
- The 1-2 target ACs for this round
- Which issues are truly **blocking** that mainline objective
- Which issues are **queued** and explicitly out of scope
- Concrete success criteria for this round

Do not start implementation until the round contract exists.

## Task Lane Rules

Use the Task system (TaskCreate, TaskUpdate, TaskList) with one required tag per task:
- `[mainline]` for plan-derived work that directly advances this round's objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` work is the round's primary success condition
- `[blocking]` work is allowed only when it truly blocks the mainline objective
- `[queued]` work must be documented but must NOT replace the round objective
- If a new bug does not block the current objective, tag it `[queued]` and keep moving on mainline work

Before executing each task in this round:
1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 18 Review Result

Mainline Progress Verdict: ADVANCED

I did not find a high-signal AC-4 runtime or disposition gap in the R18 changes.
Round 18 closes the two R17 review blockers:

- `m9_tier2a_disposition.md` is now consistent with the production graph state:
  validator relaxed, launcher not eager-forced, fixed-shape `out=` dequant path
  recorded, graph-mode 95% is binding, and the prior deferred/eager-required
  claims are removed (`development/loop7/m9_tier2a_disposition.md:81`,
  `:95`, `:111`, `:146`).
- The literal raw 8-rank NCCL-under-`torch.cuda.graph` unit harness was attempted
  and found infeasible; the accepted proof is the documented composition:
  single-rank lifted-width selector capture + eager 8-rank all-reduce equality +
  the live TP=8 production graph run (`test_ds_scorer_tp_determinism.py:178`,
  `:210`, `development/loop7/m9_tier2a_disposition.md:146`).
- The lifted+speculative queued hazard is resolved fail-closed in the validator
  (`python/sglang/srt/layers/attention/double_sparsity/validator.py:120`).

Local verification run during review:

```bash
pytest -q test/registered/unit/layers/attention/test_scorer_variants.py::TestLiftedBudgetABI
# 13 passed

pytest -q test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py::TestLiftedWidthSelectionGraphCaptured
# 2 passed

pytest -q test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py::TestTP8LiftedWidthDeterminism
# 2 passed

git diff --check 41e0af078..f9f6ec056
# clean
```

## Mainline Gaps

1. **task19 / AC-6 remains unfinished.**

   This was explicitly out of R18 scope, but it is still original-plan mainline
   work and must be the next implementation round before Loop 7 can close.

   Required implementation plan:
   - Use the existing `development/serve_double_sparsity.sh` and
     `development/benchmark.sh` tooling at the Loop-7 op-point:
     DS int8, `mem_fraction_static=0.7`, fp8 KV, TP=8, page size 64, graph on,
     and explicitly recorded radix/cache assumptions.
   - Measure DS default, graph-safe DS hybrid, DSA/native-NSA, and lifted DS where
     applicable. Do not introduce new serve/bench scaffolding.
   - Record conc-1 and conc-16 TTFT, decode TPS/req, GPU memory, graph replay
     status, admission behavior, exact server args, exact DS config, commit SHA,
     GPU type, and artifact paths.
   - Produce the consolidated DS-vs-DSA recall/perf/non-regression report and
     make it the source artifact for task20.

2. **task20 / AC-2 remains unfinished.**

   The recall-uplift evidence exists, but the final strategic-gate supersession
   decision record is still not written.

   Required implementation plan:
   - After task19 is complete, write the final gate-supersession decision record.
   - Cite M0 regime attribution, AC-1 closure, AC-3 hybrid scorer evidence,
     AC-4 production-ready lifted evidence, AC-5 64K servability, and AC-6 perf
     guardrails.
   - Explicitly state what measured evidence superseded the Loop-6
     Tier-2.A-primary ordering, and cite or preserve the R8 stride/oracle
     provenance before relying on it.

## Blocking Side Issues

None.

## Queued Side Issues

1. Preserve or cite the R8 oracle-sink provenance before task20, or cite the
   hardcoded `stride=1` call site plus the committed aggregate explicitly.
2. Remove plan/workflow markers from production code/comments/tests before final
   cleanup/merge.
3. Learned/distilled selector work remains out of scope unless explicitly approved
   under DEC-5.

## Goal Alignment Summary

ACs: 6/6 addressed (4 met, 2 partial) | Forgotten items: 0 | Unjustified deferrals: 0

| AC | Status | Review result |
|----|--------|---------------|
| AC-1 | MET | Prior oracle/fail-closed/stride/zero-hot-path evidence remains accepted. |
| AC-2 | PARTIAL | Recall uplift evidence exists; task20 final supersession record remains active. |
| AC-3 | MET | Graph-safe non-learned scorer/head/anchor path and non-regression matrix remain accepted. |
| AC-4 | MET | R18 resolves the remaining selector-capture evidence and m9 consistency gaps. |
| AC-5 | MET | 64K mem0.7 servability remains accepted; 128k remains out of Loop-7 scope. |
| AC-6 | PARTIAL | task19 conc-1/16 perf consolidation remains active. |

## Goal Tracker Update Requests

I updated `goal-tracker.md` directly:

- bumped Plan Version to 26 for Round 18 Review;
- added a Round 18 Review plan-evolution row accepting AC-4 closure;
- changed task16 and task17 verification from `pending R18 review` to `18 Review`;
- left task19 and task20 active;
- left Explicitly Deferred empty.

PENDING
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-19-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-19-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**Optional fallback**: if you could not safely update the mutable section of `goal-tracker.md` directly, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Blocking Side Issues: bug Y blocks AC-2"]
- [E.g., "Add to Queued Side Issues: cleanup Z is non-blocking"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and reconcile the Goal Tracker if justified.
