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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-20-contract.md

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
# Round 19 Review Result

Mainline Progress Verdict: ADVANCED

Round 19 advanced the AC-6 evidence by recording closed-batch decode TPS for
DS-default, DS-hybrid, and DSA at conc-1/16. The committed JSON artifacts match the
headline TPS values in `m11_perf_consolidation.md`:

- DS-default: 39.81 / 27.57 tok/s/req (`perf_ds_default_c1.json`,
  `perf_ds_default_c16.json`)
- DS-hybrid: 40.10 / 27.57 tok/s/req (`perf_ds_hybrid_c1.json`,
  `perf_ds_hybrid_c16.json`)
- DSA/native-NSA: 83.25 / 55.37 tok/s/req (`perf_dsa_c1.json`,
  `perf_dsa_c16.json`)

However, I do **not** accept full AC-6/task19 closure. The original plan requires
TTFT as part of the conc-1/16 guardrail record, and R19 did not produce fresh TTFT
artifacts.

## Acceptance Criteria Audit

| AC | Status | Evidence / Blocker |
|----|--------|--------------------|
| AC-1 | MET | Prior R8 review remains accepted: fail-closed oracle, dedicated sink, oracle-off byte-equivalence + zero-alloc replay, separated baseline, stride=1 reference, and AC-1.1 force-inclusion evidence are all tracked in the goal tracker. |
| AC-2 | PARTIAL | Recall uplift evidence exists, but task20 final strategic-gate supersession record is still not written. It should wait for corrected AC-6 evidence. |
| AC-3 | MET | Prior R7-R9 evidence remains accepted: graph-safe non-learned scorer/head/anchor variants, TP=8 determinism, within-budget parity, MMLU within 1.0pp, and N=50 graph-mode recall matrix. |
| AC-4 | MET | Prior R18 review accepted production-ready lifted-budget decode: graph-safe primitives, live graph-mode recall, disposition consistency, and lifted+speculative fail-closed guard. |
| AC-5 | MET | 64K servability at mem0.7 remains verified from the separated baseline; 128k remains out of Loop-7 scope. |
| AC-6 | PARTIAL | R19 records decode TPS, memory, graph, and admission, but not fresh conc-1/16 TTFT. `m11_perf_consolidation.md:80-83` even defines the MET claim as decode TPS/GPU mem/graph/admission only, omitting TTFT, while the plan explicitly requires `TTFT, decode TPS/req, GPU memory, graph-replay success, admission` at conc-1/16 (`development/loop7/refined_plan_v1.md:76-80`). |

## Forgotten Items Detection

No original task is missing from the tracker after this review. I corrected tracker
drift: task19 had been summarized as complete, but it is not verified because the
TTFT guardrail is absent; task19 is now active/partial again.

Tasks still active:

- `task19` / AC-6: add fresh conc-1/16 TTFT guardrails and update `m11`.
- `task20` / AC-2: write the final strategic-gate supersession record after task19
  is corrected.

## Deferred Items Audit

Explicitly Deferred is empty. No deferral currently contradicts the ultimate goal.
Learned/distilled selector work remains a queued out-of-scope follow-on under DEC-5,
not an active deferral.

## Goal Completion Summary

Acceptance Criteria: 4/6 met (0 deferred)  
Active Tasks: 2 remaining  
Estimated remaining rounds: 2  
Critical blockers: none outside the mainline gaps below

## Mainline Drift Audit

The round objective was clear and singular, but the R19 contract narrowed AC-6 to
decode TPS and omitted TTFT despite the immutable plan and R18 review requiring it.
This is mainline drift by under-scoping, not a side issue. The decode-TPS work still
advanced the goal and should be retained as part of the final AC-6 report.

Blocking Side Issues: 0  
Queued Side Issues: 3

## Implementation Review

### Mainline Gaps

1. **AC-6/task19 is incomplete: no fresh conc-1/16 TTFT artifacts.**

   Evidence:
   - The plan requires TTFT at conc-1/16 as part of the guardrail record
     (`development/loop7/refined_plan_v1.md:76-80`).
   - R18 review carried the same requirement into task19: record conc-1 and conc-16
     TTFT, decode TPS/req, GPU memory, graph replay, admission, exact args/config,
     commit, GPU type, and artifact paths (`round-18-review-result.md:44-55`).
   - R19 `m11` reports only TPS/memory/graph/admission in the table
     (`development/loop7/m11_perf_consolidation.md:9-23`) and cites the old Loop-6
     conc-16 TTFT instead of a fresh R19 conc-1/16 TTFT run
     (`development/loop7/m11_perf_consolidation.md:36-40`, `:75-78`).
   - `perf_closed_batch.py` is a non-streaming `/generate` E2E/decode probe. It records
     request wall time and completion tokens only; there is no first-token timestamp
     (`development/loop7/perf_closed_batch.py:25-44`, `:67-78`).
   - The project docs state that `bench_serving` measures TTFT and can emit details
     including `ttfts` (`docs/developer_guide/bench_serving.md:5-9`,
     `:160-172`).

   Required action:
   - Run a TTFT-capable benchmark at the same Loop-7 op-point for DS-default,
     DS-hybrid, and DSA/native-NSA at conc-1 and conc-16. Prefer the existing
     `bench_serving` path or an equivalent streaming probe that records first-token
     timestamps.
   - Commit the TTFT artifacts (p50/p99 or at least per-request arrays/details), exact
     launch args/config, graph status, memory, admission counts, commit SHA, GPU type,
     and artifact paths.
   - Update `m11_perf_consolidation.md` so AC-6 is claimed only after TTFT is present.

2. **task20 must remain blocked on corrected AC-6 evidence.**

   The final decision record should not cite the current `m11` as complete AC-6 source
   evidence. It may cite the R19 decode-TPS result, but must wait for the missing TTFT
   guardrail or explicitly document an owner-approved plan evolution.

### Blocking Side Issues

None.

### Queued Side Issues

1. Preserve/cite the R8 oracle-sink provenance before task20, or cite the hardcoded
   `stride=1` call site plus the committed aggregate explicitly.
2. Remove plan/workflow markers from production code/comments/tests before final
   cleanup/merge.
3. Learned/distilled selector work remains out of scope unless explicitly approved.

## Goal Tracker Update Requests

I updated `goal-tracker.md` directly:

- bumped Plan Version to 28 for Round 19 Review;
- added a Round 19 Review plan-evolution row rejecting full AC-6 closure;
- reactivated `task19` as partial because fresh TTFT guardrails are missing;
- left `task20` active and dependent on the corrected AC-6 source artifact;
- removed `task19` from Completed and Verified;
- left Explicitly Deferred empty.

## Progress Stagnation Check

Not stagnant. Recent rounds have materially advanced the mainline: R16-R18 closed the
AC-4 production graph hardening and disposition, and R19 added useful decode-TPS
guardrail evidence. The current gap is a concrete under-scoped AC-6 measurement, not a
repeat circular discussion.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-20-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

### Post-Alignment Check Action Items

This round follows a Full Goal Alignment Check. Pay special attention to:
- **Forgotten Items**: Codex may have identified tasks that were being ignored. Address them.
- **AC Status**: If any Acceptance Criteria were marked NOT MET, prioritize work toward those.
- **Deferred Items**: If any deferrals were flagged as unjustified, un-defer them now.
- **Queued Issues**: Keep non-blocking follow-up work queued unless it now clearly blocks mainline progress.

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-20-summary.md

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
