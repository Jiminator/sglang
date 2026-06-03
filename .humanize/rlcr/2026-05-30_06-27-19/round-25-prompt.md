Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop6/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop6/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-25-contract.md

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
# Round 24 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: Lower Bound accepted: 9/10 ACs addressed to the plan's MVP grading, 1/10 deferred. Strict/full completion: 8/10 fully met, AC-5 directional only, AC-10 deferred.

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `development/CLIENT_SLOS.md`, `goal-tracker.md`, Round 21-23 summaries/reviews, Round 24 summary, commit `ca46eced1`, the AC-5 full-context artifacts, and the R24 top-k design microbench.

## Implementation Review

Round 24 is data/evidence only; it does not modify production code. That is acceptable for this round because the round objective was to decide whether the owner-chosen full-context blocked-top-k kernel is worth building before spending more rounds on it.

The R24 microbench supports the decision to stop kernel work for Loop 6. The committed artifact reports A monolithic full-width top-k at 6.556 ms/step, B live-width/capped at 2.378 ms, C blocked `bw=8192/pk=2048` at 8.498 ms, and C-prime torch-full blocked at 12.331 ms. I reran the same no-write timing shape on idle H200 hardware and reproduced the ordering: A 6.582 ms, B 2.553 ms, C 8.707 ms, C-prime 12.314 ms. I also sampled adjacent exact full-context blocked geometries (`bw=4096,8192,16384,32768,65536`, `pk=2048` where exact) and they remained around 8.46-9.51 ms, worse than monolithic. That makes the owner decision evidence-backed for the practical blocked-top-k family under consideration.

The AC-5 full-context verifier still recomputes the accepted strict numbers from committed arrays and sidecars: c16 P99 TTFT 13.13s / 24.9 TPS, c32 25.33s / 19.5 TPS, c64 77.90s / 17.3 TPS. This is not a strict client-SLO pass per `development/CLIENT_SLOS.md` (`P99 TTFT < 22s` and `30 TPS per request`), but it is a valid DEC-3 directional result under the refined plan's Lower Bound: the footprint -> admission -> TTFT spine is validated with measured attribution, and the strict miss is recorded rather than hidden.

One evidence hygiene issue: `topk_design_microbench.json` says "C is the no-context-cap win", but the measured rows show C is worse than monolithic. The markdown finding and numeric table are correct, so this does not block the round, but the JSON note should be corrected if reused in a handoff.

## Goal Tracker Audit

| AC | Status | Evidence / blocker / justification |
|----|--------|------------------------------------|
| AC-1 | MET | Strategic gate doc verified earlier: `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`. |
| AC-2 | MET | Footprint budget and binding int8 lever verified earlier: `runs/20260530_dsv32_loop6/footprint_feasibility.md`. |
| AC-3 | MET | Compact int8 table, scale-aware consumers, launcher surface, selection-equivalence, real-mask NIAH, and decode microbench verified earlier. |
| AC-4 | MET | Lifted DS int8/mem0.7 point, full HBM budget, no-OOM long generate, and NVML plateau verified earlier. |
| AC-5 | PARTIAL / DIRECTIONAL-COMPLETE | Full strict SLO is not met: TPS misses at all conc and TTFT misses at c32/c64. Accepted for Loop 6 under DEC-3 Lower Bound because the full-context run records absolute numbers, radix/workload proof, fail-closed verifier, and measured queue attribution; R24 microbench supports owner closure by showing the chosen full-context blocked-top-k path does not reach c16 >=30. |
| AC-6 | MET | Opt-in DS and DSA-default/no-table product proof verified in R12 under owner-approved non-regression semantics. |
| AC-7 | MET / CHARACTERIZED | 3-trial DS+DSA re-sweep verified in R15 as characterized/soft-met. |
| AC-8 | MET | 64K servability at lifted DS int8/mem0.7 full-context point verified in R16. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. |
| AC-10 | DEFERRED | Owner-authorized R24 deferral to its own loop. This is allowed by the plan Lower Bound if the Tier-1 spine consumes Loop 6, but it is incomplete for full all-AC completion. |

Forgotten items: none after tracker reconciliation. All original tasks are now represented as Completed/Verified or Explicitly Deferred. The strict all-concurrency DS SLO is queued as downstream strict-SLO work, not hidden.

Deferred items audit: AC-10 deferral is valid for the Loop-6 Lower Bound because the plan explicitly allows Tier-2 to move to its own loop if Tier-1 consumes Loop 6. It should not be un-deferred in this loop. It does not contradict the Lower Bound, but it does mean the full original plan is not complete.

Goal completion summary:

```text
Acceptance Criteria: 9/10 lower-bound accepted (1 deferred)
Strict/full completion: 8/10 fully met, AC-5 directional, AC-10 deferred
Active Tasks: 0 remaining for Loop-6 Lower Bound
Estimated remaining rounds: 0 for Loop-6 Lower Bound; at least one new loop for AC-10 / strict SLO
Critical blockers: none for Lower Bound; AC-10 deferred and strict all-concurrency SLO for full completion
```

## Mainline Drift Audit

The current round's objective was clear and singular: decide whether the full-context blocked-top-k kernel should be built. R24 advanced the mainline by replacing a speculative kernel plan with measured design evidence and an owner decision. This is not side-issue churn.

Blocking Side Issues: 0 for the Loop-6 Lower Bound.

Queued Side Issues: 4: strict DS all-concurrency SLO downstream work, DSA-default c64 TPS around 29.4, cross-node wrapper smoke for future remote artifacts, and the stale R24 JSON note.

## Tracker Update

I updated `goal-tracker.md` mutable sections:

- Plan version moved to Round 24 Review.
- Added a `24-review` plan-evolution row.
- Removed AC-5 and AC-10 from Active Tasks.
- Moved AC-5 into Completed and Verified as directional-complete, explicitly not shippable/full strict.
- Kept AC-10 in Explicitly Deferred.
- Moved the strict all-concurrency SLO miss to queued downstream work.
- Added the stale JSON note as a queued evidence-hygiene issue.

I reject Claude's requested `COMPLETE` output for this checkpoint. The tracker can say the Loop-6 Lower Bound is met, but the user instructions for this audit allow `COMPLETE` only when every original AC is fully met with no deferrals. AC-10 is deferred, and AC-5 is directional rather than strict.

## Stagnation Check

Development is not stagnating. R20-R23 repeatedly tightened evidence and verifier quality around the same AC-5 axis, but R24 added new measured design evidence that resolved the remaining kernel-vs-close decision. The result is a lower-bound close, not a circular restatement. Because AC-10 remains deferred and strict full completion is not achieved, this should end as `NOT COMPLETE`, not `STOP` and not `COMPLETE`.

## Action Items

Mainline Gaps:

1. Do not claim full strict/shippable Loop-6 completion. AC-5 is directional-complete only, and AC-10 is deferred.
2. Start a separate Tier-2 loop for AC-10 if recall R&D remains desired.
3. Start a separate strict-SLO loop if the client requires all-concurrency DS `P99 TTFT < 22s` and `>=30 TPS/req`.

Blocking Side Issues:

None for the Loop-6 Lower Bound.

Queued Side Issues:

1. Correct the stale `topk_design_microbench.json` note if the R24 evidence bundle is revised.
2. Run cross-node wrapper smoke before future remote-host benchmark artifacts.
3. Treat DSA-default c64 TPS around 29.4 as a separate DSA/client-SLO tension if the client requires strict default compliance.

## Validation Performed

- `python3 runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py --verify` -> PASS; c16 13.13s/24.9 TPS, c32 25.33s/19.5, c64 77.90s/17.3.
- No-write rerun of the R24 microbench logic -> A 6.582 ms, B 2.553 ms, C 8.707 ms, C-prime 12.314 ms.
- Extra blocked-geometry timing check for `bw=4096,8192,16384,32768,65536` -> all worse than monolithic.
- `git diff --check` -> clean.
- Updated `goal-tracker.md` mutable section only; immutable section untouched.

NOT COMPLETE
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-25-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-25-summary.md

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
