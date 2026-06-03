# Code Review - Round 15

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-15-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 15 Summary — Loop 7

## Mainline objective (round-15-contract.md)
**task17 — the Tier-2.A landing disposition (deferred-with-evidence) that CLOSES AC-4.**
An `analyze` task: draft → `/humanize:ask-codex` review → integrate.

## Outcome: ACHIEVED — AC-4 closes via deferred-with-evidence; task16 explicitly deferred.

## Decision
Tier-2.A (the opt-in lifted-budget decode path) lands as an **opt-in, eager,
default-off research path with recorded served recall evidence**, and its
**production hardening (task16) is explicitly deferred to a follow-on**. This closes
AC-4 under the plan's "production-ready **or** deferred-with-evidence" branch (the
M4 dependency gate requires the disposition to exist; it now does).

## Justification (why deferred, not hardened)
The M0 oracle makes Tier-2.A **bounded-secondary**: a wider decode budget recovers
ONLY the **4K budget-limited** regime (proven served in R14: 75% → 95%, +20pp
material). **16K is budget-partial (~46% cap) and 64K is scorer-limited** — the
long-context goal that motivated the loop is served by the **landed,
production-ready Tier-2.B hybrid scorer** (AC-3 MET). The plan gates the
HIGH-COST/HIGH-RISK task16 kernel on *"the recall win justifies the heavy kernel"*;
a 4K-only win on a secondary lever does not. This is the plan-aligned,
theoretically-sound scoping decision given the measured evidence — not pragmatism
over correctness (the loop's correct primary lever, Tier-2.B, IS landed
production-ready).

## Why this is a VALID close (DEC-4/DEC-6 conditions, all satisfied)
1. **Recall evidence recorded** — M0 regime attribution + R14 served 4K recovery.
2. **DSA default untouched** — the `flashmla_kv` `dsa_index_topk` assert is unchanged;
   default-off decode byte-identical (default-off guard; full suite confirms).
3. **Research path gated out of production capture** — validator requires
   `--disable-cuda-graph` for the lifted path (the dequant allocates / not graph-safe).
4. **Eager number labeled as such** — the 95% is eager-mode; the graph number is a
   deferred-follow-on measure, not claimed here.

## Work Completed (`analyze`, Claude + ask-codex)
- **`development/loop7/m9_tier2a_disposition.md`** (new): the disposition record —
  the decision, the M0 + R14 evidence, the full landed opt-in surface (R10–R14), the
  DEC-4/DEC-6 conditions, the M0 bounded-secondary justification, and a **precise
  task16 follow-on scope** (alloc-free `out=` dequant, graph-safe fixed-shape compact
  remap, q-padding scratch, zero-alloc-replay + graph-mode recall re-measure,
  graph-captured TP=8 determinism, perf).
- **ask-codex review** (the `analyze` step): **"No high-signal invalidating issue
  found"** — the deferral is justified and the DEC-4/DEC-6 conditions are met.
  Integrated its two refinements: (a) added graph-captured TP=8 lifted-width
  determinism to the task16 follow-on (the R14 TP test is the eager/logical path);
  (b) corrected the wording so 16K (budget-partial/capped) and 64K (scorer-limited)
  are not compressed into one claim. Output cited:
  `.humanize/skill/2026-06-02_16-34-28-2810456-a017c2ed/output.md`.

## Files Changed
- `development/loop7/m9_tier2a_disposition.md` (new). Commit `b70f48d36`.
- **No production-code change** (a documentation/decision round).

## Validation
- Full DS unit suite (4 files) → **341 passed + 9 subtests** (unchanged — no code
  touched; confirms no regression).
- The disposition's claims are backed by committed artifacts (`m8`, the NIAH JSONs,
  `m0_oracle_finding_r4.md`) and the implementation gates Codex re-verified
  (validator eager-required, `_forward_lifted_budget` behind the default-off guard,
  the default `flashmla_kv` assert intact).

## AC status after R15
- **AC-4 → MET** (closed via deferred-with-evidence; the disposition record exists,
  recall evidence recorded, DSA default untouched, research path eager-gated).
- AC-1/AC-3/AC-5 MET (prior). **4/6 MET.**
- AC-2 PARTIAL (task20 final decision record), AC-6 PARTIAL (task19 perf consolidation).

## Remaining Items (active mainline)
- **task19 (AC-6, next mainline)** — final perf guardrails at conc-1/16 (TTFT, decode
  TPS/req, GPU mem, graph-replay, admission) + Tier-1 spine non-regression + the
  consolidated DS-vs-DSA recall/perf report.
- **task20 (AC-2)** — the final strategic-gate supersession decision record (supersede
  the Loop-6 Tier-2.A-primary ordering with the M0/R7/R8/R14 evidence + the AC-4 disposition).
- **task16** — Explicitly Deferred (follow-on; scope in `m9`).
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: A disposition/decision round — no new reusable engineering pitfall. The
  decision (defer a bounded-secondary lever's production hardening per the measured
  M0 evidence, closing via DEC-4/DEC-6 deferred-with-evidence) is a project-specific
  strategic call recorded in `m9`, not a cross-round engineering lesson. The
  evidence-methodology lessons it relies on (eager-vs-graph labeling, CP materiality,
  durable-tracked-acceptance-evidence) already exist.

## Goal Tracker
Updated directly (Plan Version 20): R15 Plan Evolution row; task17 → Completed and
Verified (pending R15 review); **task16 → Explicitly Deferred** with justification +
AC-impact (none on the long-context goal); Active Tasks now task19 + task20. No Goal
Tracker Update Request needed.
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
e7cf1f146 [Sparsity] Loop-7 R9: port anchor-budget variant to the graph-safe path (AC-3)
c41e5193a [Sparsity] Loop-7 R10: lifted-budget ABI + design record (AC-4 task13)
a62ce91de [Sparsity] Loop-7 R11: fail-closed lifted-budget decode opt-in at the validator
d187f59f4 [Sparsity] Loop-7 R12: lifted-budget decode index core + flash_mla_sparse_fwd kernel proof
2ba4dafc1 [Sparsity] Loop-7 R13: wire the served eager lifted-budget decode branch + enable the seam
0ad20774a [Sparsity] Loop-7 R14: binding served 4K recall recovery for the lifted-budget path
b70f48d36 [Sparsity] Loop-7 R15: Tier-2.A landing disposition (deferred-with-evidence) — closes AC-4
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-14-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-14-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-13-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-13-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-12-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-12-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-15-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
