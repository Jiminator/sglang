# Code Review - Round 25

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-25-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 25 Summary — owner-directed Loop-6 close + handoff (finalization round)

## Mainline objective (round contract)
Finalize Loop-6 at its met Minimum Acceptable Scope: correct the one flagged evidence-hygiene nit, record the
terminal state, and surface the terminal fork to the owner. Codex R24 confirmed the Loop-6 Lower Bound is met
(AC-1..AC-9 landed — AC-5 directional-complete per DEC-3, AC-7/AC-8 characterized per DEC-9 — AC-10 deferred;
**0 active tasks, 0 blocking issues**), but cannot emit COMPLETE because the audit's COMPLETE gate requires
*all* ACs fully met with no deferrals (the upper bound the plan itself defers downstream: AC-5 strict + AC-10).

## Owner decision (the terminal fork)
I surfaced the fork via AskUserQuestion. The owner chose: **"Stop Loop-6, defer the AC-10 Tier-2 work as high
priority for Loop 7 and write in draft.md. But make sure we update the roadmap and create
`development/past_implementations/study/08-current-system-architecture.md` to describe the state of our
implementation."** This round executes exactly those three deliverables + the one hygiene fix.

## What landed
1. **Evidence-hygiene fix (commit `27fca1102`).** Codex R24's one queued nit: `topk_design_microbench.json`'s
   note said "C is the no-context-cap win" while the measured rows + markdown finding show C (blocked
   bw=8192/pk=2048) is **worse** than monolithic. Corrected the note in **both** the generator script and the
   committed JSON to state B (context-capping) is the only design reaching conc-16 ≥30 and C/C′ are worse than
   monolithic; **timings unchanged** (A 6.556 / B 2.378 / C 8.498 / C′ 12.331 ms/step).
2. **Loop-7 Tier-2 draft (`development/loop7.md/draft.md`, commit `5d65eed25`).** The deferred AC-10 DS
   long-context recall R&D as the **high-priority Loop-7 mainline**: the recall gap (4K/16K/64K = 75/5/0 vs DSA
   100), the root cause (top_k kernel-locked at `index_topk=2048` + offline selector inferior to the trained
   DSA indexer; dense DS = 100% so decode is sound), the two directions from `ds_on_v32_decision.md` (PRIMARY
   adjustable-`top_k` decode kernel relaxing `indices.shape[-1]==dsa_index_topk`; SECONDARY learned/query-aware
   selector), 128k servability as secondary scope, draft ACs, and pending `gen-plan` decisions.
3. **Roadmap updated (`development/roadmap.md`, commit `5d65eed25`).** §4 LOOP 6 marked **DONE (Minimum
   Acceptable Scope)** with the outcome block; §4.0 strategic gate **DECIDED/open**; §4.1 footprint→admission
   spine **done (directional AC-5)**; §4.2 64K servability + §4.3 accuracy hardening **done**; §4.4 Tier-2
   **explicitly deferred to Loop 7**; §6 **Loop 7 promoted** to the high-priority DS recall R&D; §8 strategic +
   SLO-scope DECs **resolved**; §9 artifact index updated.
4. **As-built architecture doc (`development/past_implementations/study/08-current-system-architecture.md`,
   commit `5d65eed25`).** The implemented DS-on-V3.2 state after Loop 6: operating point (DS int8 / mem 0.7 /
   radix-on / TP=8 / page 64 / fp8 KV / flashmla_kv); compact int8 TokenLabelTable (≈1.78×, ~6.48 GB/rank) +
   write/consume/launcher/radix-fixture paths; decode selection hot path (R17 score early-exit, torch.topk
   2048, R23 deterministic tie-break, the 163840 over-scan, the R24 microbench); the kernel ABI lock; measured
   perf (conc-16 P99 TTFT 13.13s<22 directional; TPS the structural ceiling) and recall (75/5/0); the
   fail-closed verifier; deferred items with owners; and a key-files index.

## Result — Loop-6 is at its terminal state
- The **Minimum Acceptable Scope (plan Lower Bound) is met** and confirmed by Codex R24: AC-1..AC-9 landed
  (AC-5 directional per DEC-3, AC-7/AC-8 characterized per DEC-9), AC-10 owner-deferred. **0 in-scope Loop-6
  work remains.**
- The one queued evidence-hygiene item is **resolved**. All deferred/downstream work (AC-5 strict all-conc
  SLO, AC-10 Tier-2 recall) is documented, handed off (Loop-7 draft + open gate), and tracked.
- The RLCR loop **cannot self-emit COMPLETE** for a Lower-Bound close (its gate is the all-ACs upper bound the
  plan defers downstream). Per the owner's decision, **the loop is to be stopped here**: the owner will run
  `/humanize:cancel-rlcr-loop` (I am not permitted to run it). Continuing to spin the loop yields no further
  in-scope progress.

## Files Changed
- `runs/20260530_dsv32_loop6/ac5_topk_design/topk_design_microbench.{py,json}` — corrected stale note (`27fca1102`).
- `development/loop7.md/draft.md` (NEW), `development/past_implementations/study/08-current-system-architecture.md`
  (NEW), `development/roadmap.md` (Loop-6 DONE + Loop-7 promoted) — handoff docs (`5d65eed25`).
- `.humanize/rlcr/2026-05-30_06-27-19/` — goal-tracker (R25 plan-evolution row, JSON-note issue resolved,
  AC-10 handoff note, plan version → R25), round-25 contract/summary (gitignored loop state).
- **No production code changed** (finalization + handoff round).

## Validation
- `topk_design_microbench.json` note now matches the measured rows + markdown finding; timings preserved
  (A 6.556 / B 2.378 / C 8.498 / C′ 12.331). `git diff --check` clean; commits `27fca1102` + `5d65eed25`
  pushed to `jimmy`. AC-5 full-context verifier still PASS (R23 hardened). GPUs free (data/doc round).

## Remaining Items
- **None for Loop 6.** AC-10 Tier-2 recall R&D is handed off to Loop 7 (gate open; `development/loop7.md/draft.md`).
  Strict all-conc SLO is downstream. Queued/future (unchanged): cross-node wrapper smoke; DSA-default conc-64
  TPS ~29.4. The owner stops the loop via `cancel-rlcr-loop`.

## Goal Tracker Update Request
### Requested Changes:
- **Confirm the Loop-6 terminal state** (no AC status change): AC-5 directional-complete (DEC-3), AC-10
  Explicitly Deferred (now handed off to Loop 7, gate open), strict all-conc SLO queued downstream — exactly
  Codex's R24-review reconciliation, plus the R25 plan-evolution row for the owner-directed close + handoff.
- **Mark the one queued evidence-hygiene issue RESOLVED** (stale `topk_design_microbench.json` note, fixed in
  `27fca1102`).
### Justification:
This is an owner-directed finalization round: the Loop-6 Minimum Acceptable Scope was already met and
Codex-confirmed in R24; the owner chose to stop the loop and hand off. The work is the one outstanding hygiene
fix + the three handoff deliverables (Loop-7 draft, roadmap close, as-built doc). No new implementation is
in-scope; the COMPLETE gate (all ACs, no deferrals) is the upper bound the plan defers to Loop 7 / downstream.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: No new multi-round problem→solution lesson this round; it is a finalization/handoff round (one
evidence-hygiene correction + three handoff documents + owner-directed loop stop), no production code changed.
The existing lesson `BL-20260530-durable-tracked-acceptance-evidence` was *applied* (the corrected JSON note
and the new handoff docs state only measured/verified numbers, consistent with the committed evidence), but
nothing was added or updated, so Action is none / Lesson ID(s) NONE.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
1aa24cfc1 [Sparsity] Loop-6: refined plan v1 + QA ledger + DEC-5 roadmap deferral
88c6498e5 [Sparsity] Loop-6 R0: strategic recall-R&D gate + footprint feasibility budget
84d3410b9 [Sparsity] Loop-6 R1: int8-symmetric compact TokenLabelTable (flag-gated, fp16 default, CUDA-graph-safe)
e85cd2564 [Sparsity] Loop-6 R2: scale-aware proof/sanity consumers + AC-3.1/AC-6 evidence
5d8e47fb3 [Sparsity] Loop-6 R3: serve_double_sparsity.sh exposes SIGNATURE_DTYPE (compact-table selection)
8a05b1688 [Sparsity] Loop-6 R3: real-mask NIAH non-regression PASS (int8 DS vs fp16 Loop-5 baseline, TP=8)
75e68053f [Sparsity] Loop-6 R4: AC-4 mem-fraction sweep PASS (int8 lifts no-OOM ceiling 0.6->0.7, TP=8)
91e9c20a3 [Sparsity] Loop-6 R5: AC-4 evidence addendum (full HBM budget + durable no-OOM proof)
8883848e9 [Sparsity] Loop-6 R6: AC-5 client-SLO directional result (int8 @ 0.7 radix-on, TP=8) + attribution
51dd009b8 [Sparsity] Loop-6 R7: durable AC-5 evidence + corrected per-conc attribution
bd09d1ca7 [Sparsity] Loop-6 R8: exact-recomputable AC-5 evidence + reconciled attribution
57f86b66f [Sparsity] Loop-6 R9: exact ITL source + fail-closed AC-5 verifier
d6e884aa9 [Sparsity] Loop-6 R10: AC-9 within_budget from real usage.prompt_tokens
daad92923 [Sparsity] Loop-6 R10: AC-9 within-budget gate re-run on hardware (real tokens)
2fd2c6937 [Sparsity] Loop-6 R10: AC-6 opt-in / DSA-default product proof on hardware
0e1ce974d [Sparsity] Loop-6 R11: AC-6 redo — proper-methodology DSA SLO + radix-on toggle
d0cc9fdc9 [Sparsity] Loop-6 R12: benchmark scripts pass --host to bench_serving
f9bc51b13 [Sparsity] Loop-6 R12: recomputable DSA SLO evidence + honest AC-6 verdict
5e6d3afb5 [Sparsity] Loop-6 R13: AC-7 — 3-trial DS+DSA re-sweep at the lifted point (characterized)
147b6d05f [Sparsity] Loop-6 R14: AC-7 exact-recomputable metrics + fail-closed verifier
99e51ad00 [Sparsity] Loop-6 R14: AC-7 profiling discharged at AC-7 methodology
40ccc4b63 [Sparsity] Loop-6 R15: AC-7 evidence review-clean (verifier precision + provenance + reconciliation)
9915630ca [Sparsity] Loop-6 R16: AC-8 64K servability PASS at lifted DS int8/mem-0.7
ece26eb52 [Sparsity] Loop-6 R17: AC-5 decode-throughput remediation -- DS selection over-scan fix
fcc2d1cdb [Sparsity] Loop-6 R18: AC-5 conc-16 strict-decode PASS via bounded-context op-point
7f896b454 [Sparsity] Loop-6 R19: bench_serving fail-closed on empty-latency streaming runs
96bc789cc [Sparsity] Loop-6 R20: AC-5 full-context measured client workload + attribution
991666b58 [Sparsity] Loop-6 R21: rebuild AC-5 full-context evidence to R9 fail-closed standard
704be382f [Sparsity] Loop-6 R22: harden AC-5 verifier to fail closed on workload identity
8ab6c7db0 [Sparsity] Loop-6 R22: exact blocked top-k algorithm + adversarial regression suite
2715b7382 [Sparsity] Loop-6 R23: deterministic top-k tie-break + AC-5 verifier workload-volume
ca46eced1 [Sparsity] Loop-6 R24: blocked-topk design microbench -- full-context kernel does NOT win
27fca1102 [Sparsity] Loop-6 R25: correct stale microbench JSON note (evidence hygiene)
5d65eed25 [Sparsity] Loop-6 close: roadmap DONE + as-built arch doc + Loop-7 Tier-2 draft
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-24-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-24-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-23-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-23-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-22-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-22-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-25-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
