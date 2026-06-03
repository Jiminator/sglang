# Code Review - Round 15

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-15-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 15 Summary — AC-7 evidence made review-clean (verifier precision + provenance + reconciliation)

## Mainline objective (round contract)
Close the four data-only AC-7 evidence residuals from Codex's R14 review (no hardware): the
verifier wasn't fail-closed at the published 3-decimal precision, it didn't validate its own
provenance/shape, the profiling artifact wasn't reconciled to completed counts, and `ac11_analysis.md`
had a stale "attributed via AC-5" verdict sentence. AC-5's directional verdict + the open DS
strict-SLO blocker stay tracked, not this round's objective.

## What landed (commit 40ccc4b63, data-only)
1. **Verifier fail-closed at report precision + provenance/shape.** `ac7_metrics_tool.py`: `TOL`
   0.05 → **0.0005** (the comparator renders 3 decimals). Added sanity checks: every trial has a
   **64-hex `sha256`**, required scalars numeric, **3 trials/side/conc**, `completed > 0`,
   `duration_s ≥ 600` (window floor), and `len(per_req_gen_tps) == len(ttfts_s) == completed`.
   **Tamper tests now exit 1:** a median-moving concurrency (recomputes 46.987 ≠ published 46.983),
   a short SHA (`deadbeef`), and a dropped `per_req_gen_tps` element; **clean exits 0 PASS**
   (recomputes the `ac11_resweep.md` achieved/TPS/TTFT rows, DS+DSA, all conc).
2. **Profiling provenance + reconciliation.** `queue_attribution.txt` now states explicitly it is a
   **separate request-time-stats reproduction run** (the 3-trial sweep didn't enable the flag),
   identical methodology, and **reproduces** the AC-7 result (TTFT 12.8/25.4/100.8 s, achieved
   16/32/47). Added source JSONL SHA256 + DS-log SHA + per-conc window starts, and the row-count
   accounting: valid benchmark rows **320/384/378 = measured completed 256/320/320 + 120s-warmup-epoch
   64/64/58** (ReqTimeStats logs every request incl. warmup).
3. **`ac11_analysis.md` verdict** "attributed via AC-5" → cites `decode_batch_ac7.txt` +
   `queue_attribution.txt` (AC-7 methodology). The two remaining AC-5 mentions are corroboration /
   explicitly "background".

## Result
AC-7 evidence bundle is review-clean: exact-recomputable + fail-closed at published precision +
provenance-validated, profiling reconciled with explicit reproduction provenance, attribution cites
AC-7-methodology artifacts. The AC-7 result stands (characterized, DEC-9): **admission restored** (DS
achieved 16/32/47 = 100/100/73% vs Loop-5 14.5/24.6/35.7); DS-vs-DSA parity FAIL is a DEC-7
directional follow-up, not a footprint regression. The **AC-5 DS strict-SLO miss remains the open
mainline blocker**.

## Files Changed
- `runs/20260530_dsv32_loop6/ac7_resweep/`: `ac7_metrics_tool.py` (TOL + provenance/shape checks), `queue_attribution.txt` (provenance + reconciliation), `ac11_analysis.md` (verdict cites AC-7 artifacts).
- `.humanize/bitlesson.md` (durable-evidence lesson: tolerance ≤ published precision + validate SHA/shape provenance), goal-tracker (R15 row; task8/AC-7 done-characterized; evidence-bundle blocker → RESOLVED), round-15 contract/summary (gitignored loop state).

## Validation
- `ac7_metrics_tool.py --verify`: clean exit 0 PASS (recomputes published rows + sanity). Three tamper tests (rendered-value, bad SHA, length) each exit 1.
- `queue_attribution.txt` reconciliation: 320/384/378 = measured + warmup, source SHAs committed, reproduction labeled.
- `ac11_analysis.md`: no "attributed via AC-5" verdict; AC-5 only as corroboration/background.
- `git diff --check` clean; commit 40ccc4b63 pushed to `jimmy`; no hardware (data-only; no servers booted this round).

## Remaining Items
- **Open mainline blocker:** AC-5 DS strict client SLO (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). The AC-7 data confirms the root cause (admission-queue + DS throughput < DSA); the AC-5 remediation (smallest scheduling/decode/operating-point change) is the next focus after AC-8.
- **Cross-node wrapper smoke** PARTIAL (future cross-node only). **DSA conc-64 TPS ~29.5** Queued. **AC-8** (~70K probe), gated **AC-10** — later. No FlashMLA decode-assert changes; DS-fair thresholds unchanged.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260530-durable-tracked-acceptance-evidence
Notes: Extended clause (d): a fail-closed verifier's comparison TOLERANCE must be ≤ the PUBLISHED precision (R14's TOL=0.05 against a 3-decimal report let a value rendering 46.973 vs published 46.983 pass; R15 → ≤0.0005), and the verifier must VALIDATE the provenance/shape fields it relies on (full 64-hex SHA256 — a `deadbeef` SHA passed until checked; required scalars numeric; array length == completed), not just recompute the metric. Applied existing lessons: BL-20260530-clean-latency-attribution (per-conc queue bucketed by reliable `.meta.json` run windows + full row reconciliation incl. the warmup-epoch accounting), and the push-between-commits preference. No new lesson — same durable-acceptance-evidence family, sharpened across R13/R14/R15.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-14-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-14-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-13-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-13-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-12-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-12-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-15-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
