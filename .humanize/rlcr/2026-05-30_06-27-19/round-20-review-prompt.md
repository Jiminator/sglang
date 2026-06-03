# Code Review - Round 20

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-20-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 20 Summary — AC-5 full-context MEASURED client evidence (conc-16 meets strict TTFT)

## Mainline objective (round contract)
Produce NEW full-context measured AC-5 client-workload evidence (Codex R19: no more restating the
bounded-vs-kernel tradeoff without new evidence). Root-cause the streaming empty-array, run the full
client workload at the **full-context Option-B point** (no `--context-length` cap), publish measured
arrays + attribution + a fail-closed verifier.

## What landed (commit `96bc789cc`)
1. **Streaming root cause — ctx8192-specific.** The R18 empty-latency arrays were the **bounded-context
   server only**: at full-context (DS int8/mem-0.7/radix-on/`context_len=163840`/TP=8) `bench_serving`
   produces real per-request arrays in both fixed-count and window mode (probes: completed=16/64, all
   ttfts real, thousands of ITL tokens). The R19 fail-closed guard passed on the full run (no empty-latency
   rows) — it stays the durable safety net.
2. **Measured full-context AC-5 client run** (steady-state warmup 120 s / window 300 s, conc 16/32/64,
   GSP 4096 ISL / 512 OSL, radix-on proven, `--enable-request-time-stats-logging`):

   | conc | achieved | **P99 TTFT** | `<22 s`? | per-req TPS p50 | `≥30`? |
   |---:|---:|---:|:--:|---:|:--:|
   | 16 | 16.00 | **13.13 s** | ✅ | 24.9 | ❌ |
   | 32 | 31.99 | 25.33 s | ❌ | 19.5 | ❌ |
   | 64 | 47.03 | 77.90 s | ❌ | 17.3 | ❌ |

   Exact per-request arrays + **fail-closed verifier** (`ac5_fullctx_metrics_tool.py --verify` PASS;
   recomputes P99 TTFT/TPS/achieved + asserts no empty-latency rows, output_len==512, errors empty,
   len==completed, 64-hex source SHA). No inferred TTFT.
3. **Measured admission-wait attribution** (`queue_duration` from ReqTimeStats, per-conc by print-time gap,
   n=256/320/315): queue p99 **10.5 / 22.6 / 74.0 s** → P99 TTFT is **admission-queue-dominated** at every
   conc (residual ~2.6/2.7/3.9 s = prefill). KV pool fits the concurrent set (64×4608≈295K<396K) → the
   queue is throughput/decode contention, not KV admission.

## Result (DEC-3 directional)
- **conc-16 MEETS the strict tail-latency SLO at full context (P99 TTFT 13.13 s < 22 s)** with admission
  restored (achieved 16.00) — measured, verifier-checked, with attribution.
- **Per-req TPS misses 30 at every conc** (24.9/19.5/17.3). conc-16's full-context gap (24.9 < 30) is the
  residual DS-selection `torch.topk` over-scan (the R18 bounded-context op-point reached conc-16 closed-batch
  30.3; the exact full-context fix needs the blocked-topk kernel). conc-32/64 = the structural decode-batch
  ceiling (DS ≤ DSA; even DSA is 37.0/29.4, so conc-64 ≥30 is unattainable for either).
- vs R6 full-context cold-flood (17.6/11.5/9.3 TPS): steady-state TPS markedly improved (24.9/19.5/17.3) via
  the R17 decode score-fix + steady-state methodology; TTFT collapsed vs Loop-5 (57.7/132.9/292.0 s).

## Files Changed
- `runs/20260530_dsv32_loop6/ac5_fullctx/` (NEW): `ac5_fullctx_report.md`, `ac5_fullctx_arrays.json` +
  `ac5_fullctx_metrics_tool.py` (exact arrays + fail-closed verifier), `ac5_fullctx_attribution.txt`
  (measured queue_duration per conc + decode components), `get_server_info_fullctx.json` + `meta_c16.json`
  (operating-point + radix-on `.meta.json` proof).
- `.humanize/bitlesson.md` — `BL-20260531-bench-empty-stream-failclosed` extended with the R20 ctx8192-specific
  root-cause; goal-tracker (R20 row + task6 note); round-20 contract/summary (gitignored loop state).
- (No production code change this round; R17 score-fix + R19 bench fail-closed fix stand.)

## Validation
- `ac5_fullctx_metrics_tool.py --verify` → PASS (recomputes the 3 conc rows; fail-closed sanity incl. the
  R18 empty-latency class). Operating point proven: full context, int8, mem 0.7, radix-on, stats-on, TP=8.
- Streaming probes at full-context (fixed-count + window mode) both produced real arrays. `git diff --check`
  clean; commit `96bc789cc` pushed to `jimmy`. GPUs freed (all 8 at 0 MiB; no live server).

## Remaining Items
- **conc-16 per-req TPS at full context (24.9 < 30):** the residual DS-selection topk over-scan. Either the
  exact full-context blocked-topk kernel (research-grade) or an explicit owner bounded-context rescope.
- **conc-32/64:** characterized structural ceiling (DS ≤ DSA; conc-64 unattainable even for DSA).
- **Gated AC-10** — after AC-5 met + AC-3..AC-9 verified. Cross-node smoke (future-gated), DSA conc-64 TPS
  ~29.4 (queued) unchanged. No ABI-lock change; DS-fair AC-12 gate unchanged.

## Goal Tracker Update Request
### Requested Changes:
- Record R20 Plan Evolution: full-context MEASURED AC-5 evidence landed — **conc-16 meets the strict TTFT
  axis (13.13 s < 22 s)** with attribution + fail-closed verifier; per-req TPS measured (24.9/19.5/17.3),
  admission-queue-dominated.
- **Owner decision (now backed by measured full-context data):** conc-16 strict TPS (≥30) at FULL context
  needs the research-grade blocked-topk kernel; the bounded-context op-point already reaches conc-16 30.3
  (closed-batch) and is the natural deployment for the 4608-token client workload (64K servability = separate
  full-context point, AC-8/R16). Please confirm (a) invest the rounds in the full-context blocked-topk kernel,
  or (b) accept the bounded-context op-point as the conc-16 client-SLO deployment. conc-32/64 ≥30 is
  structurally unattainable (DS ≤ DSA) regardless.
### Justification:
This round delivered the measured full-context evidence Codex required (conc-16 strict TTFT met; TPS +
attribution measured, not inferred), resolving the R18/R19 measurement blockers. The only remaining AC-5
gap is the conc-16 TPS axis at full context, whose two resolution paths (research kernel vs bounded-context
deployment) reach the same conc-16 outcome — an owner steer prevents spending multiple rounds on a kernel
that does not change the structurally-unattainable conc-32/64 result.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260531-bench-empty-stream-failclosed
Notes: Extended with the R20 root-cause — the runtime empty-stream cause was localized to the **ctx8192
bounded-context server only**; at full-context `bench_serving` produces valid per-request arrays (fixed-count
+ window mode), so the empty-array was a `--context-length 8192` × GSP-shared-prefix × radix streaming
interaction, not a general window-mode bug. The R19 fail-closed guard is the durable safety net. Applied
existing lessons: BL-20260530-clean-latency-attribution (per-conc queue_duration bucketed by print-time gaps,
measured-vs-inferred, tail-to-tail), BL-20260530-cold-flood-not-steady-state-slo (steady-state warmup120/
window300), BL-20260530-durable-tracked-acceptance-evidence (exact arrays + fail-closed verifier),
BL-20260531-ds-selection-fullwidth-overscan (the conc-16 full-context TPS residual = topk over-scan). No
production code changed this round.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-19-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-19-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-18-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-18-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-17-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-17-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-20-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
