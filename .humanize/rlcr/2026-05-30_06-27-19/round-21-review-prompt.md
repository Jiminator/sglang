# Code Review - Round 21

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-21-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 21 Summary — AC-5 full-context evidence rebuilt to the R9 fail-closed standard

## Mainline objective (round contract)
Fix Codex R20 blocking issue 1: the R20 AC-5 verifier stored only a DERIVED `per_req_gen_tps` and re-checked
it, so mutating that array to 100.0 still "passed" the strict TPS axis (tamperable; below the R9 bar). Rebuild
the full-context AC-5 evidence to the R9 standard from the existing raw JSONLs (no re-run).

## What landed (commit `991666b58`, data-only)
1. **Exact committed source** (`ac5_fullctx_arrays.json`): per-request `ttfts_s`, `itl_sum_s`, `output_lens`,
   `input_lens`, `errors_empty`, gen-nonempty count, full 64-hex source SHA256, and the stored headline +
   aggregate means — NOT a stored derived TPS array.
2. **Fail-closed verifier** (`ac5_fullctx_metrics_tool.py --verify`): RECOMPUTES P99 TTFT = p99(ttfts) and
   per-req TPS p50 = p50(output_len/itl_sum) **from the raw committed arrays** (no derived metric); adds
   **aggregate-mean integrity** (sensitive to every element — catches a single-element tamper that a robust
   median misses); asserts the empty-latency class (every ttft>0, itl_sum>0, output_len==512, errors empty,
   len==completed, gen-nonempty==completed, 64-hex SHA); and validates the operating point from **all three**
   `.meta.json` sidecars (int8 / mem0.7 / radix-on / fixture / full context / TP=8 / stats-on).
   **6 tamper tests each exit 1** (single itl_sum, single output_len, single ttft=0, stored TPS p50→100,
   stored P99 TTFT→5000, sidecar disable_radix_cache→True); clean exits 0 PASS. This closes the R20 leak
   (the exact R20 analog — set stored TPS to 100 — now exits 1).
3. **Committed c32/c64 sidecars** (R20 had only c16). **Filled the decode-component breakdown** in
   `ac5_fullctx_attribution.txt`: per-req decode TPS = gen/#running-req = **24.9/19.5/17.3** at batch
   16/32/38 (matching the client arrays exactly) + the DSA FlashMLA+MoE floor reference (AC-7 verified
   46.1/37.0/29.4 → ~21.7 ms step) + the DS-selection delta (R17 microbench).

## Result (numbers unchanged — measured, now exact-recomputable + fail-closed)
| conc | achieved | P99 TTFT | <22s | per-req TPS p50 | ≥30 |
|---:|---:|---:|:--:|---:|:--:|
| 16 | 16.00 | **13.13 s** | ✅ | 24.9 | ✗ |
| 32 | 31.99 | 25.33 s | ✗ | 19.5 | ✗ |
| 64 | 47.03 | 77.90 s | ✗ | 17.3 | ✗ |

conc-16 meets the strict tail-latency SLO (<22s) at full context; per-req TPS misses 30 (the full-context
topk over-scan residual); conc-32/64 are the structural decode-batch ceiling. Directional per DEC-3.

## Files Changed
- `runs/20260530_dsv32_loop6/ac5_fullctx/`: `ac5_fullctx_arrays.json` (exact raw arrays + means), 
  `ac5_fullctx_metrics_tool.py` (recompute-from-raw + mean-integrity + sidecar invariants + fail-closed),
  `ac5_fullctx_attribution.txt` (decode-component section filled), `ac5_fullctx_report.md` (verifier
  description updated), `meta_c32.json` + `meta_c64.json` (new sidecars).
- `.humanize/bitlesson.md` — `BL-20260530-durable-tracked-acceptance-evidence` updated (recompute from RAW
  not a stored derived metric; add mean-integrity for single-element tampers; validate all sidecars);
  goal-tracker (R21 row + task6 note); round-21 contract/summary (gitignored loop state).
- (No production code change this round — data-only evidence rebuild; R17 decode fix + R19 bench fail-closed
  fix stand.)

## Validation
- `ac5_fullctx_metrics_tool.py --verify` → PASS (recomputes P99 TTFT + per-req TPS p50 + means from raw
  committed arrays == stored headline; all 3 sidecars' operating point verified; no empty-latency rows).
- 6 temporary-copy tamper tests each exit 1 (incl. the single-element itl_sum mutation that leaked before the
  mean-integrity check, and the exact R20 leak analog). `git diff --check` clean; commit `991666b58` pushed
  to `jimmy`. GPUs free (data-only round; no server booted).

## Remaining Items
- conc-16 full-context per-req TPS (24.9 < 30): the residual DS-selection topk over-scan. conc-32/64
  structural (DS ≤ DSA; conc-64 unattainable even for DSA). **Gated AC-10.** Cross-node smoke (future-gated),
  DSA conc-64 TPS ~29.4 (queued) unchanged. No ABI-lock change; DS-fair AC-12 gate unchanged.

## Goal Tracker Update Request
### Requested Changes:
1. **Mark the AC-5 full-context EVIDENCE as acceptance-grade** (R9-standard fail-closed verifier + exact raw
   arrays + all sidecars + component breakdown), resolving Codex R20 blocking issue 1. AC-5 stays Active only
   for the two genuine open decisions below.
2. **Owner approval — AC-5 measurement methodology:** approve `num_prompts=64` steady-state (warmup120/
   window300) as the AC-5 methodology instead of the literal `NUM_PROMPTS=320`. The verified cold-flood
   BitLesson (`BL-20260530-cold-flood-not-steady-state-slo`) shows np320 cold-ramps (window) or full-drains
   the queue (fixed-count → P99 TTFT ≈ full 320-request drain, ~300s, misleading), while np64-window is the
   steady-state methodology that reproduced the DSA baseline (R11/R12) and AC-7. Without approval the literal
   np320 produces a methodologically-wrong number.
3. **Owner decision — conc-16 full-context TPS axis:** the research-grade full-context blocked-topk kernel
   (within-block K=2048 under CUDA-graph) to lift conc-16 from 24.9→~30, vs accepting the bounded-context
   op-point (closed-batch 30.3, 64K servability = separate full-context point AC-8/R16). conc-32/64 ≥30 is
   structurally unattainable (DS ≤ DSA) regardless.
### Justification:
This round delivered the evidence fix Codex required (the verifier is now genuinely fail-closed at the R9 bar,
demonstrated against 6 single-field tampers). The two remaining AC-5 items are decisions the owner must make
(methodology + the kernel-vs-rescope), not more measurement — surfacing them per Codex's instruction that
methodology needs explicit owner approval and the bounded-context target is not a rescope until the owner
changes it.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260530-durable-tracked-acceptance-evidence
Notes: Extended with the R20/R21 finding — a fail-closed verifier must recompute the published metric from the
RAW committed arrays, NOT re-check a stored DERIVED array (R20's derived `per_req_gen_tps` was tamperable to
100.0 and passed; R21 recomputes per-req TPS = output_len/itl_sum from raw). Also: a robust percentile (p50/
median) is insensitive to a single-element tamper, so add an aggregate-MEAN check (sensitive to every element)
alongside the percentile, and validate the operating point from ALL sidecars — demonstrate fail-closure with
single-element tampers, not just whole-array ones. Applied: BL-20260531-bench-empty-stream-failclosed (the
empty-latency class the verifier asserts), BL-20260530-clean-latency-attribution (per-conc queue_duration +
decode-component breakdown), BL-20260530-cold-flood-not-steady-state-slo (the np64 methodology justification).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-20-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-20-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-19-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-19-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-18-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-18-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-21-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
