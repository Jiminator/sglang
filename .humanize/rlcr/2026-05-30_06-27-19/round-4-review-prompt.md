# FULL GOAL ALIGNMENT CHECK - Round 4

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 4 Summary — AC-4 mem-fraction lift + no-OOM validation (int8 compact table)

## Mainline objective (round contract)
AC-4 — boot DS with the compact int8 table, sweep `MEM_FRACTION_STATIC=0.6→0.7→0.8`, record `max_total_num_tokens` rising + the full HBM budget at each point, and survive a sustained long `/generate` at the lifted fraction with **no generation-time OOM and no monotonic memory growth**. This is the spine's payoff — the footprint reduction exists to lift the no-OOM operating point.

## Result — PASS (lifted operating point = 0.7; commit `75e68053f`)
Real single-node TP=8 H200, `serve_double_sparsity.sh SIGNATURE_DTYPE=int8`, Loop-5 mask, Option B.

| `f` | int8 table | KV pool | `max_total_num_tokens` | post-graph headroom | result |
|---:|---:|---:|---:|---:|:--|
| 0.6 | 0.87 GB | 2.38 GB | 53056 | 38.34 GB | serves |
| 0.7 | 6.48 GB | 17.73 GB | **396096** | 17.56 GB | **serves + sustained-gen no-OOM** |
| 0.8 | 12.10 GB | 33.09 GB | 739200 (attempted) | — | **boot OOM (cuda-graph capture)** |

- `max_total_num_tokens` **rises** with `f` (53056 → 396096 → 739200). The int8 table at 0.7 is **6.48 GB vs fp16's 11.52 GB** → **17.56 GB** post-cuda-graph headroom vs fp16's 12.29 GB that gen-OOM'd in Loop-5.
- **No-generation-OOM validation at the lifted fraction (0.7):** a sustained stress — **32 concurrent** `/generate`, ~4096-ISL, 256 new tokens, 3 rounds + a ~30K-token long-context request — completed **97/97 OK, 0 failed**, **no generation-time OOM**. NVML over the run rose to the generation working set then **plateaued** (last sample == max; min-free 17.9→11.9 GB steady) — **no monotonic growth**. This directly refutes fp16's Loop-5 0.7 generation-OOM: same fraction, same workload, int8 survives.
- Full HBM budget (NVML per-GPU + torch avail + `/get_server_info` + log components: weights / KV / table+scales / cuda-graph pool / headroom) captured per fraction under `runs/20260530_dsv32_loop6/memfraction_sweep_int8/`.

## The 0.8 ceiling — honest, AC-2-framed
`f=0.8` **boot-OOMs during cuda-graph capture** (verbatim: `Capture cuda graph failed: ... Tried to allocate 146.00 MiB ... 132.12 MiB free`): the int8 table (12.10 GB) + the 739K-token KV pool (33 GB) leave only 22.68 GB pool-end headroom, and the fixed ~11.6 GB cuda-graph capture pool doesn't co-fit. This is a **boot-time** OOM, **not** the AC-4 generation-time negative test, and **not** "the table is still too big". Per the verified AC-2 budget the target is *admitted KV capacity*, not `f=0.8` as a number, and the plan calls 0.7 "acceptable as a more conservative first step": **0.7's `max_total=396096` exceeds the conc-64 admission target (~114K) by ≈3.5×**, so the admission goal is met with large margin and the page-level escalation is **not** triggered. Reaching 0.8 would require trimming the fixed Option-B cuda-graph batch set (a productionization pass, out of scope) and is unnecessary.

## Files changed
- `runs/20260530_dsv32_loop6/memfraction_sweep_int8.md` + `memfraction_sweep_int8/` (per-fraction NVML/`get_server_info`/log captures, the 0.7 NVML time series, boot excerpts) — commit `75e68053f`.
- `.humanize/bitlesson.md` (+1 lesson), goal-tracker, round-4 contract/summary (gitignored loop state).

## Validation
- 3 clean TP=8 boots (0.6/0.7/0.8) with `SIGNATURE_DTYPE=int8` confirmed (`token_label_table ... dtype=torch.int8`).
- 0.7 sustained generation: 97/97 OK, no gen-OOM, NVML plateau.
- 0.8 boot-OOM captured verbatim. GPUs freed after each boot (stale-`sglang::router` cleared via explicit-PID kill).

## Remaining items
**AC-4 done** (lifted operating point 0.7, no-OOM validated, full budget recorded). Nothing in AC-4 deferred.
- **AC-5 (next mainline):** full client-SLO benchmark — `NUM_PROMPTS=320`, conc 16/32/64, 4096 ISL / 512 OSL / ~55% cache, radix-on proven, at the lifted operating point (0.7, int8). `client_slo_report.md` with strict `<22.0` TTFT + `≥30 TPS/req`, the pre-declared trial-aggregation rule, and the **required measured admission-wait vs prefill-compute attribution**. Then AC-6 hardware opt-in/DSA-default, AC-7/AC-8/AC-9, gated AC-10.
- Note: at 0.7 DS now admits ~396K tokens (vs Loop-5's 53K at 0.6) — admission should be largely restored; AC-5 measures whether that actually moves P99 TTFT toward <22s and attributes any residual to prefill compute.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-int8-memfraction-ceiling-is-cudagraph-capture
Notes: Added a lesson capturing the AC-4 finding: after the footprint lever relieves the generation-OOM, the binding HBM constraint at high mem_fraction MOVES from generation headroom to the BOOT-time cuda-graph capture pool — so 0.8's OOM is a cuda-graph-capture boot failure, not "the table is too big". The rule: don't escalate the footprint lever (page-level) on such an OOM; check the admission target first (0.7 already over-admits conc-64 ~3.5×) and define the operating point as "highest fraction that boots AND survives a sustained generate", targeting admitted KV capacity not the mem-fraction number. Applied existing lessons as context: BL-20260528-dsv32-ds-serving-boot-chain (fp16 0.7 gen-OOM baseline) and BL-20260530-verify-hardware-before-deferring (8-GPU TP=8 confirmed before the serve).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-3-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-3-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-2-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-2-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-1-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-1-review-result.md


Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Goal Tracker Audit (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md and verify:

### 1.1 Acceptance Criteria Status
For EACH Acceptance Criterion in the IMMUTABLE SECTION:
| AC | Status | Evidence (if MET) | Blocker (if NOT MET) | Justification (if DEFERRED) |
|----|--------|-------------------|---------------------|----------------------------|
| AC-1 | MET / PARTIAL / NOT MET / DEFERRED | ... | ... | ... |
| ... | ... | ... | ... | ... |

### 1.2 Forgotten Items Detection
Compare the original plan (@development/loop6/refined_plan_v1.md) with the current goal-tracker:
- Are there tasks that are neither in "Active", "Completed", nor "Deferred"?
- Are there tasks marked "complete" in summaries but not verified?
- List any forgotten items found.

### 1.3 Deferred Items Audit
For each item in "Explicitly Deferred":
- Is the deferral justification still valid?
- Should it be un-deferred based on current progress?
- Does it contradict the Ultimate Goal?

### 1.4 Goal Completion Summary
```
Acceptance Criteria: X/Y met (Z deferred)
Active Tasks: N remaining
Estimated remaining rounds: ?
Critical blockers: [list if any]
```

## Part 2: Mainline Drift Audit (MANDATORY)

Determine whether the recent rounds are still serving the original plan:
- Is the current round's mainline objective clear and singular?
- Has Claude been advancing mainline ACs, or mostly clearing side issues?
- Which findings are true **blocking side issues** versus merely **queued side issues**?

Include a short drift summary:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
Blocking Side Issues: N
Queued Side Issues: N
```

The `Mainline Progress Verdict` line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

## Part 3: Implementation Review

- Conduct a deep critical review of the implementation
- Verify Claude's claims match reality
- Identify any gaps, bugs, or incomplete work
- Reference @docs for design documents

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

## Part 5: Progress Stagnation Check (MANDATORY for Full Alignment Rounds)

To implement the original plan at @development/loop6/refined_plan_v1.md, we have completed **5 iterations** (Round 0 to Round 4).

The project's `.humanize/rlcr/2026-05-30_06-27-19/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-05-30_06-27-19/round-3-review-result.md` (previous round)
- `@.humanize/rlcr/2026-05-30_06-27-19/round-2-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-05-30_06-27-19/round-3-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-4-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-4-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
