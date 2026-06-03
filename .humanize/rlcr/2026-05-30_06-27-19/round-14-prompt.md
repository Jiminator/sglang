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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-14-contract.md

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
# Round 13 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: ACs: 8/10 addressed, 6/10 met | Forgotten items: 0 | Unjustified deferrals: 1

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-13-prompt.md`, `round-13-contract.md`, `round-13-summary.md`, `goal-tracker.md`, Round 10-12 summaries/reviews, commit `5e6d3afb5`, the AC-7 artifacts, AC-5 attribution artifacts, and the AC-11 comparator implementation.

## Implementation Review

Round 13 did make real mainline progress: it produced a 3-trial DS+DSA lifted-point sweep, and the 18 sidecars consistently show the intended local operating points: DS is `double_sparsity`, `enable_double_sparsity=true`, `signature_dtype=int8`, radix-on, `mem_fraction_static=0.7`, `max_total_num_tokens=396096`; DSA is `native_nsa`, DS disabled, radix-on, `mem_fraction_static=0.85`, `max_total_num_tokens=910784`. The sidecars also consistently record `num_prompts=64`, warmup 120, window 600, and the same pre-R13 code commit SHA (`f9bc51b13`) for both sides.

I cannot verify AC-7 as complete because the published evidence bundle is internally inconsistent and does not meet its own recomputability/provenance claims.

## Mainline Gaps

1. **AC-7's claimed recomputable metrics source contradicts the comparator on the headline achieved-concurrency result.**

   `ac11_resweep.md` reports the key disclosed DS conc-64 achieved concurrency as `46.983` / `73%` (`runs/20260530_dsv32_loop6/ac7_resweep/ac11_resweep.md:13-17`). But `ac7_resweep_metrics.json`, described as the recomputable per-trial source, records every DS conc-64 trial as `"achieved": 64` (`runs/20260530_dsv32_loop6/ac7_resweep/ac7_resweep_metrics.json:152-183`). A direct median over that JSON gives DS achieved concurrency 64, not 46.983. The same file says it contains "Per-trial metrics + medians + source JSONL SHA256, so the comparator numbers recompute" (`ac7_resweep_metrics.json:2`), but the committed JSON cannot recompute the comparator's most important AC-7 admission number.

   Required fix: rebuild the AC-7 evidence bundle from the raw JSONLs or rerun the sweep if the raw files are gone. The committed source must include the exact per-trial fields used by `benchmark_compare.py --ac11`, especially `achieved_concurrency`, `output_tps_p50`, `ttft_p99_s`, `duration`, completed/errors, workload shape, and the exact median inputs. Add a fail-closed verifier that recomputes `ac11_resweep.md` numbers from committed data and exits nonzero on mismatch.

2. **AC-7 is not exact-recomputable by the evidence standard already enforced in this loop.**

   The raw AC-7 JSONLs are not present in the repository, and `ac7_resweep_metrics.json` stores rounded summary values plus 16-hex digest prefixes, not full source SHA256s (`ac7_resweep_metrics.json:22`, `:92`, `:162`, etc.). Prior AC-5/AC-6 reviews required committed exact arrays or a fail-closed verifier when raw JSONLs are gitignored. AC-7 now relies on hand-curated summary data, which is exactly the class of evidence gap the loop already fixed in earlier rounds.

   Required fix: either commit redacted exact per-request arrays sufficient to recompute the comparator metrics, or commit a compact exact verifier bundle equivalent to AC-5/AC-6. Store full 64-character SHA256s for each raw JSONL and include the comparator command/stdout provenance used to generate the report.

3. **The comparator's failing-row profiling obligation is not actually discharged for the AC-7 methodology.**

   `ac11_resweep.md` says every failing row requires "a captured profile (`development/profile_ds.sh` or equivalent) before the comparator row can be published" (`ac11_resweep.md:21-27`). `ac11_analysis.md` claims the obligation is discharged by AC-5 attribution at an "identical workload/operating point" (`ac11_analysis.md:43-48`). That is not identical methodology: AC-5 was a single `NUM_PROMPTS=320`, `WARMUP_SECONDS=0`, `MEASUREMENT_WINDOW_S=60` directional run (`client_slo_report.md:13-14`), while AC-7 is 3 trials with `num_prompts=64`, 120s warmup, and a 600s window. AC-5 is useful context, but it is not a captured AC-7 profile/equivalent for the 120/600 sweep.

   Required fix: capture AC-7-methodology profiling or equivalent attribution for the failing rows. The minimal acceptable artifact is per-concurrency DS request-time stats and decode-batch evidence from a 64-prompt, 120/600 run at DS int8/mem0.7/radix-on, reconciled to the AC-7 completed counts, plus a short analysis tying the failing TPS/TTFT rows to the measured data.

4. **The original Loop-6 work remains incomplete.**

   AC-5 is still directional-only and fails the strict client SLO; AC-8 is still pending; AC-10 remains gated behind full Tier-1 completion. Round 13 therefore cannot output `COMPLETE`.

## Blocking Side Issues

1. **The script-level cross-node wrapper smoke is still not resolved.**

   The local node0 sequential pivot is acceptable for avoiding the host-targeting bug in the R13 local artifact, but it does not satisfy the Round 13 contract's success criterion that the wrapper smoke pass and be recorded (`round-13-contract.md:16-21`, `:32-35`). It also does not prove `benchmark.sh` / `benchmark_baseline.sh` target a remote host correctly; a localhost banner is not the cross-node smoke. Do not publish any future cross-node scripted benchmark artifact until `HOST=<remote>` wrapper stdout and the matching sidecar prove the same host.

2. **Strict DS client SLO still blocks the Ultimate Goal.**

   The verified AC-5 artifact remains a directional result, not a shippable pass: conc-32/64 TTFT still miss `<22 s`, and per-request TPS is below `30 TPS/req` at every concurrency. This should remain a mainline blocker after AC-7/AC-8 hardening is repaired.

## Queued Side Issues

1. **DSA-default conc-64 TPS remains below the client threshold.**

   This is still queued under the R12 user decision because it reproduces the pre-existing DSA baseline and is not DS-introduced.

## Goal Alignment Check

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified earlier. |
| AC-2 | MET | Feasibility budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, launcher, sidecar consumers, NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted 0.7 operating point, HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | Evidence/attribution verified; strict DS SLO still fails. |
| AC-6 | MET | Verified in R12 under the user-approved non-regression/opt-in semantics. |
| AC-7 | PARTIAL | 3-trial local sweep exists, but evidence bundle contradicts itself and profiling/provenance is incomplete. |
| AC-8 | NOT MET | Lifted ~70K-token servability probe pending. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. |
| AC-10 | NOT MET | Correctly gated behind full Tier-1 completion. |

Forgotten items: none. Every original plan task is represented in Active, Completed, or the gated AC-10 path. Deferred items: one unjustified round-contract deferral remains: the script-level cross-node wrapper smoke was marked N/A/resolved without the requested remote wrapper artifact.

## Required Implementation Plan

1. Repair AC-7 before moving on. Recover the raw AC-7 JSONLs from the run host or rerun the same local sequential sweep: DS int8/mem0.7/radix-on and DSA-default/mem0.85/radix-on, conc 16/32/64, `num_prompts=64`, 120s warmup, 600s window, 3 trials.
2. Rebuild `ac7_resweep_metrics.json` as an exact source of truth: include per-trial achieved concurrency from the JSONL `concurrency` field, exact gate metrics, completed/errors, duration, workload shape, full SHA256s, and computed medians. Do not round away values needed to reproduce `ac11_resweep.md`.
3. Add `ac7_metrics_tool.py --verify` or an equivalent fail-closed command that recomputes the comparator rows and validates the Markdown/JSON outputs from committed data alone. Include a tamper-resistant sanity check like the AC-5/AC-6 verifiers.
4. Discharge the AC-11 profiling obligation under the AC-7 methodology: capture DS request-time stats plus decode-batch evidence for the 120/600, 64-prompt sweep, reconcile it to completed counts, and update `ac11_analysis.md` to cite that artifact. AC-5 WARMUP=0 attribution may remain background context, not the AC-7 proof.
5. Keep the cross-node wrapper smoke partial unless it is actually run. If any future artifact is cross-node, first run the tiny `HOST=<remote>` wrapper smoke and commit the readiness banner plus sidecar proving same-host targeting.
6. After AC-7 is verified, complete AC-8: run the lifted ~70K-token `/generate` probe and record HTTP 200 with capacity/no instability, or a characterized ceiling with the server-side reason.
7. Return to the AC-5 strict SLO blocker with the repaired AC-7 data in hand. The next remediation must target the measured scheduling/decode bottleneck while keeping the client SLO and Option B constraints fixed.
8. Start AC-10 only after AC-3 through AC-9 are verified.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 13 Review.
- Added an R13-review plan-evolution row rejecting AC-7 verification for now.
- Kept task8/AC-7 Active as partial rather than verified.
- Changed the cross-node host issue from resolved to partial.
- Added a blocking AC-7 evidence-bundle issue for the metrics/comparator inconsistency and missing exact verifier/profiling.

## Validation Performed

- `git log --oneline -30`
- `git show --stat --oneline 5e6d3afb5`
- Inspected `round-13-contract.md`, `round-13-summary.md`, `goal-tracker.md`, and Round 10-12 summaries/reviews.
- Inspected `ac11_resweep.md`, `ac11_analysis.md`, `ac7_resweep_metrics.json`, and all AC-7 `.meta.json` sidecars.
- Used `jq` to verify AC-7 sidecar operating points: DS int8/radix-on/mem0.7, DSA-default/radix-on/mem0.85, same commit/workload fields.
- Recomputed medians from `ac7_resweep_metrics.json`; this exposed the DS conc-64 achieved-concurrency mismatch.
- Inspected `development/benchmark_compare.py` for AC-11 sidecar validation and failing-row profiling obligation.
- Inspected AC-5 attribution artifacts to compare methodology against AC-7.
- `git diff --check`

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-14-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-14-summary.md

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
