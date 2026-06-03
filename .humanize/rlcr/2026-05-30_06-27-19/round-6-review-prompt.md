# Code Review - Round 6

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-6-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 6 Summary — AC-5 client-SLO directional result (the loop's headline)

## Mainline objective (round contract)
AC-5 — run the full client workload against DS with the compact int8 table at the lifted 0.7 operating point, **radix-on proven**, and write `client_slo_report.md` with the absolute P99 TTFT + per-request TPS vs the strict SLO, a **measured admission-wait vs prefill-compute attribution**, and a directional-improvement statement vs Loop-5. Graded directional (DEC-3): accepted progress, **not** a shippable pass.

## Blocking prereq landed: int8 radix fixture (commit `8883848e9`)
`serve_double_sparsity.sh` is radix-off by default; radix-on needs a fixture artifact, and my R2 `signature_dtype` fingerprint makes the Loop-5 fp16 state fail closed for int8. **Regenerated** it: booted with `SIGNATURE_DTYPE=int8 SGLANG_DS_RADIX_OVERRIDE=1 SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, ran both M3-B fixtures — **label-capture PASSED** (cold==warm DS label SHAs bit-equal, confirming the R2 scale-aware radix capture works in the real fixture) and **fp8-scale-stability PASSED** — and `write_radix_fixture_state` → `ds_radix_fixture_state_int8.json` (fingerprint includes `signature_dtype: int8`). The benchmark server then booted **radix-on authorized** by that artifact (validator: "fixture recorded as PASSED ... artifact_sha256=f3b67943"; `disable_radix_cache=false`), proven in every `.meta.json` sidecar.

## Result — DIRECTIONAL: accepted progress, NOT shippable (DEC-3)
DS int8/0.7, radix-on, `--enable-request-time-stats-logging`, gsp 4096 ISL (median input_len ≈ 4280) / 512 OSL, conc 16/32/64, 320 prompts, **1 directional trial** (`WARMUP=0/WINDOW=60` → one full 320-prompt epoch per conc; disclosed).

| conc | achieved (vs L5) | **P99 TTFT** | `<22`? | L5 TTFT | **per-req TPS** | `≥30`? | L5 TPS |
|---:|---:|---:|:--:|---:|---:|:--:|---:|
| 16 | **16.0** / 14.5 | **12.8 s** | ✅ | 57.7 | 17.6 | ❌ | 34.0 |
| 32 | **32.0** / 24.6 | 25.5 s | ❌ | 132.9 | 11.5 | ❌ | 33.9 |
| 64 | **60.1** / 35.7 | 111.2 s | ❌ | 292.0 | 9.3 | ❌ | 33.9 |

**The spine is validated:** admission restored (achieved ≈ nominal vs Loop-5's queue-starved 14.5/24.6/35.7); **P99 TTFT collapsed 4.5×/5.2×/2.6×**; **conc 16 MEETS the strict `< 22 s`** (12.8 s). The footprint→pool→admission→TTFT chain works on the real client workload.

**Attribution (required, measured):** from `ReqTimeStats` (`queue_duration` vs `forward_duration`) + the per-conc TTFT floor: prefill-compute floor ≈ **1.3 s** (an un-queued request prefills 4096 ISL in ~1.3 s); the **residual TTFT is queue/throughput-bound** (`queue_duration` p99 ≈ 98.5 s at the high-load tail), and it is **NOT KV-pool-admission-bound** (64×4608 = 295K < the 396K pool). So conc-32/64's residual is throughput contention from the 320-request flood → the follow-up is **chunked-prefill / scheduling**, not more footprint (the plan's anticipated "prefill-bound at conc 64" risk, confirmed with data).

**NEW FINDING — the TPS/TTFT tradeoff:** per-request TPS is **below 30 at every conc** (17.6/11.5/9.3), below Loop-5's 34. Cause: restoring admission grows the decode batch (Loop-5's 53K pool decoded only ~2–3 of these 4608-token requests at conc 64 → 34 tok/s/req; the 396K pool decodes ~19–20 → server log gen ~277 tok/s ⇒ ~14 tok/s/req). The loop's premise that "DS already beats 30 TPS, only TTFT is the problem" held **only** at the queue-starved operating point; once admission is restored the `≥30 TPS/req` SLO is in genuine tension with high concurrency on this 671B MoE. Captured as `BL-20260530-admission-restore-tps-tradeoff`.

**Verdict:** DIRECTIONAL accepted progress (spine validated, TTFT collapsed, conc-16 SLO met) — **explicitly not shippable** (strict SLO not met at every conc; honestly recorded with attribution per DEC-3). Two surfaced downstream blockers, with data, **neither a footprint problem**: (1) conc-32/64 TTFT = prefill/throughput-bound (chunked-prefill follow-up); (2) per-request TPS-vs-admission tradeoff (decode optimization / operating-point choice).

## Files changed
- `runs/20260530_dsv32_loop6/`: `ds_radix_fixture_state_int8.json`, `client_slo_report.md`, `client_slo_int8/` (3× `.meta.json` sidecars [radix-on proof], `client_slo_metrics.txt`, `reqtimestats_excerpt.txt`). Raw 4 MB `.jsonl` are gitignored (`*.jsonl`); metrics embedded as tracked `.txt` (per BL-20260530-durable-tracked-acceptance-evidence). commit `8883848e9`.
- `.humanize/bitlesson.md` (+1 lesson), goal-tracker, round-6 contract/summary (gitignored loop state).

## Validation
- int8 radix fixture: both M3-B fixtures PASS; state written with `signature_dtype: int8`; server booted radix-on authorized (no override), proven in sidecars.
- Benchmark: 3 conc, 320/320 completed each; `git diff --check` clean; GPUs freed after.

## Remaining items
- **AC-6 (next, partial→hardware):** the DSA-default product property on hardware (DSA-default boot meets SLO unchanged, allocates **no** DS table; DS opt-in toggles the compact path). Then:
- **AC-7** (AC-11 DS+DSA 3-trial re-sweep at the lifted point, radix-on both), **AC-8** (~70K-token 64K servability probe at 0.7), **AC-9** (within-budget harness edit to real `usage.prompt_tokens` + live re-run), then gated **AC-10**.
- Carry forward to the report/roadmap: the conc-32/64 chunked-prefill follow-up and the per-request-TPS-vs-admission tradeoff are downstream items (not Loop-6 footprint scope).

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-admission-restore-tps-tradeoff
Notes: Added a lesson capturing the AC-5 finding: a footprint/pool lever that restores admission to fix TTFT grows the decode batch, which lowers per-request decode TPS (every in-flight request advances one token per forward step; step time grows with batch). The ">=30 TPS/req" measured at the queue-starved Loop-5 point (34 tok/s, ~2-3 decoding) does NOT hold at the restored-admission point (~14 tok/s, ~19-20 decoding). The rule: measure per-request TPS at the actual target operating point, and always pair the TTFT win with the per-request TPS + the queue-vs-forward attribution so a TTFT improvement that hides a TPS regression is caught. Applied existing lessons: BL-20260530-durable-tracked-acceptance-evidence (raw .jsonl gitignored → embedded metrics as tracked .txt + verified with git check-ignore) and the router-kill gotcha for the server re-boots.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-5-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-5-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-4-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-4-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-3-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-3-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-6-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
