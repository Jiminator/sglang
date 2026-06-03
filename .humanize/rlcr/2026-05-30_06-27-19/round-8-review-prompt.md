# Code Review - Round 8

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-8-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 8 Summary — exact-recomputable AC-5 evidence + reconciled attribution

## Mainline objective (round contract)
Fully close the AC-5 evidence + attribution so task6/AC-5 is verifiable. Codex's R7 review found the two R6 blockers only *partially* resolved: (1) the evidence was summary-only (not exactly recomputable), and (2) the attribution had contradictory row accounting (`valid=959 > 3x320=960` — false arithmetic) plus a wall-clock window split that mis-bucketed the log (306/337/316 ≠ 320/conc), with the stale R6 aggregate still in `client_slo_metrics.txt`. Data-correction round on the existing R6 hardware run (no re-run; local JSONLs + full server log present and used). No production code.

## What landed (commit bd09d1ca7, pushed to `jimmy`)
1. **Exact recomputation source** — `client_slo_int8/ac5_metrics_arrays.json` + `ac5_metrics_tool.py`. Per conc, the **exact per-request arrays**: `ttfts` (s), `tpots` (ms = `sum(itls[i])/(output_lens[i]−1)`, the sglang formula — reproduces stored `median_tpot_ms` exactly), `input_lens`, `output_lens`, plus errors-all-empty, each source JSONL's **SHA256**, and the percentile method (`numpy.percentile`). `python3 ac5_metrics_tool.py --verify` recomputes every TTFT/TPOT/TPS percentile **from the committed JSON alone** and asserts **recomputed == stored → PASS at all conc** (TTFT p50/p99 and ITL p50/p95/p99 match the JSONL bit-for-bit; TPOT p50/p99 exact). Replaces the summary-only addendum as the recomputation source.
2. **Rebuilt attribution** — `attribution_per_conc.txt`. From **benchmark rows only** (`output_len=512`), grouped per conc by **request-completion print-time** (the `[HH:MM:SS]` server-log prefix), split at the 2 largest gaps → **320 / 320 / 320**, reproducing benchmark.log's 320-completed/conc. **Full row reconciliation (exact arithmetic):** 967 parsed = **3 HEALTH_CHECK + 4 warmup (`output_len` 8/32) + 960 benchmark**; **5 invalid negative-`queue_duration` rows dropped (all conc-64) → 955 valid**; per-conc valid **320 / 320 / 315**. Per-conc queue p50/p95/p99 (10.5 / 22.3 / 99.4 s p99), tail-to-tail post-admission residual (2.3 / 3.2 / 11.8 s), measured-vs-inferred (forward_duration = completion-time, context-only). The false `959>960` and the mis-bucketing are gone; entry_time gap-clustering and the `T0+cumulative-durations` split (both tried in R7) are explicitly rejected with the reason (within-run waves rival inter-run gaps; T0 anchored on readiness mis-buckets).
3. **De-staled metrics** — `client_slo_metrics.txt`: the R6 all-conc aggregate (`N=959`) + `#running-req 19-20` line replaced with the corrected per-conc attribution + the `decode_batch_excerpt.txt` per-req-TPS figures (batch 16/~32/~38 → 17.7/11.5/9.7).
4. **Consistency** — `client_slo_report.md` + `ac5_evidence_addendum.txt` now point "recomputable" at the exact source and drop the `959`/warmup framing; the report's attribution section carries the corrected reconciliation.

## Result (verdict unchanged; now exactly recomputable + internally consistent)
DIRECTIONAL — accepted progress, explicitly NOT shippable (DEC-3). conc-16 meets strict `<22 s` (12.8 s); conc-32/64 TTFT and all-conc per-req TPS miss the strict SLO and remain the **open mainline blocker**. Every AC-5 number now recomputes from committed files; the attribution is reconciled to the exact row counts.

## Files Changed
- `runs/20260530_dsv32_loop6/client_slo_int8/`: `ac5_metrics_arrays.json` (new, exact arrays), `ac5_metrics_tool.py` (new, build/verify), `attribution_per_conc.txt` (rebuilt), `client_slo_metrics.txt` (de-staled), `ac5_evidence_addendum.txt` (pointer to exact source).
- `runs/20260530_dsv32_loop6/client_slo_report.md` (evidence bundle + corrected attribution reconciliation).
- `.humanize/bitlesson.md` (updated `clean-latency-attribution`), goal-tracker, round-8 contract/summary (gitignored loop state).

## Validation
- `ac5_metrics_tool.py --verify`: recomputed == stored, **PASS** at conc 16/32/64 (TTFT/TPOT/TPS from committed JSON alone).
- Attribution reconciliation reproduces benchmark.log (3×320 completed) and Codex's authoritative per-conc valid counts (320/320/315); print-time grouping verified against the log.
- All new files tracked (`git check-ignore` → none); `git diff --check` clean; no stale `959`/`>960`/`cumulative-durations`/`wall-clock-windows` strings remain in the tracked AC-5 files.
- No re-run, no production code change; the R6 run's `.meta.json` radix-on sidecars unchanged.

## Remaining Items
- **Open mainline blocker:** strict client SLO still fails (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). Characterized (throughput/decode-batch, not footprint).
- **Next round = hardware (AC-6):** DSA-default product proof (DSA-default boot meets SLO unchanged, allocates no DS table; DS opt-in toggles the compact int8 path), pairing the **AC-9** code edit (`within_budget` from real `usage.prompt_tokens`, fail-closed, DS-fair thresholds UNCHANGED) + its live rerun.
- Then **AC-7** (3-trial DS+DSA lifted-point re-sweep), **AC-8** (~70K servability probe), gated **AC-10**. No FlashMLA decode-assert changes (AC-3.3).

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260530-clean-latency-attribution
Notes: Sharpened BL-20260530-clean-latency-attribution with the R7→R8 fix. R7 disclosed the negative-row filtering but still (a) asserted a false `959>960`, (b) mis-grouped per conc using `T0+cumulative-durations` (gave 306/337/316 ≠ 320/conc), and (c) left the evidence summary-only. R8's additions to the lesson: CLASSIFY rows by a stable shape signature before counting (benchmark rows = `output_len=512`; warmup = `output_len` 8/32; health = HEALTH_CHECK) and reconcile valid-vs-nominal with EXACT arithmetic (not a "warmup makes valid>nominal" narrative — the truth was 960 benchmark, 5 invalid, 955 valid); group per conc by a RELIABLE key (request-completion print-time split at the largest gaps), explicitly NOT entry_time gaps (within-run waves rival inter-run gaps) and NOT a readiness-anchored T0; validate the grouping reproduces the benchmark's known per-conc completion count; and commit the EXACT per-request arrays (or checksum + recompute script) so a reviewer recomputes from committed files, not summary-vs-summary. No new lesson added — this is the same failure family, made precise across rounds. Applied existing BL-20260530-durable-tracked-acceptance-evidence (exact arrays as tracked .json + a committed --verify script; git check-ignore + git diff --check before claiming done) and the push-between-rounds preference (pushed bd09d1ca7 to jimmy).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-7-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-7-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-6-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-6-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-5-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-5-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-8-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
