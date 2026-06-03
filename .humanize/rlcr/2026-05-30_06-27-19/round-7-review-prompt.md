# Code Review - Round 7

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-7-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 7 Summary — AC-5 durable evidence + corrected per-conc attribution

## Mainline objective (round contract)
Close the two AC-5 evidence/attribution blockers from Codex's R6 review **without a re-run** (the R6 client run is real; the local benchmark JSONLs + the full server request-time-stat log are present). This is an evidence/correction round — no production code, the directional verdict is unchanged; it only makes the AC-5 result durable and the attribution honest.

## What landed (commit 51dd009b8)
Three **tracked** files under `runs/20260530_dsv32_loop6/client_slo_int8/`, all recomputable without the gitignored `*.jsonl`:

1. **`ac5_evidence_addendum.txt`** — per conc (16/32/64): completed=320, **errors=0 (all-empty proof)**, achieved concurrency, duration, ISL distribution (min/p50/p99/max ≈ 4274–4295, nominal 4096), OSL=512, **TTFT** min/p50/p90/p99/max, **TPOT** + per-req TPS (1000/median_TPOT = 17.6/11.5/9.3), **ITL**, output throughput, and the radix-on sidecar proof (`disable_radix_cache=False`, `mem_fraction_static=0.7`, `max_total_num_tokens=396096`). Every number in `client_slo_report.md` now recomputes from this file.

2. **`attribution_per_conc.txt`** — reprocessed the **full** server log: 967 rows parsed; **5 invalid `queue_duration<0` rows + 3 HEALTH_CHECK probes filtered with a disclosed drop policy → 959 valid** (the >960 nominal is per-conc warmup requests, explained). Per-conc bucketing by wall-clock run windows (`T0 + cumulative measured durations 829.6/692.7/713.1 s`, since the 3 runs are contiguous with no idle gap to cluster on). **Honest measured-vs-inferred:**
   - MEASURED: `queue_duration` (admission wait) p99 = **10.5 / 22.3 / 99.4 s**; client TTFT p99 = 12.8 / 25.5 / 111.2 s; min TTFT ≈ 1.3 s = uncontended prefill floor.
   - INFERRED: post-admission residual = `TTFT_p99 − queue_p99` = **2.2 / 3.2 / 11.8 s** (prefill + chunked-prefill/decode interleave, NOT pure prefill). Tail-to-tail (p99−p99), not p50−p50 across two distributions whose rows aren't the same request.
   - `forward_duration` (completion-time = prefill + all 512 decode steps) is reported **context-only — never used as a first-token prefill term** (the R6 misuse, corrected).
   - Conclusion: P99 TTFT is admission-wait-dominated at every conc (queue p99 ≫ residual); the queue term grows 10.5→22.3→99.4 s while the prefill floor stays ~1.3 s. Still NOT KV-pool-bound (64×4608=295K < 396K pool) → throughput contention.

3. **`decode_batch_excerpt.txt`** — the TPS root cause, tracked + quantified: steady-state decode batch is **16 / ~32 / ~38** (this **corrects the R6 summary's "#running-req 19-20"** figure), aggregate gen ~270–370 tok/s, so per-req decode TPS = gen/`#running-req` = **17.7 / 11.5 / 9.7 tok/s** — reproducing the client p50 TPS (17.6 / 11.5 / 9.3) almost exactly.

**`client_slo_report.md`** updated: softened "spine validated" → **"directional characterization, not yet validated"**; rewrote the attribution section to the corrected measured-vs-inferred framing; fixed the prefill-floor and decode-batch figures; kept the **strict-SLO miss explicit** (conc-32/64 TTFT 25.5/111.2 s > 22 s; per-req TPS < 30 at every conc) as a **live mainline blocker**; references all three addenda.

## Result (unchanged verdict, now durable + honest)
DIRECTIONAL — accepted progress, explicitly NOT shippable (DEC-3). conc-16 meets strict `<22 s` (12.8 s); conc-32/64 TTFT and all-conc per-req TPS miss the strict SLO and remain an open mainline blocker. The footprint→admission→TTFT spine is *characterized* (not yet validated) by clean, recomputable per-conc evidence; the residual is throughput/decode-batch, not footprint.

## Files Changed
- `runs/20260530_dsv32_loop6/client_slo_int8/`: `ac5_evidence_addendum.txt`, `attribution_per_conc.txt`, `decode_batch_excerpt.txt` (all new, tracked).
- `runs/20260530_dsv32_loop6/client_slo_report.md` (directional-characterization wording + corrected attribution).
- `.humanize/bitlesson.md` (+1 add, 1 update), goal-tracker, round-7 contract/summary (gitignored loop state).

## Validation
- All three evidence files tracked (`git check-ignore` → none ignored); `git diff --check` clean (stripped a trailing EOF blank line); commit 51dd009b8 pushed to `jimmy`.
- Numbers cross-check: addendum per-req TPS (17.6/11.5/9.3, from median TPOT) ≈ decode-batch gen/#running-req (17.7/11.5/9.7); attribution queue p99 (10.5/22.3/99.4) + residual (2.2/3.2/11.8) ≈ client TTFT p99 (12.8/25.5/111.2).
- No re-run, no production code change; the R6 run's `.meta.json` radix-on sidecars are unchanged.

## Remaining Items
- **Open mainline blocker:** strict client SLO still fails (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). Characterized (throughput/decode-batch, not footprint); to be solved or characterized at the operating-point level before any strict/shippable claim.
- **AC-6 (task7, next):** DSA-default product property on hardware (DSA-default boot meets SLO unchanged, allocates no DS table; DS opt-in toggles the compact path).
- **AC-7** (3-trial DS+DSA re-sweep at the lifted point), **AC-8** (~70K-token 64K servability probe), **AC-9** (within-budget harness edit from real `usage.prompt_tokens` — DS-fair thresholds UNCHANGED), then gated **AC-10**. No FlashMLA decode-assert changes (AC-3.3).

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-clean-latency-attribution, BL-20260530-admission-restore-tps-tradeoff
Notes: Added BL-20260530-clean-latency-attribution capturing the R6->R7 cross-round fix: when attributing a tail latency metric from a server request-time-stat log under continuous batching — parse the full log and disclose total-vs-valid + a filtering policy for impossible rows (queue_duration<0, HEALTH_CHECK) and reconcile the count gap (warmup); state measured (queue_duration, client TTFT, min-TTFT prefill floor) vs inferred (post-admission residual = TTFT_p99-queue_p99, which includes chunked-prefill/decode interleave, not pure prefill); never use a completion-time counter (forward_duration) as a first-token term; compare tail-to-tail (p99-p99) not p50-p50 across two distributions; bucket per-conc by wall-clock run windows when runs are contiguous; attach the decode-batch excerpt if citing batch growth as the TPS root cause. Updated BL-20260530-admission-restore-tps-tradeoff's Validation Evidence to the corrected decode-batch figures (steady-state batch 16/~32/~38, per-req 17.7/11.5/9.7 tok/s; the R6 "#running-req 19-20 -> ~14 tok/s" was imprecise). Applied existing BL-20260530-durable-tracked-acceptance-evidence (embed metrics as tracked .txt, not gitignored .jsonl/.csv; git check-ignore + git diff --check before claiming done) and the push-between-rounds preference (pushed 51dd009b8 to jimmy).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-6-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-6-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-5-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-5-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-4-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-4-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-7-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
