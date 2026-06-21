# Code Review - Round 1

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop11b/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/round-1-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 1 Summary

## Objective
Make the M-B M4 verdict PUBLISHABLE and close out honestly. Round 0 delivered M-A + a directional verdict,
but Codex rejected the completion: the `--ac11` comparator REFUSED, same-memory was deferred, conc-64 was
admission-capped, the benchmark emitted no reuse/no-op evidence, AC-4 was not the spec'd tax guard, and AC-8
ledger/evidence/push were incomplete. All five gaps are now closed.

## How each Codex Round-0 gap was resolved
1. **AC-9 comparator refused + same-memory deferred →** DS + DSA re-run from ONE frozen HEAD (commit_sha
   99ac584ac). BOTH op-points now `--ac11` comparator-ACCEPTED: `ac11_production_envelope` (DS0.8/DSA0.85)
   rc=3 and `ac11_same_memory` (DS0.8/DSA0.8) rc=3 — honest absolute DS FAIL@64, directional PASS. (rc=3 =
   absolute SLO fail, the published verdict; rc=2 would be refusal — neither refused.)
2. **AC-2/3 admission-capped conc-64 →** the clean re-run reached nominal: running-req **peak 63** (≥61).
   Time-averaged achieved 58.9 is a real DS smaller-KV-pool effect at mem 0.8 (queue-bound TTFT), not a
   measurement failure — stated as such.
3. **AC-4 not the spec'd guard →** replaced sweep-derived TPOT with a DEDICATED controlled fixed-bs probe
   (distinct-prefix, GRAPH, mem 0.8 both sides): bs64 DS 39.83ms/DSA 37.70ms = 1.056; bs30 31.85/30.14 =
   1.057 — both ≤1.10 PASS; bs30 31 850µs ≪ 380 000. Loop-10 per-step parity held.
4. **AC-5/9 missing reuse/no-op evidence →** B1 extended `bench_serving` to emit per-request `cached_tokens`
   + DS no-op counters and added fail-closed `trial_evidence.py`. Every trial shows ~54% measured prefix
   reuse. No-op proven via direct evidence (0 dense_fallback across all 6 trials; top_k 2048 < 4096 ctx;
   4303 DS decode batches). The per-request DS meta_info aggregate is unwired for GLM (`Glm4MoeAttention`/
   dsa-backend never reaches DeepseekV2's `_publish_ds_request_summary`) — documented gap + recommended
   backend-side fix (`ac5_no_op_evidence.md`).
5. **AC-8 ledger/evidence/push →** results.md + queue.md regenerated to the R1 state; the stale `a4be98c4`
   capacity claim fixed to ld32 `35155ac4` (504640 reconfirmed); evidence committed as lossless `.meta.json`
   + `.evidence.json` sidecars + comparator md/json + `EVIDENCE_SHA256.txt` content hashes (raw 248MB jsonls
   gitignored); push recorded as pending owner direction (origin = public sgl-project upstream).

## The verdict (publishable)
Table-free DS on GLM-5.1-FP8 MEETS the client SLO (decode-TPS p50 ≥30, P99 TTFT <22s) at conc 16 (40.73 /
1.59s) and 32 (34.13 / 2.99s), and FAILS at 64 (26.98 <30, 25.08s ≥22). Native DSA ALSO fails @64. DS is
competitive-to-better than DSA at both op-points (TPS ratio 0.98–1.03, TTFT ratio 0.44–0.76). A throughput
FAIL is the honest, complete deliverable per the plan. Headline: `runs/20260616_mb/R1_HEADLINE_VERDICT.md`.

## Significant finding (real bug found + fixed)
The "clean re-run" surfaced a genuine DS server crash: the per-request error-abort path called
`req.check_finished()`, renamed upstream (#25725) to `update_finish_state()` — a semantic merge conflict that
crashed the whole TP=8 server when the DS selector sanitized a row. Fixed (99ac584ac). It escaped detection
because the abort path only runs on a selector error, which representative load never triggered; the
100%-identical-prefix tax burst (unrepresentative) was the first thing to trip it. Full analysis +
why-it-escaped in `runs/20260616_mb/R1_DS_CRASH_FINDING.md`.

## Files changed (R1)
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` — check_finished→update_finish_state crash-fix.
- `python/sglang/bench_serving.py` + `runs/20260616_mb/trial_evidence.py` — B1 reuse + no-op evidence (committed 9af9d7835).
- `runs/20260616_mb/{ds_only,dsa_tax,mb_v2}.sh` — corrected sweep-first + distinct-prefix runners.
- `runs/20260616_mb/{R1_HEADLINE_VERDICT,R1_DS_CRASH_FINDING,ac5_no_op_evidence}.md`, `results_v2/ac11_*.{md,json}`, EVIDENCE_SHA256.txt.
- `development/{results,queue}.md`, `runs/20260616_ma/capacity_ds_evidence.md` (a4be98c4→35155ac4).

## Validation
- Both comparators accepted (rc=3) at matched commit_sha; verdict reproduces Round 0.
- DS verdict sweep: 0 selector errors, 0 crashes, peak 63, ~54% reuse/trial.
- AC-4 dedicated probe both bs PASS. AC-5 no-op proven by direct evidence.

## Remaining Items
- Push: pending explicit owner direction (origin = public upstream). Recorded, not skipped.
- Follow-up (out of scope): wire per-request DS summary backend-side so GLM populates meta_info.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260617-ds-abort-path-merge-drift
- Notes: A feature-branch error-handling path that calls a method renamed by an upstream merge is a semantic
  merge conflict invisible to git and to happy-path tests — it only fires on the error path. When a rename
  lands via merge, grep the feature branch's NEW call sites for the old name. Also: a 100%-identical-prefix
  load burst is NOT representative of production reuse and can trip reuse-path edges the real workload never hits.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
5b00026a2 [loop11b] R0: GATE A PASS (radix recall on-vs-off equivalence, ld32 mask)
9423fac83 [loop11b] R0: GATE B PASS (cross-rank identity + no-dense-fallback, ld32 mask)
d992ce547 [loop11b] R0: GATE C PASS + MINT radix fixture (DEC-1 content-hash, ld32 mask)
bd7300b57 [loop11b] R0: no-override boot AUTHORIZED (AC-0.2/0.3 live, ld32 + DEC-1 fixture)
ac8c32d4f [loop11b] R0: DEC-1 path portability AUTHORIZED live (altpath) + DSA AC-7 probe
104fdd359 [loop11b] R0: M-A COMPLETE — op-point re-established (task5 + task6 done)
e802e1e1c [loop11b] R0: M-B AC-4 tax guard driver (bench_one_batch DS-vs-DSA bs64/bs30)
787e1d3eb [loop11b] R0: AC-4 tax guard folds into the sweep (bench_one_batch unsuitable)
d68a73cda [loop11b] R0: comparator tweaks (DEC-4 floor->2, DEC-6 ratio report-only) + sweep harness
72cb24751 [loop11b] R0: locked sweep RUNNING (production-envelope); ledger checkpoint
d625d7715 [loop11b] R0: locked sweep verdict — DS meets SLO @ conc16/32, FAILS @ conc64
94313249e [loop11b] R0: task10 (part) — DSA op-point caps + reconcile loop8 throughput warning
be71d4fc3 [loop11b] R0: task10 production UX pass (Cat-A/B docs + runbook; no ABI)
c6e3e943e [loop11b] R0: task10 DONE (UX pass); checkpoint — awaiting DSA matched re-run
d672d962f [loop11b] R0: matched-op-point verdict (task7/8/9 done) — DS meets SLO to conc32
425cdbcef [loop11b] R0: close-out — regenerate results.md (M-A+M-B+M-C complete)
65997cb4c [loop11b] R0: close-out evidence preflight — residual probe evidence + ignore .pt dumps
5df030348 [loop11b] R0: task11 close-out complete — all 11 tasks done; queue finalized
9af9d7835 [loop11b] R1: bench_serving emits prefix-reuse + DS no-op evidence (AC-5/AC-9)
8cde27faa [loop11b] R1: clean M-B re-run orchestrator (both op-points, tax probe, evidence)
73338e539 [loop11b] R1: fix mb_v2 tax_probe local-var bug; task10 serve-script de-plan
4ceba0ead [loop11b] R1: queue checkpoint — bench evidence + cleanup done, mb_v2 clean re-run running
86ddf6faf [loop11b] R1: fix stale a4be98c4 capacity claim (Codex gap 5) — note ld32 504640 reconfirm
1a29be00d [loop11b] R1: fix DS error-abort crash — check_finished was renamed upstream (#25725)
99ac584ac [loop11b] R1: document DS crash finding + selector reuse-edge; mb_v2 emits selector-error count
8fbe848ed [loop11b] R1: M-B verdict re-established clean — comparators ACCEPT both op-points
811c40420 [loop11b] R1: AC-5 no-op proof (dense_fallback=0 + structural sparsity) + GLM meta_info gap doc
9d2c4253d [loop11b] R1: headline M-B verdict + AC-4 dedicated per-step tax (both PASS)
f1b90c797 [loop11b] R1: AC-8 close-out — results.md + queue.md regenerated to the R1 publishable state
8e4407822 [loop11b] R1: complete evidence package — DSA server_info + crash-probe txt + crash-log hashes
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-16_11-53-55/round-0-summary.md
- @.humanize/rlcr/2026-06-16_11-53-55/round-0-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/round-1-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
