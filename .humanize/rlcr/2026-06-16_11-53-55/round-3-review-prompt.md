# Code Review - Round 3

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop11b/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/round-3-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 3 Summary

## Objective
Fix the two blocking gaps from Codex's R2 review: **AC-5** (`total_tokens_mean` was numerically wrong) and
**AC-8** (stale `queue.md` + unresolved push). AC-5 and the AC-8 ledger/evidence work are complete; the push
is now owner-authorized to a fork but blocked on a working GitHub credential in this environment (see below).

## Mainline gap 1 — AC-5: `total_tokens` metric was mislabeled (FIXED)
Codex found it and I verified on the committed R2 artifact: the DS publishers emit
`sparsity_rate = 1 - selected/total` (pruned fraction), but `bench_serving` derived
`total = selected / sparsity_rate` (assumes the KEPT fraction) → reported `total_tokens_mean=3588.7` vs the
true `selected/(1-sparsity_rate)=4770.3`. `trial_evidence.py` PASSed only because `2048 < 3588` still held.

Fix (`8df44a59c`, Codex's plan):
- Added explicit `total_tokens` to `DoubleSparsityRequestStats` + `meta_info_for_request`; BOTH publishers
  (GLM `dsa_backend.maybe_publish_ds_request_summary` + DeepseekV2) set it from the host `seq_len`.
- `bench_serving` captures + aggregates `total_tokens` DIRECTLY (and emits a per-request `total_tokens`
  array), with a backward-compat fallback `selected/(1-sparsity_rate)` only when `total_tokens` is absent and
  `0 ≤ sparsity_rate < 1`.
- `trial_evidence.py` STRENGTHENED: refuses when the reported aggregate disagrees with the per-request
  `selected_tokens`/`total_tokens` arrays, or any row violates `sparsity_rate == 1 - selected/total`.
- 2 unit tests updated + green. trial_evidence proven to REFUSE the exact R2 mislabel (3588.7 vs 4774) and
  PASS a correct record.
Validated end-to-end: smoke + full re-run `results_r3/` → `total_tokens_mean ≈ 4765` (true seq-len), all 6
DS trials `trial_evidence.py` PASS with the consistency gate. Verdict CONCLUSION unchanged (2048 < true
total); only the number is corrected. Decode timing unchanged (host-side, zero GPU sync).

## Mainline gap 2 — AC-8: ledgers + evidence + push
- **Full re-run (`results_r3/`, HEAD 8df44a59c)** supersedes `results_r2/` (retired via `SUPERSEDED.md`):
  both comparators rc=3, verdict reproduces (DS PASS@16/32, FAIL@64; DSA also fails@64). `c805b4be5`.
- **Raw evidence committed losslessly** (`*.jsonl.gz`/`*.log.gz` + `EVIDENCE_SHA256.txt` raw+gz hashes +
  `REPRODUCE.md`). **Preflight CLEAN**: git tree clean; all evidence tracked; hash verify lossless; BOTH
  comparator replays from the decompressed committed artifacts rc=3; all 6 DS `trial_evidence.py` PASS.
- **Ledgers regenerated to ONE current state** (`3dc0cb4ef`): `queue.md` op-point mask row = REGENERATED
  (was GONE/regen-mandatory), task2/task3 DONE, task11 ACTIVE-until-push, R3 round-history added;
  `results.md` points at `results_r3`, close-out NOT marked complete until push.
- **PUSH (DONE):** owner-authorized push to the fork `Jiminator/sglang` `dev/double-sparsity-standalone`
  completed (fast-forward `cd2d1e7c1..2ce2adf4e`); public upstream NOT used. The first push was rejected by
  GitHub's 100 MB file limit on an accidental 252 MB tqdm-spam log
  (`results_v2/crash_evidence_r1/log_ds_c64.txt`, committed in R1; key lines already quoted in
  R1_DS_CRASH_FINDING.md). Purged it from history via `git filter-branch` (commits after `8e4407822` re-SHA'd,
  content intact; a pre-rewrite backup branch `loop11b-backup-pre-filter` was kept), then the fast-forward
  push succeeded. meta.json commit_sha run-time stamps (e.g. `8df44a59c`) stay internally consistent
  (DS == DSA) so the comparator gate holds.

## The verdict (unchanged)
DS PASS@conc16 (40.70 / 1.58s) + conc32 (34.05 / 3.00), FAIL@conc64 (26.91 < 30, 25.11s ≥ 22). DSA also
fails @64. Both comparators rc=3. Competitive-to-better than DSA at both op-points; ≤6% per-step tax.

## Files changed (R3)
- `metrics.py`, `models/deepseek_v2.py`, `layers/attention/dsa_backend.py`, `bench_serving.py`,
  `runs/20260616_mb/trial_evidence.py`, `test/.../test_double_sparsity_unit.py` — the total_tokens contract.
- `runs/20260616_mb/mb_r3.sh`; `results_r3/` (corrected verdict + .gz evidence + REPRODUCE + manifest);
  `results_r2/SUPERSEDED.md` (+ R2 evidence removed from tree). `development/{results,queue}.md` regenerated.

## Validation
- 2 unit tests green; trial_evidence catches the mislabel + passes correct records; full re-run all 6 PASS;
  both comparators rc=3 replayed from committed artifacts; preflight clean.

## Queued (not blocking)
Plan-workflow terminology in PRE-EXISTING comments (`batch_result_processor.py`, `benchmark_compare.py`);
new R3 code is terminology-clean. Clean in a focused pass.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260617-ds-total-tokens-explicit-not-rate-inverse
- Notes: publish derived quantities (total = selected/(1-sparsity_rate)) as EXPLICIT fields, not by inverting
  a rate whose convention can be misread; and make the fail-closed validator cross-check the aggregate
  against the per-request arrays + the metric contract, not just an ordering (selected < total).

## Goal Tracker Update Request
### Requested Changes:
- Mark AC-5 RESOLVED (R3): explicit total_tokens; all 6 DS trials trial_evidence PASS with consistency gate.
- Mark AC-8 evidence + ledgers RESOLVED (R3): results_r3 committed losslessly + replay-validated; ledgers current.
- Mark AC-8 PUSH DONE: pushed to owner fork `Jiminator/sglang` `dev/double-sparsity-standalone`
  (fast-forward `cd2d1e7c1..2ce2adf4e`); a 252 MB accidental log was purged from history to satisfy GitHub's
  size limit. AC-8 (and the loop's close-out) is now complete.
### Justification:
Every AC-5/AC-8 item is done + verified, including the owner-authorized push. The verdict (DS PASS@16/32,
FAIL@64) is published with correct, consistency-gated per-trial evidence that replays from committed artifacts.

## Round-4 Codex review — CLEAN (after iterative fixes)
The owner-authorized Codex review (`--base cd2d1e7c1`, since `origin/main` 11605767e shares no ancestor with
the branch; the loop's auto-detected base + Codex's bubblewrap sandbox are blocked in this env) flagged
findings across iterations, all fixed + pushed:
- [P3] `build_corpus.py` creates its gitignored output dir before writing (`da12616a5`).
- [P2] `benchmark_compare.py`: the report verdict now exposes `client_slo_verdict` (gating, matches exit) +
  `directional_verdict`; the directional ratio is labeled REPORT-ONLY (DEC-6, non-gating).
- [P2] `trial_evidence.py`: fails CLOSED on partial/length-mismatched DS arrays; contract over row-aligned arrays.
- [P2] `test_maybe_abort_on_ds_error`: mocks `update_finish_state` (the R1 finisher rename).
- Greened the comparator unit suite: added the locked Option-B field `disable_custom_all_reduce` to the
  fixtures (pre-existing gap) + updated `test_tps/ttft_gate_fail` + `test_too_few_trials` to DEC-6 (directional
  ratios report-only) + the 2-trial floor. Full suite: 383 passed (`9ab62e6ad`).
FINAL re-review: "I did not find any verified, high-signal correctness issues introduced by the diff." Branch
pushed to `Jiminator/sglang` at HEAD `9ab62e6ad`. The loop's deliverable is complete.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
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
44310f230 [loop11b] R1: complete evidence package — DSA server_info + crash-probe txt + crash-log hashes
c16c0d202 [loop11b] R2: wire GLM/dsa-backend DS per-request summary (AC-5) — host-side, graph-robust
b5c4d72be [loop11b] R2: verdict re-established + AC-5 PASS + raw evidence committed (lossless)
8062039d8 [loop11b] R2: AC-8 ledgers regenerated to final state + push status; de-AC the new backend comment
df18a93d0 [loop11b] R3: fix total_tokens metric semantics (AC-5) — explicit field, not rate-inverse
96202e4c4 [loop11b] R3: corrected verdict evidence (results_r3) + supersede results_r2
2ce2adf4e [loop11b] R3: ledgers to one current state (AC-8) — results_r3, mask=regenerated, close-out ACTIVE-until-push
e0935e5a9 [loop11b] R3: AC-8 close-out COMPLETE — pushed to owner fork Jiminator/sglang
da12616a5 [loop11b] R3 review fix [P3]: build_corpus.py creates the output dir before writing
101926d76 [loop11b] R3 review fixes [P2 x2]: report verdict vs exit consistency + fail-closed partial DS evidence
9ab62e6ad [loop11b] R3 review fixes: DS abort test rename + comparator report verdict/labels + green test suite
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-16_11-53-55/round-2-summary.md
- @.humanize/rlcr/2026-06-16_11-53-55/round-2-review-result.md
- @.humanize/rlcr/2026-06-16_11-53-55/round-1-summary.md
- @.humanize/rlcr/2026-06-16_11-53-55/round-1-review-result.md
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-16_11-53-55/round-3-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
