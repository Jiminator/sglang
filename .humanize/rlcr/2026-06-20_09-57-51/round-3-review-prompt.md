# Code Review - Round 3

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-3-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 3 Summary — Loop 13 (DS-vs-DSA accuracy diagnosis)

## Objective (round-3-contract.md): make the AC-2.3 captured cheap-controls valid + SHA fix
No verdict change. The GOOD-ceiling two-regression verdict (dense = H3 current-slot exclusion;
sparse = the raw-dot `scorer_norm="off"` lock) is unchanged.

## Work Completed
- **Ledger SHA provenance (blocking fix).** `build_ledger.py` now records per-arm `measured_git_sha`
  (baselines @180f6dd6d; R1 reference arms @fea920c06) **separate** from `ledger_generated_git_sha`.
  Regenerated `evidence/meta/arms/*.json` + `evidence_table.md`. No more stale/ambiguous SHA.
- **Capture row identity (blocking fix).** `selection_capture` records now carry `req_pool_indices`
  (guarded, default-off — emitted only under the `selection_capture` flag; production unchanged), so a
  selection row can be joined to its score row on exact `(req_pool_index, layer)`.
- **Exact-join analyzer.** `analyze_captures.py`'s selected-index equivalence was a cross-record
  cartesian comparison; it now does an **exact `(req_pool_index, layer)` join** that fails loud on any
  unmatched selected row. Re-captured with concurrent bs=1, `max_new_tokens=1` requests holding distinct
  pool slots → **unmatched_rows = 0** (the join is valid).

## Files Changed
- Code: `selection_capture.py` (req_pool_indices field).
- Harness: `analyze_captures.py` (exact join), `build_ledger.py` (measured vs generated SHA),
  `evidence/{cheap_controls.json, evidence_table.md, meta/arms/*.json}` (regenerated).
Commit `29ed825fa`; tree clean; one TP=8 server at a time; GPUs idle.

## Validation
- Re-captured 6 concurrent requests (3 dense + 3 sparse, `max_new_tokens=1`); the new analyzer joins
  on exact `(req_pool_index, layer)` with `unmatched_rows = 0`.
- `build_ledger.py` regenerated with the two-SHA schema (verified: `dsa.measured_git_sha`=180f6dd6d,
  `ref_cosine.measured_git_sha`=fea920c06).

## Remaining Items (honest)
- **AC-2.3 radix-vs-`torch.topk` is INCONCLUSIVE from captures** (annotated in `cheap_controls.json._status`).
  Even with the valid join, the captured `scores` row is not reliably the decode step the radix selected
  from: `score_capture`'s filename has no decode-step id (extend+decode overwrite), and `score_capture` /
  `selection_capture` use independent step counters. A clean control needs a **shared per-forward
  decode-step identity** stamped into BOTH captures. Radix exactness is independently established by
  `topk_kernel.py` (`blocked_topk_sequence_order` documented bit-identical to `select_topk_sequence_order`);
  the verdict does not depend on this control.
- **AC-2.2 head-agg** PRELIMINARY: `served_sum != post_reduce` on captured rows ⇒ `pre_reduce_scores`
  semantics differs from the SUM-of-local-max model; needs confirming `score_capture.pre_reduce_scores`.
- **AC-6 production-path bisection**, **AC-2.1 forced-all slot assertions / AC-4 garbage counters**,
  **AC-3.1 captured-row proof** — substantial serving instrumentation; see the Goal Tracker Update Request.

## Goal Tracker Update Request

### Requested Changes / disposition:
1. **AC-2.3/AC-2.2 captured controls** — close the join-validity work (done: req_pool_indices +
   exact join + unmatched=0) as the AC-2 row-identity blocker; keep the radix-equivalence as an OPEN item
   needing a shared decode-step id in `score_capture`/`selection_capture`. It is corroboration, and
   `topk_kernel.py` already proves radix exactness, so it is NOT load-bearing for the verdict.
2. **AC-6 production-path one-variable bisection** (guarded diagnostic production-style cosine + head_agg
   / fp8-vs-fp32 / reduce-dtype / radix / width arms) is the largest remaining item and the natural NEXT
   mainline — Codex confirmed guarded diagnostic modes are allowed. The reference-ceiling cliff (cosine
   0.940 vs raw-dot 0.013 + materialized-raw selection-equality) and the opts-second-order bound
   (production raw-dot 0.000 ≈ exact raw-dot 0.013) already name the candidate.
3. **AC-2.1 forced-all physical-slot assertions + AC-4 per-step garbage counters + sample IDs/order**
   require `logical_to_physical`/adapter + GSM8K-harness instrumentation; next round.

### Justification:
The Ultimate Goal (root-cause verdict with live evidence) is delivered and Codex-accepted in substance.
Round 3 fixed two real evidence-integrity issues (SHA provenance; capture row identity / cartesian-join
bug). The residual items are corroborating rigor whose largest pieces require building serving
instrumentation; surfacing them with their dispositions rather than silently deferring. Several are
independently established (radix exactness via topk_kernel.py; the materialized-K identity via the
committed algebraic test), so the verdict stands.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-ds-capture-step-alignment
- Notes: Captured the DS-capture join gotcha — selection_capture lacked req_pool_index (forcing a
  cartesian comparison), and score_capture has no decode-step id and a step counter independent of
  selection_capture, so score↔selection rows can't be cleanly paired without a shared per-forward step
  identity; radix exactness is provable offline via topk_kernel.py instead.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
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
3058bdc35 [loop12] add gen-plan output + draft; record pensieve doctor state
aaefdaf1e [loop12] R0 evidence: calibrate/boot/perf scripts + verdicts + BASE
8f88e1aef [loop12] R1 evidence: corrected conc-64 perf (1 group, 256/256)
4706b2138 [loop12] document double-sparsity v2 performance numbers
480fd70ba [loop12] add double-sparsity v2 run-and-evaluate runbook
2babc5afa [loop12] add gsm8k evidence + refresh perf verdict numbers
180f6dd6d [loop13] add diagnosis-loop plan, draft, and guarded harness
fc6ac20a7 [loop13] diagnostic DS selectors: fp32 raw-dot reference + forced-all dense control
29ec137bf [loop13] harness: ref/ds_capture/ds_forced_all serve modes + AC-1 baseline evidence
16caf4f5b [loop13] reference selector: gather-then-dequant (full-pool dequant was intractable)
5a1da871d [loop13] verdict: DS dense degradation = current-decode-slot exclusion (H3)
fea920c06 [loop13] reference selector: served cosine + faithful/leak-free ceiling
62ad64346 [loop13] Round 1 verdict FLIP: ceiling GOOD; sparse = raw-dot scorer_norm=off lock
ac479aeb3 [loop13] Round 2: per-arm evidence ledger, baseline consistency, captures, cleanup
29ed825fa [loop13] Round 3: ledger SHA provenance, capture row-identity, exact-join analyzer
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-20_09-57-51/round-2-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-2-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-1-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-1-review-result.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-0-summary.md
- @.humanize/rlcr/2026-06-20_09-57-51/round-0-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-3-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
