# Code Review - Round 2

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop13/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-2-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 2 Summary — Loop 13 (DS-vs-DSA accuracy diagnosis)

## Objective (round-2-contract.md): make the evidence reproducible/captured + fix correctness
No verdict change. The GOOD-ceiling two-regression verdict (dense = H3 current-slot exclusion;
sparse = the raw-dot `scorer_norm="off"` lock) stands and is unchanged.

## Work Completed
- **DSA-baseline consistency (blocking fix).** Gate/writeup now use the measured **batched** DSA
  0.975/0.973 (was the plan's prior-session 0.953). Sparse gap = 3.3 pp; **GATE stays GOOD**.
  `run_meta.json` SHA corrected (baselines @180f6dd6d, current @HEAD).
- **Verdict softened (blocking fix).** ROOT_CAUSE/gate now label the sparse attribution
  **reference-ceiling** and mark the production-path one-variable bisection **pending**.
- **AC-4 per-arm JSON ledger.** `build_ledger.py` generates `evidence/meta/arms/*.json` for all 8
  arms (config, full server args, scores read from the `.out` files, DS selected-vs-total by
  regime) and regenerates `evidence_table.md`. Fields needing harness instrumentation (per-example
  sample IDs/order; per-step length-cap garbage counters) are listed as `fields_not_instrumented`
  — honest, not faked.
- **AC-3.1 materialized-K equality.** `test_reference_selectors.py::test_materialized_raw_equals_absorbed_raw`
  + `evidence/ac3_1_materialized_k.json`: the materialized fp32 `K_label` score is selection-equal to
  the absorbed raw-dot (max |Δ| 4.8e-6, bit-identical top-k). The identity is exact algebra
  (input-independent), so the synthetic-row proof is conclusive.
- **AC-2 capture pipeline.** `ds_capture` → 1872 score + 104 selection `.pt` dumps →
  `analyze_captures.py` → `cheap_controls.json`, end-to-end. (See limitation below.)
- **Code-comment cleanup.** Removed plan-workflow terms (`AC-*`, `H3`) from my new code/harness
  comments (pre-existing comments untouched). 5/5 CPU tests still pass.

## Files Changed
- Code: `absorbed_latent.py`, `deepseek_v2.py` (comment cleanup only this round; the `normalize`
  control + reference selectors were committed in Round 1).
- Harness/evidence: `serve.sh` (comment cleanup), `build_ledger.py` (new), `evidence/meta/arms/*.json`
  (8 arms), `evidence_table.md` (regenerated), `evidence/{gate_ac5.md, ROOT_CAUSE.md, cheap_controls.json,
  ac3_1_materialized_k.json, meta/run_meta.json}`.
Commit `ac479aeb3` (+ the earlier consistency edits); tree clean; one TP=8 server at a time; GPUs idle.

## Validation
- `python3 development/loop13/test_reference_selectors.py` → 5/5 pass (incl. the AC-3.1 materialized-K
  equality test).
- `build_ledger.py` regenerated the table data-driven from the `.out` files (scores match the
  committed runs). `cheap_controls.json` produced from real `ds_capture` `.pt` dumps.
- Gate re-checked with the consistent measured baseline: dense 0.950 (2.5 pp), sparse 0.940 (3.3 pp)
  → GOOD.

## Remaining Items (honest — do not over-claim)
- **AC-2.2/2.3 numbers are PRELIMINARY.** `analyze_captures.py`'s selected-index-equivalence is a
  cross-record cartesian comparison (not aligned by `(req_pool_index, layer, decode_step)`), and the
  head-agg `pre_reduce_scores` semantics is unconfirmed (served-SUM ≠ post-reduce). Annotated
  PRELIMINARY in `cheap_controls.json`; secondary corroboration only — the verdict does not depend on them.
- **AC-2.1 forced-all physical-slot assertion JSON** + **AC-4 per-step garbage counters** + sample
  IDs/order: not built — require `logical_to_physical`/adapter capture instrumentation in the serving path.
- **AC-6 production-path bisection** incomplete: reference-ceiling cliff (cosine 0.940 vs raw-dot
  0.013 + materialized-raw proof) names the sparse candidate and the opts are bounded second-order
  (production raw-dot 0.000 ≈ exact raw-dot 0.013), but the one-variable production arms are not run.

## Goal Tracker Update Request

### Requested Changes / scope decisions needed:
1. **AC-6 "production-style cosine" conflicts with the no-fix constraint.** Cosine is NOT servable on
   the production graph-safe path without implementing the materialized per-head signature there —
   which IS the recommended fix. The plan says "no fix this loop." So either (a) accept the
   reference-ceiling AC-6 attribution (cliff cosine 0.940 vs raw-dot 0.013 + materialized-raw proof +
   opts-second-order) as the diagnosis-loop result, or (b) authorize landing the production cosine
   path (a fix-adjacent change) to complete the production-path bisection. **Recommend (a).**
2. **AC-2.1 forced-all assertions + AC-4 garbage counters + sample-ID/order** require building
   `logical_to_physical`/adapter capture instrumentation + harness changes. Confirm whether this is in
   scope for a diagnosis loop whose verdict already does not depend on it (it would CORROBORATE the
   already-decisive forced-all GSM8K recovery 0.620→0.950), or defer to a follow-up.
3. **AC-2.2/2.3 captured cheap-controls** need a per-`(req,layer,step)`-aligned analyzer + confirmation
   of `pre_reduce_scores` semantics before their numbers are load-bearing.

### Justification:
The Ultimate Goal is a root-cause verdict with live evidence. That verdict is delivered and
Codex-accepted in substance (GOOD ceiling; dense=H3, sparse=raw-dot scorer lock), and the decisive
controls (forced-all/anchor dense recovery; faithful raw-dot vs faithful cosine sparse;
materialized-raw selection-equality) are live GSM8K + proven. The open items are CORROBORATING rigor
whose two largest pieces (production-style cosine; adapter garbage-counter instrumentation) either
conflict with the "no fix" constraint or require building serving instrumentation beyond a diagnosis
loop. Surfacing this for an explicit scope decision rather than silently deferring.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: Round 2 was evidence-completeness + correctness consolidation; the load-bearing lessons
  (BL-20260620-ds-current-slot-exclusion, BL-20260620-ds-rawdot-scorer-lock) were captured in
  Rounds 0–1 and are unchanged.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-20_09-57-51/round-2-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
