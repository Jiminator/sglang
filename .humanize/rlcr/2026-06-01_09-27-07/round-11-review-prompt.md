# Code Review - Round 11

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-11-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 11 Summary — Loop 7

## Mainline objective (round-11-contract.md)
**Complete AC-4 / task13 by making the lifted-budget opt-in fail closed at
startup.** `enable_lifted_budget_decode=true` must raise a clear
*recognized-but-not-implemented/selected* error until the opt-in decode backend
path (task14) is implemented/selected — closing the two R10-review startup holes.

## Outcome: ACHIEVED — task13 DONE; the R10-review Blocking Side Issue is resolved.

## Problem (R10 review, verified in code before fixing)
`validate_double_sparsity` accepted a lifted config because the model-topk block's
`elif lifted: logger.info(...)` branch passed it through (`validator.py:235-240`),
even though no lifted decode backend exists. Two concrete holes (both confirmed):
- **silent no-op**: `top_k=2048, enable_lifted_budget_decode=true, lifted_budget_top_k=4096`
  booted the locked 2048 selector (the wider budget was never honored);
- **wide-into-old-assert**: `top_k=4096, enable_lifted_budget_decode=true, lifted_budget_top_k=8192`
  booted toward the default `flashmla_kv` `indices.shape[-1] == dsa_index_topk` (2048) assert.

## Work Completed (`coding`, Claude)
1. **Capability seam.** Added `ds_lifted_budget_decode_available()` in
   `selection_kernel.py` (returns `False` today), mirroring the existing
   `ds_scorer_is_graph_safe` idiom. This is the **single one-line seam** the
   decode-path landing (task14) flips to `True` once the path exists.
2. **Fail-closed validator gate.** `validate_double_sparsity` now raises a clear
   "recognized but not implemented/selected" `ValueError` whenever
   `enable_lifted_budget_decode` is set while
   `ds_lifted_budget_decode_available()` is `False`. Placed **right after the
   channel_mask_path check, before the capability/model-topk block**, so it is
   **hf_config-independent** — it cannot be skipped when the model config can't be
   resolved. The error names both failure modes and the remedy.
3. **Steering + defaults preserved.** The no-flag `top_k > index_topk` steering
   (toward the ABI, not `SGLANG_DS_ALLOW_TOPK_MISMATCH`) is unchanged. The
   model-topk block's lifted-shape validation (`lifted_budget_top_k > index_topk`
   + info log) is retained as the **post-backend** layer (reachable once the seam
   flips). Default-off path, the DSA `dsa_index_topk` assert, and the
   `SGLANG_DS_ALLOW_TOPK_MISMATCH` ablation are untouched.
4. **Design doc.** `m7_lifted_budget_design.md` updated: the "Landed" section now
   states the R11 fail-closed gate + the enablement seam (task14 knows the
   one-line flip and that the model-topk lifted validation activates afterward).

## Files Changed
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` —
  `+ ds_lifted_budget_decode_available()` seam (returns `False`).
- `python/sglang/srt/layers/attention/double_sparsity/validator.py` —
  `+` hf_config-independent fail-closed gate on `enable_lifted_budget_decode`.
- `test/registered/unit/layers/attention/test_scorer_variants.py` —
  flip the stale `test_validator_topk_gt_index_topk_requires_flag` "WITH flag
  passes" assertion to no-flag-steering-only; `+` two fail-closed case tests.
- `development/loop7/m7_lifted_budget_design.md` — R11 fail-closed gate + seam.
- Commit `a62ce91de` (local — see Push note).

## Validation
- `pytest test_scorer_variants.py -k LiftedBudget` → **9 passed** (incl. both new
  fail-closed cases now RAISE, and the no-flag steering still raises).
- Full DS unit suite (`test_scorer_variants.py`, `test_double_sparsity_unit.py`,
  `test_ds_scorer_tp_determinism.py`) → **322 passed + 9 subtests**, no regressions.
- Config parse-level ABI validation unchanged (the field still parses; the
  fail-closed guard lives at the server validator).

## Push note
`git push` was **blocked by the loop bash hook** ("Current commits should stay
local … use `--push-every-round`"). The commit is local-only; the loop manages it.
(For this loop the harness explicitly overrides the standing round-boundary-push
preference.)

## Remaining Items (active mainline, NOT queued-out)
- **task14 (next mainline)** — eager research lifted-budget decode path
  (`flash_mla_sparse_fwd` + request-local compact remap, dedup, `-1`/pad mask,
  R23 tie-break); flips `ds_lifted_budget_decode_available()`.
- **task15** — kernel-correctness/safety tests incl. the direct
  `flash_mla_sparse_fwd` 4K-topk smoke.
- **task16** — production hardening (alloc-free `out=` dequant + CUDA-graph),
  gated behind the recall win.
- **task17** — Tier-2.A landing disposition record (closes AC-4 via
  landed-or-deferred-with-evidence).
- **task19 / task20** — AC-6 perf consolidation + final strategic-gate
  supersession decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## AC-4 status
**task13 DONE** (opt-in ABI fields + Codex-reviewed design + fail-closed validator
gate). **AC-4 itself remains NOT MET** — the decode path is task14–17. The seam is
in place so task14 is a clean continuation.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: Selected `BL-20260602-ds-flag-must-be-config-borne-not-env` (the gate
  reads `config.enable_lifted_budget_decode`, config-borne — reaches TP workers,
  no env read) and `BL-20260528-dsv32-ds-serving-boot-chain` (fail closed at boot
  with a clear error). Both were *applied*, not extended: a one-round, single-gate
  validator fix following established patterns, not a multi-round pitfall.

## Goal Tracker
Updated directly (Plan Version 13): R11 Plan Evolution row added; task13 moved to
Completed and Verified (pending R11 review); the R10-review Blocking Side Issue
moved to Resolved (R11); Blocking Side Issues now empty; task14 flagged as the next
mainline with the seam note. No Goal Tracker Update Request needed.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
9ca1f5133 [Sparsity] Loop-7 recall R&D: plan + QA (active plan gitignored)
9914a3004 [Sparsity] Loop-7 M0: selection-recall oracle diagnostic math
8074cb1cf [Sparsity] Loop-7 M0: oracle sink + AC-1.1 force + flag-gated hook
c6ffcdea6 [Sparsity] Loop-7 M0: DS served-recall baseline at mem 0.7 (N=20)
78f6b5d17 [Sparsity] Loop-7 M0: oracle budget-vs-scorer evidence (A-vs-B decider)
a1e2c72dc [Sparsity] Loop-7 M0: A-vs-B decision (Codex-adjudicated)
599d7cc99 [Sparsity] Loop-7 M1: flag-gated cosine scorer (Tier-2.B candidate)
e2674f4f4 [Sparsity] Loop-7 M1: cosine scorer MEASURED — 16K recall 5%->40%
c5a829def [Sparsity] Loop-7: oracle trial-file read fresh; gitignore transient artifacts
273622705 [Sparsity] Loop-7 R1: length-conditional hybrid scorer (best of both regimes)
72c704edf [Sparsity] Loop-7 R2: scorer variants correct + production-safe
fc8871372 [Sparsity] Loop-7 R3: fix anchor over-budget + TP=8 logical-path matrix
bf2ce9b2b [Sparsity] Loop-7 R4: oracle fail-closed + config-borne + 64K binding re-run
9f76ad659 [Sparsity] Loop-7 R5: binding DS-vs-DSA same-node served-recall matrix (AC-2)
cb02b6673 [Sparsity] Loop-7 R6: port Tier-2.B scorer to the graph-safe Triton path (AC-3)
9a37590ec [Sparsity] Loop-7 R7: binding AC-3 non-regression matrix (graph-mode N=50 + MMLU)
f05cb730e [Sparsity] Loop-7 R8: close AC-1 (oracle-off zero-hot-path + stride reference)
e7cf1f146 [Sparsity] Loop-7 R9: port anchor-budget variant to the graph-safe path (AC-3)
c41e5193a [Sparsity] Loop-7 R10: lifted-budget ABI + design record (AC-4 task13)
a62ce91de [Sparsity] Loop-7 R11: fail-closed lifted-budget decode opt-in at the validator
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-10-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-10-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-9-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-9-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-8-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-8-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-11-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
