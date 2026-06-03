# Code Review - Round 13

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-13-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 13 Summary — Loop 7

## Mainline objective (round-13-contract.md)
**task14 (completion) — wire the served opt-in *eager* lifted-budget decode branch
end-to-end and flip the availability seam.**

## Outcome: ACHIEVED — task14 DONE; the served eager branch is wired + enabled.

## What it does
When `enable_lifted_budget_decode` is set (default off, eager-only until task16),
DS decode selects up to `lifted_budget_top_k` logical positions, converts them to
physical KV slots, runs the R12 request-local compact remap, dequantizes the
selected fp8 slots via `dequantize_k_cache_paged`, and attends them with
`flash_mla_sparse_fwd` (no 2048 cap) — instead of the default `flashmla_kv` path.

## Work Completed (`coding`, Claude)
1. **Config** (`config.py`): enforce `lifted_budget_top_k % 128 == 0` (the
   `flash_mla_sparse_fwd` `topk % (2*B_TOPK)` block constraint), alongside `> top_k`.
2. **Validator** (`validator.py`): when enabled, require `top_k == index_topk` (the
   base budget stays the DSA budget; `lifted_budget_top_k` is the SEPARATE wider
   width), `lifted_budget_top_k > index_topk`, `% 128`, and **`--disable-cuda-graph`**
   (the dequant allocates internally, not graph-safe). The R11 "not implemented"
   fail-closed gate is replaced by these checks (kept as defense if a build ever
   ships the flag without the backend).
3. **Selection width** (`selector.py` + `dsa_backend.py`):
   `DoubleSparsitySelector.max_top_k` and the backend's `ds_max_top_k` (which sizes
   `ds_topk_indices_out` + `ds_graph_state`) widen to `lifted_budget_top_k` when
   enabled — one value cascades the selection/output buffers to lifted width; the
   R23 tie-break is unchanged.
4. **Decode branch** (`dsa_backend.py` + `lifted_budget.py`): `forward_decode`
   routes the lifted case (the physical FUSE_TOPK `page_table_1`) to a new
   `_forward_lifted_budget` → `build_lifted_compact_kv` (remap +
   `dequantize_k_cache_paged` for the fp8 store, gather for bf16) → the existing
   `_forward_flashmla_sparse`. Behind a default-off
   `getattr(self, "ds_lifted_budget_decode", False)` guard so the default DSA/DS
   decode is byte-identical and the `flashmla_kv` `dsa_index_topk` assert is untouched.
5. **Seam** (`selection_kernel.py`): `ds_lifted_budget_decode_available()` → `True`.

## Files Changed
- `double_sparsity/config.py` (`%128` validation).
- `double_sparsity/lifted_budget.py` (`+ build_lifted_compact_kv` decode helper).
- `double_sparsity/selection_kernel.py` (seam → True).
- `double_sparsity/selector.py` (lifted `max_top_k`).
- `double_sparsity/validator.py` (lifted gating: eager-required, `top_k==index_topk`).
- `dsa_backend.py` (`ds_lifted_budget_decode`/`ds_max_top_k` at init; `forward_decode`
  lifted route; `_forward_lifted_budget`).
- `test_scorer_variants.py` (`TestLiftedBudgetABI`: `%128` reject/accept + validator
  gating; replaced the stale R11 "not implemented" assertions).
- `test_lifted_budget_decode.py` (GPU served-helper tests at 4096/8192).
- `m7_lifted_budget_design.md` (served branch landed + updated risks).
- Commit `2ba4dafc1` (local — loop hook keeps commits local until completion).

## Validation
- `TestLiftedBudgetABI` (config `%128` + validator gating) + `test_lifted_budget_decode`
  → **24 passed** (incl. the new 4096/8192 served-helper GPU tests).
- GPU served-helper at **4096 and 8192** widths via the production
  `build_lifted_compact_kv`: prefix-sharing, `valid_lengths` < width, and an
  **interior `-1` from within-row dedup**, all matched vs a reference attention; the
  4096 case confirms a request attends 3000 > 2048 rows (no cap).
- Full DS unit suite (4 files) → **337 passed + 9 subtests**, no regressions.
  (Fixed a partial-backend stub by reading the new flag via `getattr(..., False)`.)
- Default-off path byte-identical; DSA `dsa_index_topk` assert +
  `SGLANG_DS_ALLOW_TOPK_MISMATCH` untouched; no new plan-marker leakage.

## Remaining Items (active mainline, NOT queued-out)
- **task15 (remaining)** — a **live served NIAH 4K recall-recovery sweep** (eager,
  N≥20 + CIs; the M0 oracle predicted recall@4096 ≈ 100% vs recall@2048 ≈ 44%) —
  the binding recall evidence; + TP=8 selected-index equality at the lifted
  4096/8192 width (extend the existing TP determinism harness).
- **task16** — production hardening (alloc-free `out=`/scratch dequant + CUDA-graph
  capture), gated behind the recall win; the path stays eager-required until then.
- **task17** — Tier-2.A landing disposition record.
- **task19 / task20** — AC-6 perf consolidation + final strategic-gate decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## AC-4 status
**task14 DONE** (served eager branch wired + enabled). **AC-4 NOT MET** — served 4K
recall-recovery evidence + TP=8 equality (task15), task16 hardening, and the task17
disposition remain.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: This round *applied* the R12 lesson
  `BL-20260602-flash-mla-sparse-fwd-compact-decode-contract` (the `%128` width, the
  `<0`/`>=s_kv` masking, and the request-local compact remap) directly in the config
  `%128` check, the validator gating, and the decode helper. The wiring specifics
  (widening the single `ds_max_top_k` cascades the metadata buffers; FUSE_TOPK gives
  a physical `page_table_1` at decode; `getattr`-default guards for partial test
  backends) are codebase-structural, not a reusable cross-round pitfall — no new or
  updated lesson.

## Goal Tracker
Updated directly (Plan Version 16): R13 Plan Evolution row added; task14 → **done**;
task15 → partial (decode-helper tests done; live recall + TP=8 remain). No Goal
Tracker Update Request needed.
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
d187f59f4 [Sparsity] Loop-7 R12: lifted-budget decode index core + flash_mla_sparse_fwd kernel proof
2ba4dafc1 [Sparsity] Loop-7 R13: wire the served eager lifted-budget decode branch + enable the seam
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-12-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-12-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-11-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-11-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-10-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-10-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-13-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
