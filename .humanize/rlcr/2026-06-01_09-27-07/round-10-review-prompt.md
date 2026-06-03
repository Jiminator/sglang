# Code Review - Round 10

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-10-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 10 Summary — Loop 7

## Mainline objective (round-10-contract.md)
**Land the Tier-2.A lifted-budget ABI (AC-4 / task13)**: the opt-in config fields
+ validator gating + the Codex-reviewed design/disposition record (the AC-4
foundation the decode path plugs into).

## Outcome: ACHIEVED — task13 done; the AC-4 ABI is landed and the decode-path design is recorded.

## Work completed
1. **ABI config fields.** `DoubleSparsityConfig.enable_lifted_budget_decode: bool`
   (default `False`) + `lifted_budget_top_k: int` (default `0`), in
   `_ALLOWED_FIELDS` + dataclass + `parse_double_sparsity_config`.
   `__post_init__` validation: `lifted_budget_top_k` must be `> top_k` when
   enabled; **set-without-flag and flag-without-budget both fail closed** (no
   silent no-op).
2. **Validator gate.** `top_k > index_topk` is **rejected unless
   `enable_lifted_budget_decode`** is set, and `lifted_budget_top_k` must be
   `> index_topk`; the error steers to the ABI and explicitly forbids
   `SGLANG_DS_ALLOW_TOPK_MISMATCH` / `max_top_k` / Twilight fields as the
   mechanism. **Default-off leaves the DSA `dsa_index_topk` assert + the
   equality-mismatch ablation escape unchanged.**
3. **task13 design/disposition record** (`m7_lifted_budget_design.md`),
   **reviewed via `ask-codex`** and integrated: the physical →
   `page_table_1_flattened` → **request-local compact** dequant-index remap;
   prefix-sharing is safe per-request but **within-row duplicates are not**
   (`flash_mla_sparse_fwd` would double-attend → dedup after remap); **`-1` pads
   masked before dequant** (a `-1` into `dequantize_k_cache_paged` is invalid);
   the alloc-free `out=` dequant + CUDA-graph landing **deferred to task16** per
   DEC-4/DEC-6 (eager research path first, gated off production capture); a
   **direct `flash_mla_sparse_fwd` 4K-topk smoke** required (local coverage is
   sparse-prefill top-k ≤ 512).
4. **Bundled (Codex R9 queued #1 + claim-correction #4).** Clamped
   `_force_include_anchor`'s temp shape to `A = min(anchor_budget, top_k, max_seq)`
   (bit-identical — clamped-out slots are invalid anyway; bounds a pathological
   opt-in budget) + a new **over-budget (`anchor_budget > top_k`, seq_len < K)**
   GPU eager-vs-graph test.

## Validation
- `TestLiftedBudgetABI`: config accept/reject matrix (valid lifted; reject
  lbk≤top_k, lbk-without-flag, flag-without-lbk; Twilight/`max_top_k` still
  rejected as unknown) + the validator `top_k > index_topk` gate via a
  monkeypatched `get_dsa_index_topk`.
- `test_anchor_over_budget_graph_matches_eager`: over-budget anchor bit-identical
  eager-vs-graph.
- **354 DS unit tests pass.**

## Files changed
`config.py` (ABI fields + validation + `_coerce_bool` field name),
`validator.py` (top_k>index_topk gate), `selection_kernel.py` (anchor temp clamp),
`test_scorer_variants.py` (`TestLiftedBudgetABI` + over-budget anchor test),
`m7_lifted_budget_design.md` (new). Commit `c41e5193a` (pushed).

## AC-4 status
ABI + validator gating + design/disposition **landed (task13 DONE)**; AC-4 itself
remains NOT MET pending the decode path + served recall evidence + the task17
disposition (task14–17). The plan's "landed-or-deferred-with-evidence" branch is
the planned closure given Tier-2.A is bounded-secondary (the long-context goal is
served by the landed Tier-2.B hybrid scorer).

## Remaining items (queued, justified)
- **task14/15: eager research lifted-budget decode path + correctness/safety
  tests** (incl. the required direct `flash_mla_sparse_fwd` 4K-topk smoke).
- **task16: production hardening** (alloc-free dequant + CUDA-graph), gated behind
  the recall win.
- **task17: Tier-2.A landing disposition record** (closes AC-4).
- **AC-6 perf consolidation (task19) + final strategic-gate decision record
  (task20)**.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: the ABI + validator gating follow the established config-borne-flag
  pattern; the one design-hygiene point (a wider-budget `top_k>index_topk` needs
  its OWN opt-in ABI, distinct from the equality-mismatch `SGLANG_DS_ALLOW_TOPK_MISMATCH`
  ablation escape) is captured in the validator message + the design doc, not a
  reusable multi-round pitfall.

## Goal Tracker Update Request
- **task13** (AC-4): lifted-budget ABI + design record DONE (R10).
- **Resolve queued**: anchor-budget temp-shape clamp (done R10); MMLU data_dir
  (done R9).
- **Keep Active**: task14–17 (AC-4 decode path/tests/disposition), AC-6/task19–20
  (perf + final decision record).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-9-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-9-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-8-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-8-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-7-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-7-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-10-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
