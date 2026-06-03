# Code Review - Round 16

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-16-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 16 Summary — Loop 7

## Mainline objective (round-16-contract.md)
**task16 (part 1) — land the graph-safe lifted-budget decode primitives and prove
them zero-alloc under real CUDA-graph capture/replay + eager-equivalent.**

(The R15 review STALLED the loop and overrode the deferred-with-evidence close,
requiring task16 production hardening to be implemented. This round lands the
technical core; the backend/cuda-graph-runner plumbing + live re-measure is part 2.)

## Outcome: ACHIEVED — the graph-safe primitives are landed and PROVEN zero-alloc on GPU.

## The blocker, and the fix
The load-bearing graph-safety problem was the **dynamic shape**:
`build_lifted_compact_kv` produced a `total_valid`-length compact buffer that varies
per decode step (uncapturable), and `dequantize_k_cache_paged` allocates internally.

## Work Completed (`coding`, Claude)
1. **`dequantize_k_cache_paged_out(quant, page_table_1_flattened, out, group_size=128)`**
   (`dsa/dequant_k_cache.py`): alloc-free dequant writing into a caller-owned bf16
   scratch (no internal `torch.empty`); the existing allocating
   `dequantize_k_cache_paged` is now a **thin wrapper** around it (byte-identical).
2. **`build_lifted_compact_index_fixed` / `build_lifted_compact_kv_fixed`**
   (`double_sparsity/lifted_budget.py`): a **fixed-shape, fully-tensorized** graph-safe
   compact builder (no `.item()`, no dynamic boolean-mask shapes). It keeps a fixed
   `[bs*lifted_width]` layout — every lane gets a compact row at ordinal
   `b*width+lane`, so the compact buffer is always `[bs*width, 1, 576]`. Masked /
   within-row-duplicate lanes write a **safe in-bounds physical slot** into the
   dequant input (never `-1`) and `-1` into the compact index (so
   `flash_mla_sparse_fwd` masks them). A request attends **exactly the same valid
   (post-dedup) slots as the eager builder — identical attention** — in a capturable
   fixed shape. `build_lifted_compact_kv_fixed` runs it + the alloc-free `out=` dequant
   into preallocated scratch.

## Files Changed
- `dsa/dequant_k_cache.py` (`+ dequantize_k_cache_paged_out`; existing API wraps it).
- `double_sparsity/lifted_budget.py` (`+ build_lifted_compact_index_fixed` / `_kv_fixed`).
- `test_lifted_budget_decode.py` (CPU fixed-layout/dedup; dequant `out=` equivalence;
  GPU CUDA-graph zero-alloc capture/replay at 4096/8192).
- `development/loop7/m7_lifted_budget_design.md` (task16 primitives DONE + part-2 scope),
  `m9_tier2a_disposition.md` (superseded banner — the deferral is overridden).
- Commit `714cf62b2` (local — loop hook keeps commits local until completion).

## Validation
- `TestLiftedCompactIndexFixed` (CPU): the fixed `b*width+lane` layout, safe-slot for
  masked lanes, `-1` compact index, within-row dedup keep-first — **1 passed**.
- `TestLiftedBudgetGraphSafe` (GPU, H200): `dequantize_k_cache_paged_out` **byte-identical**
  to the allocating dequant; and a **real `torch.cuda.CUDAGraph` capture** of
  (fixed builder → dequant `out=` into scratch → `flash_mla_sparse_fwd`) **replays
  ZERO-alloc** (`assert_no_alloc_in_region`) at **4096 and 8192** and matches the eager
  reference — **3 passed**.
- Full DS unit suite (4 files) → **345 passed + 9 subtests** (was 341; +4 R16), no
  regression. No existing runtime path changed (the backend still uses the eager
  builder; the validator still requires `--disable-cuda-graph`); default byte-identical.

## Remaining Items (active mainline — task16 part 2, next round)
- Wire the fixed scratch (incl. a **q head-padding** scratch) into `DSGraphState` /
  `allocate_graph_state`; route `_forward_lifted_budget` through the scratch-backed
  fixed path under capture (keep the eager path for non-graph runs).
- **Relax the validator `--disable-cuda-graph`** requirement for the lifted path.
- **Live** boot with CUDA graph + **graph-mode 4K recall re-measure** (the eager 95%
  is not the graph number) + perf/memory; graph-captured TP=8 determinism.
- **task17 redo**: the production-ready landing disposition after part 2 lands.
- Then **AC-6/task19** (perf consolidation) + **AC-2/task20** (final decision record).
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## AC-4 status
task16 part 1 (graph-safe primitives + zero-alloc proof) **done**; the hardest
technical risk is implemented and proven on GPU. **AC-4 stays NOT MET** until part 2
(backend integration + validator relax + live graph-mode re-measure) + the task17
production-ready disposition.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260602-flash-mla-sparse-fwd-compact-decode-contract
- Notes: added the **graph-safety corollary** — the dynamic-`total_valid` compaction
  is uncapturable; use a FIXED `[bs*width]` layout (masked/dup lanes → safe in-bounds
  slot in the dequant input + `-1` compact index) instead of compacting, paired with
  an alloc-free `dequantize_k_cache_paged_out`, and PROVE it with a real CUDAGraph
  capture + `assert_no_alloc_in_region(replay)`. A reusable graph-safety technique for
  any DS dynamic-length gather/dequant path.

## Goal Tracker
Updated directly (Plan Version 22): R16 Plan Evolution row; task16 → in progress
(primitives done R16; part-2 integration/validator/live remain); task17 redo pending.
No Goal Tracker Update Request needed.
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
0ad20774a [Sparsity] Loop-7 R14: binding served 4K recall recovery for the lifted-budget path
b70f48d36 [Sparsity] Loop-7 R15: Tier-2.A landing disposition (deferred-with-evidence) — closes AC-4
714cf62b2 [Sparsity] Loop-7 R16: graph-safe lifted-budget decode primitives + zero-alloc replay proof
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-15-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-15-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-14-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-14-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-13-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-13-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-16-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
