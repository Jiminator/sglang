# Code Review - Round 0

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-0-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 0 Summary — Loop 6 Milestone 1 (strategic gate + feasibility budget)

## Mainline objective (round contract)
Deliver **Milestone 1**: the two analyze-only, pre-coding gates that the whole loop depends on — AC-1 (strategic recall-R&D decision) and AC-2 (the binding HBM footprint-feasibility budget). No source/test/serve/bench code touched; no hardware run. This is the correct opening round: per loop discipline the spine starts "gate → feasibility → footprint", and AC-2's lever selection is binding on AC-3 (so footprint code must not start before this budget exists).

## What was done

### AC-1 — strategic gate (`analyze` → Codex, integrated by Claude)
Artifact: `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`.
- **Decision:** pursue DS long-context-recall R&D on V3.2, but strictly **after** the engineering spine lands; direction = a **custom sparse-matmul DECODE kernel mirroring NSA/DSA with an adjustable `top_k`** (relaxing the `indices.shape[-1] == dsa_index_topk` hard cap); learned/query-aware selector secondary. DSA stays the production default; the DS compact path stays opt-in/reversible.
- **Rationale (evidence-grounded):** DS `top_k` is kernel-locked to the native DSA `index_topk=2048` (the shared `flashmla_kv` decode kernel asserts `indices.shape[-1]==dsa_index_topk`, not bypassable by `SGLANG_DS_ALLOW_TOPK_MISMATCH=1`); DS NIAH recall 75/5/0 at 4K/16K/64K vs DSA 100 at the **same** budget+kernel; dense (seq≤2048) DS recall=100% proves DS decode is sound → the gap is **selection quality** vs the trained DSA indexer, not budget size, and widening `top_k` is not an available lever without a new kernel.
- **Sequencing:** gated behind this doc AND a landed spine; must not block/regress the spine; legitimately deferrable to its own loop. Note: DS's value proposition is stronger on a model with no trained sparse indexer (deferred GLM-5.1 / 128k).

### AC-2 — footprint feasibility budget (`analyze` → Codex, integrated + verified by Claude)
Artifact: `runs/20260530_dsv32_loop6/footprint_feasibility.md`.
- **Grounded in real Loop-5 hardware anchors** (verified against `runs/20260528_dsv32_mvp/` boot logs and `token_label_table.py`): f=0.6 → table 1.55 GB, `max_total=53056`, headroom 37.78 GB, **serves** (admits 35.7/64); f≈0.70 → table 11.52 GB, `max_total=396096`, headroom 12.29 GB, **gen-OOM**; f=0.897 → table-alloc 31.18 GiB with 7.20 GB free, **boot-OOM**. Table formula `61·max_tokens·16·16·2`; KV ≈ 46.9 KiB/token.
- **Admission target:** 53056/35.7 = 1486 tok/admitted-req → **≥95K** tokens to admit conc 64, **~114K** with a 20% margin.
- **Per-lever budget** (freed-HBM, scale overhead, `f` needed, table bytes, predicted headroom, predicted conc-64) for: (i) fp16 + raise `mem_fraction_static` (no code), (ii) **int8 same-`label_dim`** (0.5625× fp16, ~1.78× win — scale overhead included), (iii) page-level/two-stage (~1/64×, NIAH-non-regression class).
- **Binding decision:** int8 is **predicted sufficient** → int8 is the selected **compaction** lever (build int8, not page-level). **But** the budget honestly rules the no-code fp16-f-bump *in*: the ~114K target sits at `f≈0.62`, far below the `f≈0.70` known gen-OOM, so the cheapest path is a **hardware fp16 mem-fraction sweep first** (`f≈0.612 / 0.625 / 0.650`); int8 is built only if that sweep fails (allocator residual / fragmentation / unmodeled gen memory). Page-level reserved for int8 insufficiency.
- **Primary equivalence gate fixed for AC-3.1:** `top-k overlap@2048 ≥ 0.99` vs fp16 (binding); secondary diagnostics (selected-token recall, score-error dist, rank displacement, NIAH trend) recorded-only.
- **Min-reversible-opt-in justification** + caveats (predicted budget; NVML/torch-residual HBM accounting + no-OOM long-generate is the hardware confirmation; conc-64 TTFT may be prefill-bound → admission-vs-prefill attribution required downstream).
- **Claude verification note appended** to the artifact: anchors confirmed verbatim from logs; corrected the Anchor-B label (`≈0.77-0.8` → `≈0.70`, matching the recorded "0.7 OOMs during generation"), which makes the fp16-sweep window conservative; flagged that the linear headroom fit overestimates real headroom by ~3.7 GiB at 396K, so hardware accounting is authoritative.

## Files created/modified
- `runs/20260530_dsv32_loop6/ds_on_v32_decision.md` (new, AC-1 artifact)
- `runs/20260530_dsv32_loop6/footprint_feasibility.md` (new, AC-2 artifact)
- `.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md` (completed truncated IMMUTABLE ACs; populated Active Tasks; logged 2 evolution entries; moved AC-1/AC-2 to Completed-pending-verification)
- `.humanize/rlcr/2026-05-30_06-27-19/round-0-contract.md` (new)
- `.humanize/rlcr/2026-05-30_06-27-19/round-0-summary.md` (this file)

## Tests added/passed
None — this is an analyze-only round (two decision/budget documents). No code, so no unit tests. The selection-equivalence unit test (`top-k overlap@2048 ≥ 0.99`) is **specified** here and will be implemented with the AC-3 footprint code.

## Remaining items (downstream, gated on this round)
- **Next round (AC-3/AC-4 sequencing):** per the AC-2 finding, the next hardware step should **sweep fp16 DS at `f≈0.612/0.625/0.65` first** (cheapest minimum lever) with full HBM accounting + no-OOM long generate (AC-4). Build the int8 compact `TokenLabelTable` (AC-3, flag-gated, fp16 default, CUDA-graph-safe) only if the fp16 sweep fails to admit conc-64 with generation headroom. **Do not** touch the FlashMLA `indices.shape[-1]==dsa_index_topk` assert in any spine work (AC-3.3).
- AC-5 client-SLO benchmark (with admission-vs-prefill attribution), AC-6 opt-in/DSA-default, AC-7/AC-8 hardening (soft), AC-9 within-budget-from-real-tokens (opportunistic), AC-10 recall R&D (gated on AC-1 + landed spine) — all later rounds.

## Note for review
The AC-2 budget makes a non-trivial, evidence-based recommendation that the **no-code fp16 mem-fraction bump may be the true minimum lever** (test-first), while still selecting **int8** as the binding *compaction* lever per the footprint ladder. This refines the AC-4/AC-3 execution order (sweep fp16 first; int8 conditional) and is logged in the Plan Evolution Log — it stays within the plan's Lower/Allowed bounds ("minimum lever", "0.7 acceptable as a conservative first step", "not mem_fraction=0.8 as a number in itself"). Flagged here explicitly for Codex's judgment.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Analyze-only round (two decision/budget documents); no code defect was discovered or solved, so no new lesson is warranted (lessons are added when a problem is solved, not for documentation rounds). Existing lessons were applied as cited context, not modified: BL-20260529-ds-longcontext-needle-recall-vs-topk (kernel-lock + selection-quality gap) grounds AC-1; BL-20260529-ds-vs-dsa-memfraction-admission-asymmetry (the mem-0.6 admission asymmetry, achieved 14.5/24.6/35.7, "0.7 OOMs during generation") grounds AC-2.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
1aa24cfc1 [Sparsity] Loop-6: refined plan v1 + QA ledger + DEC-5 roadmap deferral
88c6498e5 [Sparsity] Loop-6 R0: strategic recall-R&D gate + footprint feasibility budget
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
(first round, no prior history)

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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-0-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
