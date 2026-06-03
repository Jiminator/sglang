# Code Review - Round 6

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-6-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 6 Summary — Loop 7

## Mainline objective (round-6-contract.md)
Port the Tier-2.B scorer (`scorer_norm ∈ {cosine, hybrid}` + `head_agg ∈ {max,
mean}`) into the **graph-safe Triton decode selector (AC-3 landed path)**, with
eager-vs-graph selection-equality evidence on GPU, and relax the
guard/`_force_eager_select` so these variants run under CUDA graph instead of
requiring `--disable-cuda-graph`.

## Outcome: ACHIEVED — winning scorer landed on the production CUDA-graph path, bit-identical to eager.

## Work completed
1. **Triton scorer port.** `_logical_score_kernel` gains 3 `tl.constexpr`
   (`SCORER_NORM` 0/1/2, `HEAD_AGG_MEAN`, `HYBRID_THRESHOLD`): cosine =
   unit-normalized dot (scale-ignored, normalize-then-sum to match eager); hybrid
   = per-request `seq_len > threshold` switch read in-kernel; head_agg mean =
   sum-then-divide. R17 early-exit + int8 dequant preserved; default (off/max)
   byte-identical.
2. **Config-borne threading.** Flags flow through `_logical_score_triton`,
   `retrieve_topk_graph_safe` (+ its fallback), the deepseek_v2 graph-safe call
   site, and `capture_decode_step`.
3. **Guard relaxation.** New `ds_scorer_is_graph_safe(config)` (= `anchor_mode ==
   "off"`); the validator, `_force_eager_select`, and the capture guard now only
   force eager for a non-default `anchor_mode`. cosine/hybrid/head_agg run under
   CUDA graph. serve script no longer auto-adds `--disable-cuda-graph` for them.
4. **Eager-vs-graph equality (GPU).** `TestGraphSafeScorerEqualsEager`: the
   graph-safe Triton scorer produces **bit-identical** `selected_indices` +
   `valid_lengths` to the eager `retrieve_topk_via_labels` for all 12 combos
   `scorer_norm{off,cosine,hybrid} × head_agg{max,mean}` on **fp16 AND int8**,
   short/long requests crossing the hybrid threshold.
5. **Live under CUDA graph.** A `scorer_norm=hybrid` server boots with CUDA graph
   ON (`Capture cuda graph begin` on all 8 TP ranks; validator allowed it) —
   previously impossible.

## Validation
- **345 DS unit tests pass** (incl. the new GPU eager-vs-graph equality test, the
  graph-safe guard/predicate tests, the existing CUDA-graph 100-step replay, and
  the TP=8 determinism matrix).
- **Production (graph-mode) recall, N=20** (`niah_ds_hybrid_graphsafe.json`):
  hybrid 16K **25% [8.7,49.1]** vs graph default 5% [0.1,24.9] (+20 pp, marginally
  material); 4K 75% == default (the ≤8192 raw regime is identical to default, as
  designed); 1024w 100% parity.

## Honest correction (good rigor)
R5's eager-mode numbers (hybrid 4K=85%, 16K=40%) were measured with
`--disable-cuda-graph`. The graph-safe production path gives 75% / 25%. The
*scorer code* is bit-identical eager-vs-graph (proven), so the gap is **upstream
eager-vs-graph model-forward numerics** (the query projection feeding the scorer
shifts a few needles under CUDA-graph capture; it affects the default too). The
**binding production recall is the graph-mode number**; the eager research number
over-stated it. This is exactly why AC-3 requires a *landed* graph-safe path.

## Files changed
`selection_kernel.py` (kernel + threading + `ds_scorer_is_graph_safe`),
`cuda_graph.py` (guard + capture call), `validator.py` (guard relax),
`deepseek_v2.py` (`_force_eager_select` + graph-safe call site),
`serve_double_sparsity.sh`, `test_scorer_variants.py` (GPU equality + guard +
predicate tests), `m3_graphsafe_scorer_finding.md` (new),
`niah_ds_hybrid_graphsafe.json` (new). Commit `cb02b6673` (pushed).

## Remaining items (queued, justified) — task #15
- **N≥50 binding 16K** at the graph op-point (firm up the marginal 25%).
- **MMLU ≤1.0pp re-anchor** (single-node mem0.7), DSA vs DS-hybrid.
- **graph-vs-eager perf delta** (AC-6, conc-1/16) now that hybrid runs under graph.
- **anchor_mode graph-safe port** (still eager-only).
- **AC-4 lifted-budget** (task13–17), **AC-6 consolidation + final decision
  record** (task19–20).
- R5 evidence-label cleanup (DSA op-point label, materiality wording): queued.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-eager-vs-graph-recall-differs-despite-identical-scorer
- Notes: even with a bit-identical eager-vs-graph SCORER (proven), the served
  recall differs between an eager (`--disable-cuda-graph`) server and a CUDA-graph
  server because upstream model-forward numerics (the query projection feeding the
  scorer) differ under capture. Production recall MUST be measured on the
  graph-safe path; an eager research measurement can over-state it.

## Goal Tracker Update Request
- **task8** (AC-3): graph-safe scorer support DONE (R6) — bit-identical
  eager-vs-graph selection, serves under CUDA graph. Mark done.
- **task12** (AC-2,AC-3): graph-safe port covered; remaining = N≥50 16K + MMLU +
  perf (task #15).
- **Keep Active**: task #15 (AC-3 measurement matrix + anchor port), AC-4
  (task13–17), AC-6 (task19–20).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-5-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-5-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-4-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-4-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-3-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-3-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-6-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
