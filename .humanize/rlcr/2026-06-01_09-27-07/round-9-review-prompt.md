# FULL GOAL ALIGNMENT CHECK - Round 9

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop7/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 9 Summary — Loop 7

## Mainline objective (round-9-contract.md)
**Port the anchor-budget variant to the graph-safe path (AC-3 completion)**: a
tensorized, fixed-shape, alloc-free post-topK force-include for
`anchor_mode ∈ {recency,global,strided}` that is bit-identical to the eager
`_force_include_anchor`, wired into the graph-safe path, with the guard relaxed so
anchor no longer requires `--disable-cuda-graph`.

## Outcome: ACHIEVED — AC-3 variant coverage is complete on the production path.

## Work completed
1. **Tensorized `_force_include_anchor`.** Replaced the per-row Python loop
   (`.item()`, `for b`) with a fully tensorized, fixed-shape, host-sync-free
   implementation: `effective_budget = min(anchor_budget, valid_count, seq_len)`;
   anchor positions via `_anchor_positions_tensor` (recency/global/strided +
   strided ascending set-dedup); evict the k lowest-score non-anchor selected
   (stable score-asc / position-asc tie-break via `_stable_argsort_ascending`);
   insert the first k missing anchors; re-sort. **Bit-identical to the former
   reference — fuzz 2000/2000.** Used by BOTH the eager and graph-safe paths, so
   they cannot diverge.
2. **Graph-safe integration.** `retrieve_topk_graph_safe` runs the force-include
   after the top-K; `anchor_mode`/`anchor_budget` thread through the deepseek_v2
   graph-safe call site and `capture_decode_step`.
3. **Guards relaxed.** `ds_scorer_is_graph_safe()` now returns `True` (every
   non-learned variant is graph-safe); the validator / `_force_eager_select` /
   capture guard no longer force eager for anchor; the serve script only adds
   `--disable-cuda-graph` for the recall-oracle diagnostic.

## Validation (GPU)
- **Eager-vs-graph bit-identical selection** over the full
  `scorer_norm{off,cosine,hybrid} × head_agg{max,mean} × anchor_mode{off,recency,
  global,strided}` matrix (24 combos) on **fp16 + int8**
  (`TestGraphSafeScorerEqualsEager`).
- **Real CUDA-graph capture/replay**: a hybrid+recency-anchor selection captured
  in a `torch.cuda.CUDAGraph` replays **byte-identical to eager + 0 new
  allocations** (`test_anchor_graph_safe_replay_zero_alloc`).
- **TP=8 cross-rank determinism** holds (`test_ds_scorer_tp_determinism.py`).
- Default (anchor off) byte-identical; **346 DS unit tests pass**.

## AC-3 status
All three AC-3 non-learned variants are now flag-gated + graph-safe +
non-regressing: channel-normalization (cosine/hybrid) + head-aggregation [R6] and
anchor-budget (recency/global/strided) [R9]; default byte-identical; within-budget
parity + MMLU ≤1.0pp + binding 16K uplift [R7]; TP=8 equality [R3/R9]. **AC-3
variant coverage complete on the production CUDA-graph path.**

## Files changed
`selection_kernel.py` (tensorized `_force_include_anchor` + `_anchor_positions_tensor`
+ `_stable_argsort_ascending` + graph-safe threading), `cuda_graph.py`,
`validator.py`, `deepseek_v2.py`, `serve_double_sparsity.sh`,
`test_scorer_variants.py` (24-combo matrix + anchor replay no-alloc + guard tests),
`m6_anchor_graphsafe_finding.md` (new), `mmlu_{dsa,default,hybrid}_graph.json`
(data_dir patch). Commit `e7cf1f146` (pushed).

## Remaining items (queued, justified)
- **AC-4 lifted-budget** (task13–17): the opt-in Tier-2.A adjustable-budget decode
  (the oracle gate justifies bounded Tier-2.A) — the next major workstream.
- **AC-6 perf consolidation (conc-1/16 TTFT/decode-TPS/mem) + final strategic-gate
  supersession decision record** (task19–20): the end milestone.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-tensorize-per-row-eviction-for-graph-safe
- Notes: porting a per-row Python eviction/force-include loop to a graph-safe
  tensor op (fixed shape, no `.item()`/host sync) — fuzz it bit-identical against
  the original per-row reference before swapping, and reuse the deterministic
  (score-asc, position-asc) tie-break so the stable Python `list.sort` order is
  reproduced exactly.

## Goal Tracker Update Request
- **AC-3 anchor graph-safe follow-up** → done (R9); **AC-3 variant coverage complete.**
- **Resolve queued** "MMLU data_dir" (patched R9).
- **Keep Active**: AC-4 (task13–17), AC-6/task19–20 (perf + final decision record).
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-06-01_09-27-07/round-8-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-8-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-7-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-7-review-result.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-6-summary.md
- @.humanize/rlcr/2026-06-01_09-27-07/round-6-review-result.md


Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Goal Tracker Audit (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md and verify:

### 1.1 Acceptance Criteria Status
For EACH Acceptance Criterion in the IMMUTABLE SECTION:
| AC | Status | Evidence (if MET) | Blocker (if NOT MET) | Justification (if DEFERRED) |
|----|--------|-------------------|---------------------|----------------------------|
| AC-1 | MET / PARTIAL / NOT MET / DEFERRED | ... | ... | ... |
| ... | ... | ... | ... | ... |

### 1.2 Forgotten Items Detection
Compare the original plan (@development/loop7/refined_plan_v1.md) with the current goal-tracker:
- Are there tasks that are neither in "Active", "Completed", nor "Deferred"?
- Are there tasks marked "complete" in summaries but not verified?
- List any forgotten items found.

### 1.3 Deferred Items Audit
For each item in "Explicitly Deferred":
- Is the deferral justification still valid?
- Should it be un-deferred based on current progress?
- Does it contradict the Ultimate Goal?

### 1.4 Goal Completion Summary
```
Acceptance Criteria: X/Y met (Z deferred)
Active Tasks: N remaining
Estimated remaining rounds: ?
Critical blockers: [list if any]
```

## Part 2: Mainline Drift Audit (MANDATORY)

Determine whether the recent rounds are still serving the original plan:
- Is the current round's mainline objective clear and singular?
- Has Claude been advancing mainline ACs, or mostly clearing side issues?
- Which findings are true **blocking side issues** versus merely **queued side issues**?

Include a short drift summary:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
Blocking Side Issues: N
Queued Side Issues: N
```

The `Mainline Progress Verdict` line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

## Part 3: Implementation Review

- Conduct a deep critical review of the implementation
- Verify Claude's claims match reality
- Identify any gaps, bugs, or incomplete work
- Reference @docs for design documents

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

## Part 5: Progress Stagnation Check (MANDATORY for Full Alignment Rounds)

To implement the original plan at @development/loop7/refined_plan_v1.md, we have completed **10 iterations** (Round 0 to Round 9).

The project's `.humanize/rlcr/2026-06-01_09-27-07/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-06-01_09-27-07/round-8-review-result.md` (previous round)
- `@.humanize/rlcr/2026-06-01_09-27-07/round-7-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-06-01_09-27-07/round-8-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-9-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-9-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
