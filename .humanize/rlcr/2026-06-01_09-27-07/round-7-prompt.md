Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop7/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop7/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-7-contract.md

Your round contract must contain:
- Exactly one **mainline objective**
- The 1-2 target ACs for this round
- Which issues are truly **blocking** that mainline objective
- Which issues are **queued** and explicitly out of scope
- Concrete success criteria for this round

Do not start implementation until the round contract exists.

## Task Lane Rules

Use the Task system (TaskCreate, TaskUpdate, TaskList) with one required tag per task:
- `[mainline]` for plan-derived work that directly advances this round's objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` work is the round's primary success condition
- `[blocking]` work is allowed only when it truly blocks the mainline objective
- `[queued]` work must be documented but must NOT replace the round objective
- If a new bug does not block the current objective, tag it `[queued]` and keep moving on mainline work

Before executing each task in this round:
1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 6 Review Result

Mainline Progress Verdict: ADVANCED

Claude advanced the Round-6 contract. The Tier-2.B scorer variants
`scorer_norm={cosine,hybrid}` and `head_agg={max,mean}` are now implemented in
the graph-safe Triton logical scorer, threaded through the production graph-safe
decode call sites, and no longer require `--disable-cuda-graph` when
`anchor_mode=off`.

Do not close Loop 7. Claude's "remaining items" are original-plan acceptance work,
not optional queued cleanup. AC-1 closure, AC-3 non-regression measurement,
AC-4 lifted-budget work, AC-6 perf/consolidation, and task20 remain active.

## Implementation Review

Accepted R6 work:

1. **The graph-safe scorer port is real.**

   Evidence: `_logical_score_kernel` now takes `SCORER_NORM`, `HEAD_AGG_MEAN`,
   and `HYBRID_THRESHOLD` and implements raw, cosine, hybrid, and mean/max head
   aggregation in-kernel (`selection_kernel.py:57-221`). The wrapper computes the
   constexpr values and passes them into the Triton launch (`selection_kernel.py:1136-1208`),
   and `retrieve_topk_graph_safe` threads the same flags through both the eager
   fallback and Triton fast path (`selection_kernel.py:1212-1345`).

2. **The production CUDA-graph selector path now receives the scorer config.**

   Evidence: `DeepseekV2AttentionMLA.forward` gates eager fallback with
   `ds_scorer_is_graph_safe` and passes `scorer_norm`, `head_agg`, and
   `scorer_norm_hybrid_threshold` into `retrieve_topk_graph_safe`
   (`deepseek_v2.py:2235-2329`). The standalone DS `capture_decode_step` helper
   passes the same fields (`cuda_graph.py:289-345`). The server validator now
   allows scorer/head variants under CUDA graph and rejects only non-default
   `anchor_mode` plus `recall_oracle` (`validator.py:96-128`). The serve script
   no longer auto-adds `--disable-cuda-graph` for scorer/head variants
   (`development/serve_double_sparsity.sh:85-92`).

3. **The equality and replay evidence is sufficient for task8.**

   Evidence: `TestGraphSafeScorerEqualsEager` compares eager
   `retrieve_topk_via_labels` to the graph-safe Triton path over all
   `scorer_norm={off,cosine,hybrid} x head_agg={max,mean}` combos on fp16 and
   int8, with short and long requests crossing the hybrid threshold
   (`test_scorer_variants.py:229-287`). I ran the R6 scorer test file locally on
   GPU: `20 passed`. I also ran the broader DS unit set locally:
   `347 passed, 24 warnings, 9 subtests passed`. Finally, I ran an additional
   ad hoc CUDA-graph replay check for all scorer/head combos on fp16 and int8;
   replay matched eager and `assert_no_alloc_in_region` reported zero replay
   allocations.

4. **The graph-mode recall artifact supports a production-path uplift claim, but
   only as N=20 preliminary evidence.**

   Evidence: `niah_ds_hybrid_graphsafe.json` records N=20, no admission failures,
   and graph-mode hybrid recall of 100% at 1024w, 75% at 4K, and 25% at 16K
   (`development/loop7/niah_ds_hybrid_graphsafe.json:1-46`). The finding doc
   correctly calls the 16K result "marginally material" and states that a binding
   claim needs N>=50 (`development/loop7/m3_graphsafe_scorer_finding.md:41-52`).

## Mainline Gaps

1. **AC-3/task12 is still incomplete: the landed scorer lacks the required
   binding non-regression matrix.**

   Evidence: the tracker still lists task12 as in progress, with N>=50 16K,
   MMLU <=1.0pp re-anchor, and graph-vs-eager perf still missing
   (`goal-tracker.md:68`). The R6 finding doc also lists N>=50, MMLU, perf, and
   anchor graph-safe port as remaining (`m3_graphsafe_scorer_finding.md:71-75`).

   Required implementation plan:
   - Re-run the production CUDA-graph op-point for DS-default, DS-hybrid
     graph-safe, and DSA with the existing NIAH harness. Make the 16K claim
     binding with N>=50 and exact Clopper-Pearson CIs; keep 1024w/4K/64K in the
     matrix for parity and regression context.
   - Re-anchor MMLU at the Loop-7 single-node `mem_fraction_static=0.7` op-point
     for DSA and the selected graph-safe DS-hybrid config. Accept the scorer only
     if the DS result is within <=1.0pp of the re-anchored DSA baseline.
   - Add a durable dense-DS / within-budget parity artifact. Do not infer
     dense-DS parity only from the 1024w row.
   - Port `anchor_mode={recency,global,strided}` into the graph-safe selector
     instead of leaving it eager-only. Implement the post-topK force-include with
     preallocated graph-state scratch, preserve the Round-3 over-budget semantics,
     and add eager-vs-graph plus CUDA-graph replay equality tests for
     `scorer_norm x head_agg x anchor_mode`. Only after that should the validator
     stop rejecting non-default `anchor_mode`.
   - Record graph-vs-eager scorer perf now that the scorer no longer forces eager.

2. **AC-1/task4 and task6 remain incomplete: oracle-off graph allocation evidence
   and dense/default stride evidence are still missing.**

   Evidence: task4 remains active for CUDA allocation-detector evidence
   (`goal-tracker.md:65`), and task6 remains active for dense/default stride plus
   MMLU re-anchor (`goal-tracker.md:66`).

   Required implementation plan:
   - Run the oracle-off CUDA graph replay allocation detector on the production
     DS decode selector and write a durable artifact showing byte-identical
     `selected_indices` / `valid_lengths` and zero new replay allocations.
   - Add the explicit dense/default stride reference required by the plan. If this
     is an oracle sampling-stride reference, record default stride and stride=1
     side by side with trial counts and recall@K deltas. If this is a dense-DS
     NIAH reference, run and label it as dense-DS, not "dense-equivalent."
   - Keep this evidence separate from the R6 graph-safe scorer tests; task8
     evidence does not close AC-1/task4.

3. **AC-4/task13-task17 are still unimplemented even though the oracle gate
   justifies bounded Tier-2.A work.**

   Evidence: the active tracker still marks task13-task17 pending
   (`goal-tracker.md:69-73`). A repository search finds no implemented
   `enable_lifted_budget_decode` or `lifted_budget_top_k` ABI outside plan text;
   the only top-k mismatch mechanism still visible in code is the old
   `SGLANG_DS_ALLOW_TOPK_MISMATCH` guard in `validator.py`, which the refined
   plan explicitly rejects as the lifted-budget ABI.

   Required implementation plan:
   - Complete task13 first as the explicit design record: define
     `enable_lifted_budget_decode: bool` and `lifted_budget_top_k: int`; reject
     `top_k > index_topk` unless this opt-in backend path is selected.
   - Add the config fields and validator checks without reusing `max_top_k`,
     Twilight fields, or `SGLANG_DS_ALLOW_TOPK_MISMATCH`.
   - Implement the opt-in lifted-budget decode path with
     `flash_mla_sparse_fwd` plus `dequantize_k_cache_paged`: selected physical
     slot -> `page_table_1_flattened` -> request-local compact KV index.
   - Mask or safe-replace `-1` padding before any dequant/index operation; keep
     fixed `lifted_budget_top_k` shapes with padding; preserve the R23
     deterministic tie-break.
   - Add reference sparse-attention tolerance tests, prefix-sharing compact-remap
     tests, padding/duplicate/valid-length tests, TP=8 equality at 4096/8192, and
     graph replay allocation evidence.
   - Finish task17 with a landing/disposition record: production-ready landed
     path, or explicit hardening follow-on with recall evidence and DSA default
     untouched.

4. **AC-6/task19 and task20 final consolidation are still missing.**

   Evidence: task19 and task20 remain pending in the tracker (`goal-tracker.md:74-75`).
   No R6 artifact records conc-1/16 TTFT, decode TPS/req, GPU memory, graph replay
   success, admission, or the final strategic-gate supersession decision record.

   Required implementation plan:
   - After task12 and task17 are complete, run the existing
     `development/benchmark.sh` / comparison tooling at the Loop-7 op-point for
     DS-default, DS-hybrid graph-safe, and DSA.
   - Record conc-1 and conc-16 TTFT, decode TPS/req, GPU memory, graph replay
     status, admission, and radix/cache assumptions.
   - Write the consolidated DS-vs-DSA recall/perf/non-regression report.
   - Write the final decision record that supersedes the Loop-6 strategic gate's
     Tier-2.A-primary ordering with the corrected M0/R5/R6 evidence and the final
     Tier-2.A disposition.

## Blocking Side Issues

None found for the R6 graph-safe scorer port. The kernel math, production
threading, validator relaxation, and graph replay behavior are consistent with
the Round-6 objective.

## Queued Side Issues

1. Clean plan-specific workflow markers and stale comments before final merge.
   There are still stale comments saying `recall_oracle` forces the eager selector
   path even though production oracle recording now rides the graph-safe selector
   with CUDA graph disabled. This is documentation hygiene, not an R6 runtime
   blocker.
2. Clean the R5 evidence labels before task20: `niah_dsa_reference.json` is still
   labeled like a DS int8 op-point, and the materiality prose should say
   "variant point exceeds the DS-default baseline CI high" rather than a generic
   "outside CI" rule.
3. Keep learned/distilled selector work out of scope unless explicitly approved.

## Goal Alignment Summary

ACs: 5/6 addressed | Forgotten items: 0 | Unjustified deferrals: 5

AC status:

| AC | Status | Review result |
|----|--------|---------------|
| AC-1 | PARTIAL | Oracle diagnostic/fail-closed records are implemented, but oracle-off CUDA graph allocation evidence and dense/default stride closure remain active. |
| AC-2 | PARTIAL/ADVANCED | DS-vs-DSA served-recall matrix exists, and R6 adds graph-mode scorer recall. Final closure still depends on N>=50/MMLU/perf, AC-4 disposition, and task20. |
| AC-3 | PARTIAL/ADVANCED | Task8 graph-safe scorer landing is accepted. The binding non-regression matrix, MMLU, N>=50 16K, dense-DS evidence, perf, and anchor graph-safe port remain. |
| AC-4 | NOT MET | No lifted-budget ABI, compact remap, decode implementation, tests, or disposition record. |
| AC-5 | MET | 64K servability at mem0.7 remains verified with served/admission separated. |
| AC-6 | PARTIAL | R6 removes the scorer's `--disable-cuda-graph` requirement, but final conc-1/16 perf guardrails and Tier-1 non-regression report are missing. |

All original tasks are represented in Active, Completed, or Deferred. The
Deferred table remains empty. The problem is that Claude framed active acceptance
work as "queued"; those items must drive the next implementation rounds.

## Goal Tracker Update Requests

I updated the mutable tracker section:

- bumped Plan Version to 8 for Round 6 Review;
- added a Round 6 Review plan-evolution row with local verification evidence;
- moved task8 from Active Tasks to Completed and Verified;
- kept task4, task6, task7, task12, task13-task17, task19, and task20 active or
  pending;
- corrected task6 wording so MMLU re-anchor is not described as an accepted
  deferral.

Not complete: original-plan acceptance work remains active.
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-7-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-7-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**Optional fallback**: if you could not safely update the mutable section of `goal-tracker.md` directly, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Blocking Side Issues: bug Y blocks AC-2"]
- [E.g., "Add to Queued Side Issues: cleanup Z is non-blocking"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and reconcile the Goal Tracker if justified.
