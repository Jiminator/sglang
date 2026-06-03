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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-3-contract.md

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
# Round 2 Review Result

Mainline Progress Verdict: ADVANCED

Claude made real Round-2 progress: the physical `hybrid` misuse is now rejected, the production CUDA-graph runner is protected by startup validation, variant launcher knobs are exposed, and `anchor_mode` plus a first TP-shaped gloo test exist. However, the work is not complete against `development/loop7/refined_plan_v1.md`, and two Round-2 claims do not match the implementation evidence: anchor budget clamping is wrong for over-budget anchors, and TP determinism coverage is not the claimed scorer/head/anchor matrix.

Do not stop the loop. Several original-plan tasks are still pending or deferred, and the Round-2 anchor/TP issues must be fixed before task10/task11 can be marked complete.

## Mainline Gaps

1. **AC-3/task11: TP determinism is only partially tested, not the required scorer/head/anchor matrix.**

   Evidence: `test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py:20` sets `WORLD = 2`, while the original plan requires TP=8 selected-index equality. The test methods cover only five hand-picked combinations (`:92-105`): `off/max/off`, `cosine/max/off`, `off/mean/off`, `off/max/recency`, and `cosine/mean/strided`. It omits `scorer_norm="hybrid"`, `anchor_mode="global"`, most `head_agg × anchor_mode` combinations, and the production logical selector wiring. The worker calls `compute_token_scores` + `select_topk_sequence_order` directly (`:60-69`) instead of exercising `retrieve_topk_via_labels` with `req_pool_indices`, `req_to_token`, `seq_lens`, config threading, and anchor application in the same path production uses.

   Required fix: replace the five-case test with a real parameterized matrix. Use logical-mode inputs and call `retrieve_topk_via_labels` or `DoubleSparsitySelector.retrieve_topk` so `hybrid`, `head_agg`, `anchor_mode`, `anchor_budget`, `seq_lens`, `req_to_token`, and all-reduce are tested together. Cover `scorer_norm={off,cosine,hybrid}`, `head_agg={max,mean}`, and `anchor_mode={off,recency,global,strided}`. Keep the fast 2-rank CPU/gloo test for unit signal if needed, but task11 cannot be closed until there is TP=8 equality evidence for the full matrix or a documented TP=8 hardware test artifact.

2. **AC-3/task12: AC-3 remains an eager/safety result, not a production-ready selector with non-regression evidence.**

   Evidence: the new startup guard in `validator.py:96-115` explicitly rejects non-default variants under CUDA graph instead of making them graph-safe. `deepseek_v2.py:2245-2248` still routes any non-default scorer to the eager selector. That is a correct safety stopgap, but it means the graph-safe Triton scorer port, MMLU re-anchor, dense-DS/within-budget parity, DSA same-node reference, N>=50 binding 16K, 64K, per-variant attribution, and eager-vs-graph perf evidence are still absent. Claude's own summary defers this matrix.

   Required fix: port the winning non-learned variants into `retrieve_topk_graph_safe`/Triton, including scorer normalization, head aggregation, and anchor handling without capture-time allocation or host sync. Then run the full task12 matrix: DSA same-node reference, DS baseline, each independent variant, 1K/1.5K/4K/16K/64K NIAH with N>=50 for binding 16K, MMLU at mem0.7, dense-DS, within-budget parity, TP=8 selected-index equality, and eager-vs-graph perf deltas.

3. **AC-1/AC-2: Oracle fail-closed and 64K oracle evidence are still deferred.**

   Evidence: `_maybe_record_recall_oracle()` still returns silently when no active trial exists (`selection_kernel.py:972-975`), filters the harness-provided needle span instead of rejecting out-of-range positions (`:981-985`), and swallows all exceptions (`:1004-1006`). This is the same fail-open behavior that previously produced missing 64K oracle records. Round 2 did not change it.

   Required fix: make oracle-enabled mode fail closed. Validate active trial state and the full harness-provided needle span, remove span filtering, emit explicit failure artifacts keyed by request/trial/layer/decode-step, and add expected-record-count assertions to the oracle sweep. Re-run 4K/16K/64K oracle sweeps before treating task7/AC-2 attribution as binding.

4. **AC-4/task13-task17: Tier-2.A is still unimplemented despite the 4K oracle gate.**

   Evidence: repository search finds no implementation of `enable_lifted_budget_decode` or `lifted_budget_top_k` outside plan/review text. No compact-domain remap path, sparse decode path, correctness tests, or landing/disposition record exists. The original plan makes task13-task17 conditional on the oracle-uplift gate, and the earlier M0 evidence says 4K score-only recall@4096 recovers the needle.

   Required fix: execute task13-task17 exactly as planned. Add the explicit lifted-budget ABI and validators; reject `top_k > index_topk` unless the opt-in backend is selected; implement physical slot -> `page_table_1_flattened` -> compact-index remap; mask `-1` before dequant; use fixed `lifted_budget_top_k` with padding; preserve the R23 tie-break; add reference-attention, prefix-sharing, padding, duplicate, valid-length, graph-replay allocation, and TP equality tests; then write the production-ready-or-disposition record.

5. **AC-6/task19-task20: consolidation, perf guardrails, and final decision record remain missing.**

   Evidence: no Round-2 artifact records conc-1/16 TTFT, decode TPS/req, GPU memory, graph replay success, admission, or Tier-1 spine non-regression for the scorer variants or any lifted-budget path. The tracker still has task19/task20 pending.

   Required fix: after task12 and task17 are done, run the conc-1/16 guardrail suite and write the final DS-vs-DSA recall/perf/non-regression report. Then write the strategic-gate supersession decision record that cites the corrected M0 evidence and final disposition.

## Blocking Side Issues

1. **Anchor force-include mishandles `anchor_budget > top_k`, so recency/strided semantics are wrong.**

   Evidence: `_anchor_positions()` clamps `budget` only to `seq_len` (`selection_kernel.py:756-767`), and `_force_include_anchor()` protects every selected token already in the oversized anchor set before choosing missing anchors (`:792-805`). With `top_k=3`, `seq_len=8`, and early tokens initially selected, review reproduction showed:

   ```text
   budget 5 recency anchors [3, 4, 5, 6, 7] forced [3, 4, 5]
   budget 5 strided anchors [0, 2, 4, 5, 7] forced [0, 2, 4]
   budget 10 recency anchors [0, 1, 2, 3, 4, 5, 6, 7] forced [0, 1, 2]
   ```

   Recency with only three selectable anchor slots should force the most recent three positions (`[5,6,7]`), not `[3,4,5]` or unchanged early positions. Strided should regenerate/clamp to three evenly-spaced anchors over `[0, seq_len)`, not take the first three positions of an oversized strided set. The claimed "budget > top_k handling" test is not present in `test_scorer_variants.py`; the current tests cover budget > seq_len for `_anchor_positions` but not force-inclusion with `top_k < anchor_budget`.

   Required fix: compute `effective_anchor_budget = min(anchor_budget, len(real), n)` before generating anchors in `_force_include_anchor`. Generate anchors from that effective budget, not from the raw budget. For recency this yields the most recent effective positions; for global the earliest effective positions; for strided the effective evenly-spaced positions. Add regression tests for recency and strided with `anchor_budget > top_k`, including the `anchor_budget >= seq_len` case.

## Queued Side Issues

1. **Plan-specific markers remain in code/tests.**

   Examples remain in `test_ds_scorer_tp_determinism.py:1` and `test_scorer_variants.py:1`. This violates the implementation-note cleanup rule but does not block the next correctness round.

## Goal Alignment Summary

ACs: 5/6 addressed | Forgotten items: 0 | Unjustified deferrals: 5

AC-1, AC-2, AC-3, AC-5, and AC-6 have some progress, but only AC-5 is effectively complete. AC-4 remains unimplemented. The unjustified deferral buckets are: oracle fail-closed/64K evidence, graph-safe scorer port + full AC-3 matrix, Tier-2.A task13-task17, M4 perf/consolidation task19, and task20 final decision record.

## Tracker Update

I updated `.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md` mutable section only:

- accepted the Round-2 production CUDA-graph startup guard as resolving the Round-1 blocking side issue;
- accepted the physical-path `hybrid` reject as a completed task8 subitem;
- accepted the launcher variant knobs and removed that queued issue;
- rejected the requested task10 completion because over-budget anchor force-inclusion is wrong;
- rejected the requested task11 completion because coverage is TP=2, partial, and missing `hybrid`/`global`/full logical-path matrix;
- added the over-budget anchor bug as a blocking AC-3 side issue.

## Required Next Implementation Plan

1. Fix `_force_include_anchor` to clamp the effective anchor budget to the current selected count before generating anchors, and add recency/strided `anchor_budget > top_k` regression tests.
2. Replace the TP determinism test with a parameterized logical-path matrix over `scorer_norm × head_agg × anchor_mode`, including `hybrid` and `global`; add or record TP=8 equality evidence before closing task11.
3. Make the oracle fail closed and re-run 4K/16K/64K oracle sweeps with expected-record-count checks.
4. Port scorer/head/anchor variants into the graph-safe path and prove graph replay allocation safety.
5. Run the full task12 non-regression/measurement matrix.
6. Execute task13-task17 for lifted-budget decode and write the landing/disposition record.
7. Finish task19/task20 consolidation and final decision record.

## Verification Run During Review

Targeted tests passed:

`python3 -m pytest test/registered/unit/layers/attention/test_scorer_variants.py test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py -q`

Result: 18 passed, 5 warnings.

Not complete: original-plan tasks and Round-2 anchor/TP gaps remain.
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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-3-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-3-summary.md

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
