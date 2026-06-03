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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-2-contract.md

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
# Round 1 Review Result

Mainline Progress Verdict: ADVANCED

Claude advanced the AC-3 selector mainline by landing and measuring a length-conditional hybrid scorer on the eager path: the committed artifact reports 4K 17/20 (85%) and 16K 8/20 (40%) at the Loop-7 int8/mem0.7 op-point. This is useful progress and fixes the uniform-cosine 4K regression as a research signal.

The work is still incomplete against `development/loop7/refined_plan_v1.md`. The round summary correctly admits several items are "remaining", but those are plan-derived acceptance work, not optional follow-up. AC-3 is not closed, AC-4 has not started despite the 4K oracle gate, and AC-1/AC-2/AC-6 still have binding evidence gaps.

## Mainline Gaps

1. **AC-3: The hybrid is measured only as an eager-path candidate, not a production-ready selector.**

   Evidence: `development/loop7/m1_hybrid_finding.md:18-20` explicitly says the hybrid result is eager-path only and still needs graph-safe Triton support, MMLU re-anchor, dense-DS/within-budget parity, N>=50, DSA same-node reference, TP=8 determinism, and per-variant measurement. The artifact itself only covers 4K and 16K with N=20 (`development/loop7/recall_hybrid.json:4-31`). This does not satisfy AC-3's non-regression matrix or TP=8 equality bar.

   Required fix: complete task11/task12 before any AC-3 closure claim. Port the winning scorer into the graph-safe path, then run the full matrix: DSA same-node reference, DS baseline, hybrid and each independent variant, 1K/1.5K/4K/16K/64K NIAH with N>=50 for binding 16K, MMLU at mem0.7, dense-DS, within-budget parity, TP=8 cross-rank selected-index equality, and eager-vs-graph perf deltas.

2. **AC-3/task8: `scorer_norm=hybrid` is not length-conditional in the physical score path.**

   Evidence: `compute_token_scores()` treats `hybrid` as `cosine` unconditionally because it sets `cosine_like = norm_mode in ("cosine", "hybrid")` and has no `seq_len`/threshold input (`selection_kernel.py:417-453`). That contradicts the Round 1 contract's success criterion that hybrid be per-request raw<=threshold/cosine>threshold in both eager logical and physical scorer paths. Any caller using physical-domain scoring with `hybrid` gets the exact 4K regression path that hybrid was supposed to avoid.

   Required fix: make physical-domain hybrid impossible to misapply. Add `seq_lens` and `hybrid_threshold` to the physical scorer path and branch per request, or reject `scorer_norm="hybrid"` when logical request lengths are unavailable. Add a regression test where a short-context physical call would choose raw over cosine.

3. **AC-3/task10: The anchor-budget ablation is incomplete.**

   Evidence: the config has only `anchor_budget: int` and describes only most-recent recency slots (`config.py:47-51`); the implementation only calls `_force_include_recency_anchor()` (`selection_kernel.py:740-779`, `:870-873`). The original plan requires deterministic anchor-budget ablations covering recency/global/strided, independently flag-gated.

   Required fix: add an explicit `anchor_mode` config with `off|recency|global|strided`, keep `anchor_budget`, and implement one deterministic anchor generator used by eager and graph-safe paths. Recency selects the most recent logical positions, global selects the earliest stable global positions, and strided selects evenly spaced positions over `[0, seq_len)`. Preserve exactly `top_k` selected entries by evicting the lowest-scoring non-anchor entries, then add tests for all modes, budget > top_k handling, short sequences, duplicate prevention, and selected-index ordering.

4. **AC-3/task11: TP determinism coverage for the new flags is still missing.**

   Evidence: the new `test_scorer_variants.py` is explicitly CPU-only (`test_scorer_variants.py:1-4`) and contains no process group/all-reduce or rank parameterization. The existing TP test is the old placeholder/default two-rank synthetic check (`test_double_sparsity_unit.py:4662-4736`) and does not exercise `scorer_norm`, `head_agg`, or `anchor_budget`.

   Required fix: add a parameterized TP-shaped test over `scorer_norm={off,cosine,hybrid}`, `head_agg={max,mean}`, and each anchor mode. Use divergent per-rank local scores, run the same all-reduce/top-k path, and assert identical `selected_indices` and `valid_lengths` on every rank. Keep the `DoubleSparsityTPMisconfigured` and `DoubleSparsityRebindError` fail-fast tests in the same matrix.

5. **AC-1/AC-2: The oracle fail-closed and 64K re-run work was deferred but remains required.**

   Evidence: `_maybe_record_recall_oracle()` still silently returns when the oracle is enabled but no active trial exists, filters out-of-range needle spans, and swallows all exceptions (`selection_kernel.py:925-973`). The M0 artifact still admits 64K records are absent and infers the 64K scorer-limited verdict (`development/loop7/m0_oracle_finding.md:9`, `:24-26`).

   Required fix: make oracle-enabled mode fail closed. Validate active trial state and the full harness-provided needle span, record explicit failure artifacts keyed by request/trial/layer/decode-step, remove span filtering, and add expected-record-count assertions in the oracle sweep. Re-run 4K/16K/64K oracle sweeps with no missing lengths before binding task7/AC-2 conclusions.

6. **AC-4/task13-task17: Tier-2.A is still not started even though the 4K oracle gate was met.**

   Evidence: `m0_oracle_finding.md:13` says 4K score-only recall@4096 materially recovers the needle, and the plan makes task13-task17 conditional on that gate. No `enable_lifted_budget_decode` / `lifted_budget_top_k` ABI, compact remap, sparse decode path, correctness tests, or landing/disposition record exists.

   Required fix: execute task13-task17 exactly as planned. Add the explicit lifted-budget ABI and validators; implement physical slot -> `page_table_1_flattened` -> compact-index remap; mask `-1` before dequant; use fixed `lifted_budget_top_k` with padding; preserve the R23 tie-break; add reference-attention, prefix-sharing, padding, duplicate, valid-length, graph-replay, and TP equality tests; then write the production-ready-or-disposition record.

7. **AC-6/M4: Consolidation and perf guardrails are absent.**

   Evidence: no Round 1 artifact records conc-1/16 TTFT, decode TPS/req, GPU memory, graph replay success, admission, or Tier-1 spine non-regression for the hybrid or any lifted-budget path. Task19/task20 remain pending in the tracker.

   Required fix: after the selector and Tier-2.A disposition are complete, run the conc-1/16 guardrail suite and write the final DS-vs-DSA recall/perf/non-regression report plus the strategic-gate supersession decision record.

## Blocking Side Issues

1. **The graph-capture fix does not cover the production CUDA graph runner.**

   Evidence: Round 1 added a fail-fast guard only in `capture_decode_step()` (`cuda_graph.py:295-309`). Production CUDA graph capture uses `cuda_graph_runner._capture_graph()` to capture the full model forward (`cuda_graph_runner.py:901-903`, `:1110-1115`). In that real forward path, a non-default scorer sets `_force_eager_select=True` (`deepseek_v2.py:2245-2248`) and then calls `selector.retrieve_topk()` (`deepseek_v2.py:2319-2328`) instead of failing before capture. That eager selector can allocate and host-sync inside capture; it is not the clear "serve with --disable-cuda-graph" failure claimed by the summary.

   Required fix: add a production-path guard before graph capture/replay for DS non-default scorer configs. The definitive implementation is to port hybrid/head/anchor scoring into `retrieve_topk_graph_safe()` and pass the config through; until that is landed, startup or capture must reject `ds_scorer_is_default(config)==False` when CUDA graph is enabled. Add a unit/integration test that simulates normal graph capture metadata, not only `capture_decode_step()`.

## Queued Side Issues

1. **The launcher does not expose the new variant knobs.**

   `development/serve_double_sparsity.sh` now pins the Loop-7 op-point correctly, but the JSON only includes `scorer_norm` (`serve_double_sparsity.sh:69-74`). `head_agg`, `anchor_budget`, and `scorer_norm_hybrid_threshold` require overriding the whole JSON through extra args, which is brittle for the required task12 per-variant matrix. Add launcher env knobs before running that matrix.

2. **Plan-specific markers remain in production/test code.**

   Examples: `selection_kernel.py:346` and `test_scorer_variants.py:1-3` still include Loop/AC wording. This does not block recall work, but it should be cleaned before final merge.

## Goal Alignment Summary

ACs: 5/6 addressed | Forgotten items: 0 after tracker update | Unjustified deferrals: 6

AC-1, AC-2, AC-3, AC-5, and AC-6 have partial or completed progress, but only AC-5 is effectively complete. AC-4 remains unimplemented despite the 4K oracle gate. The unjustified deferral buckets are: oracle fail-closed/64K evidence, graph-safe scorer port, full AC-3 validation matrix, full anchor-mode ablation, Tier-2.A task13-task17, and M4/perf consolidation.

## Tracker Update

I updated `.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md` mutable section only:

- recorded Round 1 hybrid as useful eager-path progress, not AC-3 closure;
- accepted the `LOOP7_MEASUREMENT=1` launcher fix and removed the old op-point mismatch from blocking issues;
- kept graph capture as blocking because production CUDA graph capture is still not covered;
- marked task9 head aggregation as implemented/unit-tested, with TP/matrix evidence still under task11/task12;
- kept task8 active due physical-path hybrid mismatch, graph-safe gap, and missing binding matrix;
- kept task10 active because only recency anchoring exists, not the original recency/global/strided ablation;
- added the launcher variant-knob limitation as queued.

## Verification Run During Review

Targeted local tests passed:

`python3 -m pytest test/registered/unit/layers/attention/test_scorer_variants.py test/registered/unit/layers/attention/test_double_sparsity_unit.py::TestDoubleSparsityConfigParser test/registered/unit/layers/attention/test_double_sparsity_unit.py::TestCUDAGraphCapture -q`

Result: 27 passed, 21 warnings.
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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-2-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-2-summary.md

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
