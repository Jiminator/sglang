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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-9-contract.md

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
# Round 8 Review Result

Mainline Progress Verdict: ADVANCED

Round 8 advanced the stated round contract: AC-1's remaining oracle-off replay
allocation evidence and dense/default stride reference are now present and
locally verified. Do not close Loop 7. Claude's "remaining items" are still
original-plan acceptance work, not optional queue cleanup: AC-3 anchor graph-safe
support, AC-4 lifted-budget decode, AC-6 perf/consolidation, and task20 remain
active.

## Implementation Review

Accepted R8 work:

1. **AC-1 oracle-off byte-equivalence + zero-allocation evidence is real.**

   Evidence: `development/loop7/oracle_off_graph_replay_alloc.json` records
   `recall_oracle: false`, CUDA graph capture/replay, 120 replay steps,
   byte-identical selected indices and valid lengths, `replay_allocation_delta_bytes: 0`,
   and `verdict: PASS`. I reran `python development/loop7/oracle_off_replay_alloc.py`;
   it regenerated the same PASS artifact. I also ran the new targeted GPU test:
   `pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py::TestCUDAGraphCapture::test_oracle_off_replay_byte_identical_and_zero_alloc`
   -> `1 passed`.

2. **The stride=1 dense oracle reference is sufficient for AC-1.**

   Evidence: `development/loop7/oracle_stride_reference.json` records
   `emitted_stride_value_counts: {"1": 14640}`, `records_success: 14640`, and
   `default_equals_stride1: true`. The implementation call site hardcodes
   `oracle_payload_for_row(..., stride=1)` (`selection_kernel.py` around the
   oracle hook), so the "default stride" is the dense stride. I reran
   `python development/loop7/oracle_stride_reference.py`; it regenerated the
   committed JSON with no diff.

3. **The bundled evidence-label cleanup is directionally correct.**

   Evidence: `development/loop7/niah_recall_matrix.py` now states the directional
   materiality rule (`h > base_hi`) in its docstring, matching the executable
   matrix rule. The MMLU JSONs now include op-point, graph-mode, transport, and
   deterministic example-seed metadata.

## Mainline Gaps

1. **AC-3 is still incomplete for the anchor-budget variant under CUDA graph.**

   Evidence: `ds_scorer_is_graph_safe()` still returns true only when
   `anchor_mode == "off"` (`python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:428`),
   and validation still rejects non-default `anchor_mode` unless CUDA graph is
   disabled (`python/sglang/srt/layers/attention/double_sparsity/validator.py:105`).
   This is exactly the original-plan anchor-budget variant, so it cannot be left
   as "queued" if Loop 7 is meant to close.

   Required implementation plan:
   - Extend `DSGraphState` with fixed-shape anchor scratch sized to `[max_bs, max_top_k]`
     plus any candidate/merge buffers needed for post-topK replacement.
   - Implement graph-safe post-topK force-include for `anchor_mode={recency,global,strided}`
     after `retrieve_topk_graph_safe`: preserve exactly `top_k`, clamp over-budget
     anchor counts as the eager path does, evict the lowest-ranked non-anchor
     selected positions, and keep `-1` padding safe.
   - Thread `anchor_mode` and `anchor_budget` through the production graph-safe
     call sites and capture helper; remove the validator rejection only after
     graph-safe behavior exists.
   - Add eager-vs-graph equality and replay no-allocation tests across
     `scorer_norm={off,cosine,hybrid} x head_agg={max,mean} x anchor_mode={off,recency,global,strided}`.

2. **AC-4/task13-task17 are still unimplemented although the oracle gate justifies bounded Tier-2.A work.**

   Evidence: `DoubleSparsityConfig` still has no `enable_lifted_budget_decode`
   or `lifted_budget_top_k` fields (`python/sglang/srt/layers/attention/double_sparsity/config.py:77`),
   and the top-k mismatch check still only exposes the old
   `SGLANG_DS_ALLOW_TOPK_MISMATCH` escape (`python/sglang/srt/layers/attention/double_sparsity/validator.py:188`).

   Required implementation plan:
   - Write the task13 design/disposition document first, defining the exact ABI:
     `enable_lifted_budget_decode: bool` and `lifted_budget_top_k: int`.
   - Add those fields to `DoubleSparsityConfig` and validation. Reject
     `top_k > index_topk` unless the new opt-in lifted-budget backend path is
     selected. Do not use `max_top_k`, Twilight fields, or
     `SGLANG_DS_ALLOW_TOPK_MISMATCH` as the lifted-budget mechanism.
   - Implement the opt-in path using `flash_mla_sparse_fwd` plus
     `dequantize_k_cache_paged`: selected physical slot ->
     `page_table_1_flattened` -> request-local compact KV index.
   - Mask or safe-replace `-1` padding before any dequant/index operation, keep
     fixed `lifted_budget_top_k` shapes with padding, and preserve the R23
     deterministic tie-break.
   - Add reference sparse-attention tolerance tests, prefix-sharing compact-remap
     tests, padding/duplicate/valid-length tests, TP=8 equality at 4096/8192, and
     graph replay allocation evidence.
   - Finish task17 with a landing/disposition record: production-ready landed
     path, or explicit hardening follow-on with recall evidence and DSA default
     untouched.

3. **AC-6/task19 and task20 final consolidation are still missing.**

   Evidence: Round 8 added AC-1 artifacts only. There is still no Loop-7 final
   artifact recording conc-1/16 TTFT, decode TPS/req, GPU memory, graph-replay
   status, admission, Tier-1 spine non-regression, or the final strategic-gate
   supersession decision record. `development/loop7/m0_decision.md` is useful
   source text, not the final task20 artifact.

   Required implementation plan:
   - After task17 exists, run the existing `development/benchmark.sh` /
     comparison tooling at the Loop-7 op-point for DS-default, graph-safe
     DS-hybrid, and DSA.
   - Record conc-1 and conc-16 TTFT, decode TPS/req, GPU memory, graph replay
     status, admission, radix/cache assumptions, and exact server configs.
   - Write the consolidated DS-vs-DSA recall/perf/non-regression report.
   - Write the final decision record that supersedes the Loop-6 strategic gate's
     Tier-2.A-primary ordering with final M0/R4/R7/R8 evidence and the AC-4
     disposition.

## Blocking Side Issues

None found that block the Round-8 AC-1 objective. The oracle-off replay artifact
and stride reference pass the relevant checks.

## Queued Side Issues

1. The R8 stride artifact is durable, but its row-level source
   `.sglang_ds_oracle/sink.jsonl` is gitignored. The committed JSON plus the
   hardcoded `stride=1` call site are enough for AC-1, but task20 should cite
   that provenance explicitly or archive a hash/count of the raw sink used.
2. Claude's summary says the MMLU artifacts were enriched with `data_dir`, but
   the committed `mmlu_{dsa,default,hybrid}_graph.json` files do not contain
   that field; e.g. `development/loop7/mmlu_dsa_graph.json:1-11`. The updated
   runner would emit it, so regenerate or patch the three JSONs before task20.
3. Plan/workflow markers in production comments/tests remain cleanup work before
   final merge.

## Goal Alignment Summary

ACs: 5/6 addressed | Forgotten items: 0 | Unjustified deferrals: 4

AC status:

| AC | Status | Review result |
|----|--------|---------------|
| AC-1 | MET | R8 closes oracle-off byte-equivalence/zero-allocation and stride=1 dense reference; prior fail-closed/oracle sink/AC-1.1 evidence remains accepted. |
| AC-2 | PARTIAL/ADVANCED | DS-vs-DSA recall uplift and A-vs-B gate evidence exist. Final closure still depends on AC-4 disposition, AC-6 consolidation, and task20. |
| AC-3 | PARTIAL/ADVANCED | Hybrid scorer graph-mode non-regression is accepted, but anchor-budget graph-safe support remains missing. |
| AC-4 | NOT MET | No lifted-budget ABI, decode path, tests, graph safety evidence, or disposition record. |
| AC-5 | MET | 64K servability at mem0.7 remains verified with served/admission separated. |
| AC-6 | PARTIAL | No final conc-1/16 perf guardrails or Tier-1 non-regression report. |

I updated the mutable tracker section:

- bumped Plan Version to 10 for Round 8 Review;
- added a Round 8 Review plan-evolution row;
- moved task4, task6, and task7 to Completed and Verified;
- kept AC-3 anchor graph-safe follow-up, task13-task17, task19, and task20 active;
- added queued provenance/metadata hygiene for the R8 stride sink and MMLU `data_dir` mismatch.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-9-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-9-summary.md

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
