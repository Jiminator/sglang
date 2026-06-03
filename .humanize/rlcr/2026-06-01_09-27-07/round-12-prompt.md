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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-12-contract.md

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
# Round 11 Review Result

Mainline Progress Verdict: ADVANCED

Round 11 advanced the stated contract. The R10 startup blocker is resolved:
`enable_lifted_budget_decode=true` now fails closed in `validate_double_sparsity`
while the lifted decode backend is unavailable, including the two concrete R10
cases and the hf-config-unavailable path. Do not close Loop 7: task14-task17,
task19, and task20 remain original-plan mainline work.

## Implementation Review

Accepted R11 work:

1. The fail-closed validator gate is real and early enough.

   Evidence: `python/sglang/srt/layers/attention/double_sparsity/validator.py:96`
   imports `ds_lifted_budget_decode_available()` and rejects
   `config.enable_lifted_budget_decode` before the CUDA-graph variant guard, page
   checks, backend checks, capability check, and model-topk block. The gate is
   therefore independent of `hf_config` resolution, matching the R11 contract.

2. The availability seam currently fails closed.

   Evidence:
   `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:457`
   defines `ds_lifted_budget_decode_available()` and returns `False`. Because no
   task14 decode branch exists yet, this is the correct current behavior.

3. The tests cover the R10 reproductions.

   Evidence:
   `test/registered/unit/layers/attention/test_scorer_variants.py:528` covers
   the silent-no-op case (`top_k=2048`, lifted budget 4096), and
   `test_scorer_variants.py:544` covers the wide-into-old-assert case
   (`top_k=4096`, lifted budget 8192). The stale "WITH flag passes" assertion was
   removed from `test_validator_topk_gt_index_topk_requires_flag`.

4. Local verification passed.

   Commands run:
   - `pytest -q test/registered/unit/layers/attention/test_scorer_variants.py::TestLiftedBudgetABI`
     -> `9 passed`
   - Direct validator probe with `get_model_config()` forced to raise still
     produced the recognized-but-not-implemented error for both R10 cases.
   - `git diff --check HEAD~1..HEAD` -> no whitespace errors.

No high-signal implementation bug was found in the R11 code path.

## Mainline Gaps

1. **AC-4 task14-task17 are still unimplemented and remain mainline.**

   This is not a regression in R11, but it is still unfinished original-plan work.
   R11 completed the task13 ABI/fail-closed surface only; AC-4 itself is not met.

   Required implementation plan:
   - Keep `top_k` as the base DSA/indexer budget and use
     `lifted_budget_top_k` as the fixed padded lifted selection width.
   - Implement the explicit lifted decode branch gated by
     `enable_lifted_budget_decode`; until task16 hardening, require eager/no
     production CUDA graph for that branch.
   - In the branch, select `lifted_budget_top_k` logical positions with the
     existing deterministic score-desc/position-asc tie-break, map logical
     positions to physical slots, mask or safe-replace pads before dequant,
     build request-local compact dequant indices, and pass compact-domain
     indices to `flash_mla_sparse_fwd`.
   - Dedup or assert uniqueness after physical remap; duplicates within one
     query row must not reach `flash_mla_sparse_fwd` as multiple valid entries.
   - Add task15 tests: reference sparse-attention tolerance on deterministic
     fp8/dequant cases, prefix-sharing remap, invalid padding, duplicate
     handling, `valid_lengths`, TP=8 equality at 4096/8192, and the direct
     `flash_mla_sparse_fwd` 4K-topk smoke/accuracy test.
   - For task16, add an alloc-free `out=`/scratch `dequantize_k_cache_paged`
     variant plus q-padding/compact-index scratch and prove zero allocation under
     graph replay before allowing production graph use.
   - For task17, write the landing disposition with served recall evidence and
     CIs. If production hardening is carried forward, the record must show the
     recall evidence, the DSA default untouched, and the research path gated out
     of production capture.

2. **AC-6/task19 and task20 are still pending.**

   Required implementation plan:
   - After task17 exists, run the existing measurement/benchmark tooling at the
     Loop-7 op-point for DS-default, graph-safe DS-hybrid, and DSA.
   - Record conc-1 and conc-16 TTFT, decode TPS/req, GPU memory, graph replay
     status, admission, radix/cache assumptions, and exact server configs.
   - Write the consolidated DS-vs-DSA recall/perf/non-regression report.
   - Write the final decision record superseding the Loop-6 Tier-2.A-primary gate
     with the final M0/R4/R7/R8/R9/R10/R11 evidence and the AC-4 disposition.

## Blocking Side Issues

None. The R10 lifted-budget startup blocker is resolved by R11.

## Queued Side Issues

1. Preserve/cite the R8 raw oracle-sink provenance before task20, or cite the
   hardcoded `stride=1` call site plus committed aggregate explicitly.
2. Remove plan/workflow markers and stale variant comments before final merge.
3. Learned/distilled selector work remains out of scope unless explicitly
   approved under DEC-5.

## Goal Alignment Summary

ACs: 6/6 addressed | Forgotten items: 0 | Unjustified deferrals: 0

| AC | Status | Review result |
|----|--------|---------------|
| AC-1 | MET | Prior R8 evidence remains accepted. |
| AC-2 | PARTIAL | Recall/MMLU evidence exists; task19 consolidation and task20 final record remain. |
| AC-3 | MET | R9/R10 graph-safe variant coverage and anchor clamp remain accepted. |
| AC-4 | PARTIAL / NOT MET | task13 is now accepted as complete; task14-task17 are still missing. |
| AC-5 | MET | 64K servability at mem0.7 remains verified. |
| AC-6 | PARTIAL | Final conc-1/16 perf guardrail report is still missing. |

The tracker represents all original plan tasks in Active, Completed, or the empty
Deferred table. There are no forgotten task IDs. The Explicitly Deferred table is
empty; the remaining work is active, not justified away.

## Goal Tracker Update Requests

I updated `goal-tracker.md` mutable state directly:

- bumped Plan Version to 14 for Round 11 Review;
- added a Round 11 Review plan-evolution row;
- marked task13 verified in the Completed and Verified table;
- kept task14-task17, task19, and task20 active;
- made no changes to the immutable section.

No requested tracker change was rejected.

PENDING
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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-12-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-12-summary.md

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
