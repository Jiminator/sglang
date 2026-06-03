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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-14-contract.md

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
# Round 13 Review Result

Mainline Progress Verdict: ADVANCED

Round 13 advanced the stated task14 contract. The lifted-budget seam is open,
startup gating is stricter, the selector/backend buffers widen to
`lifted_budget_top_k`, and `forward_decode` now has a default-off lifted branch
that remaps physical slots into compact KV indices before calling
`flash_mla_sparse_fwd`. This does not complete Loop 7 or AC-4: task15 live recall
and lifted TP equality, task16 production hardening, task17 disposition, task19
perf consolidation, and task20 final decision record remain original-plan work.

## Implementation Review

Accepted R13 work:

1. The lifted-budget validator and config gates are real.

   Evidence: `config.py:147-166` rejects non-`%128` lifted widths. When the flag is
   enabled, `validator.py:111-126` requires the backend seam and
   `--disable-cuda-graph`; `validator.py:235-249` requires `top_k == index_topk`
   and `lifted_budget_top_k > index_topk`. This matches the R13 eager-only contract.

2. The selector/output width is actually widened for the opt-in path.

   Evidence: `selector.py:89-92` uses `lifted_budget_top_k` as
   `DoubleSparsitySelector.max_top_k`; `dsa_backend.py:497-506` mirrors that value
   into `ds_max_top_k`; `dsa_backend.py:852-863` and `:1165-1178` allocate
   `ds_topk_indices_out` and `DSGraphState` to that width.

3. The decode branch is wired and keeps the default path separate.

   Evidence: `dsa_backend.py:1978-1987` intercepts the lifted case before the
   default `flashmla_kv` route; `dsa_backend.py:2093-2124` calls
   `build_lifted_compact_kv` and then `_forward_flashmla_sparse`;
   `lifted_budget.py:203-215` masks pads through the R12 compact remap and passes
   only `page_table_1_flattened` to `dequantize_k_cache_paged`. The old
   `flashmla_kv` cap assert is still present at `dsa_backend.py:2210-2213`.

4. The availability seam is flipped.

   Evidence: `selection_kernel.py:457-471` now returns `True`, and a direct probe
   returned `True` on this runner.

5. Local validation reproduced the claimed unit results.

   Commands run:
   - `pytest -q test/registered/unit/layers/attention/test_lifted_budget_decode.py test/registered/unit/layers/attention/test_scorer_variants.py::TestLiftedBudgetABI`
     -> `24 passed`
   - `pytest -q test/registered/unit/layers/attention/test_lifted_budget_decode.py test/registered/unit/layers/attention/test_scorer_variants.py test/registered/unit/layers/attention/test_double_sparsity_unit.py test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py`
     -> `337 passed, 24 warnings, 9 subtests passed`
   - `git diff --check d187f59f4..2ba4dafc1` -> clean
   - `ds_lifted_budget_decode_available()` probe -> `True`

No high-signal implementation bug was found in the task14 wiring itself.

## Mainline Gaps

1. **AC-4 task15 is still incomplete.**

   R13 added helper/kernel correctness at 4096/8192, but the original plan still
   requires binding served recall evidence and lifted-width TP equality. The
   current tests do not prove that the live server recovers 4K recall, and
   `test_ds_scorer_tp_determinism.py` still covers only the small AC-3 scorer
   matrix, not `lifted_budget_top_k` 4096/8192.

   Required implementation plan:
   - Add a direct backend-level test that calls the wired lifted method, not only
     `build_lifted_compact_kv`, with deterministic fp8 KV, prefix sharing,
     duplicate physical slots, `valid_lengths < width`, and widths 4096 and 8192.
   - Extend the TP determinism harness with a lifted-budget case for
     `max_top_k in {4096, 8192}` and `max_seq_len >= max_top_k`, using the same
     logical production selector path and 8-rank all-reduce. Assert identical
     `selected_indices` and `valid_lengths` across all ranks.
   - Run the live served NIAH 4K recall-recovery sweep using the existing
     Loop-7 serve/harness tooling, eager mode, `enable_lifted_budget_decode=true`,
     `lifted_budget_top_k=4096`, `top_k=index_topk=2048`, and N>=20 with exact
     Clopper-Pearson CIs. Record DS default/hybrid 2048 vs lifted 4096 on the same
     node and state whether the uplift exceeds the baseline CI.
   - Record the exact server args, DS config, commit, GPU type, trial count,
     admission status, and artifact paths. Do not claim AC-4 recall recovery from
     helper tests alone.

2. **AC-4 task16 and task17 remain pending.**

   The current lifted path is explicitly eager-only because both
   `build_lifted_compact_kv` and `dequantize_k_cache_paged` allocate. That is
   acceptable for task14, but not for the production-ready landed path required
   before Loop 7 closes.

   Required implementation plan:
   - Add `dequantize_k_cache_paged_out(quant_k_cache, page_table_1_flattened, out)`
     in `dequant_k_cache.py`, backed by the existing Triton kernel shape but
     writing into caller-owned bf16 scratch.
   - Add a fixed-shape lifted compact builder for graph mode: preallocate
     `page_table_1_flattened_scratch` with length `max_bs * lifted_budget_top_k`,
     `compact_indices_scratch [max_bs, lifted_budget_top_k]`, and
     `compact_kv_scratch [max_bs * lifted_budget_top_k, 1, 576]`. Invalid and
     duplicate lanes must write a safe physical slot into the dequant input and
     `-1` into `compact_indices`, so `flash_mla_sparse_fwd` masks them.
   - Preallocate q head-padding scratch in the backend and route the lifted branch
     through the scratch-backed path when CUDA graph capture is enabled.
   - Add allocation-replay tests at 4096 and 8192 proving the lifted decode branch
     replays with zero new allocations and matches the eager/reference output.
   - Only after those tests pass, relax the validator's `--disable-cuda-graph`
     rejection for the lifted path. Then write the task17 Tier-2.A disposition
     record with the live recall evidence, graph-safety evidence, default DSA path
     status, and perf/memory impact.

3. **AC-6 task19 and AC-2 task20 remain pending.**

   Required implementation plan:
   - After task17 exists, run the existing Loop-7 serve/benchmark tooling at the
     selected op-point for DS default, graph-safe DS hybrid, DSA, and lifted-budget
     DS if the recall sweep shows a material win.
   - Record conc-1 and conc-16 TTFT, decode TPS/req, GPU memory, graph replay
     status, admission behavior, radix/cache assumptions, exact launch configs,
     and artifact paths.
   - Write the consolidated DS-vs-DSA recall/perf/non-regression report.
   - Write the final strategic-gate supersession decision record using the
     M0/R4/R7/R8/R9/R10/R11/R12/R13 evidence plus the task17 AC-4 disposition.

## Blocking Side Issues

None.

## Queued Side Issues

1. Preserve or cite the R8 oracle-sink provenance before task20, or cite the
   hardcoded `stride=1` call site plus committed aggregate explicitly.
2. Remove plan/workflow markers and stale comments before final merge.
3. Learned/distilled selector work remains out of scope unless explicitly approved
   under DEC-5.

## Goal Alignment Summary

ACs: 6/6 addressed | Forgotten items: 0 | Unjustified deferrals: 0

| AC | Status | Review result |
|----|--------|---------------|
| AC-1 | MET | Prior R8 oracle-off/stride/fail-closed evidence remains accepted. |
| AC-2 | PARTIAL | Recall/MMLU evidence exists; task19 consolidation and task20 final record remain. |
| AC-3 | MET | R9/R10 graph-safe variant coverage remains accepted. |
| AC-4 | PARTIAL / NOT MET | task14 is now accepted as complete, but task15 live recall + TP equality, task16 hardening, and task17 disposition remain. |
| AC-5 | MET | 64K servability at mem0.7 remains verified. |
| AC-6 | PARTIAL | Final conc-1/16 perf guardrail report remains missing. |

The tracker represents all original plan tasks in Active, Completed, or the empty
Deferred table. There are no forgotten task IDs and no Explicitly Deferred items.
The remaining work is active, not justified away.

## Goal Tracker Update Requests

I updated `goal-tracker.md` mutable state directly:

- bumped Plan Version to 17 for Round 13 Review;
- added a Round 13 Review plan-evolution row;
- moved task14 from Active to Completed and Verified;
- kept task15, task16, task17, task19, and task20 active;
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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-14-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-14-summary.md

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
