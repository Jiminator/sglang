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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-10-contract.md

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
# Round 9 Review Result

Mainline Progress Verdict: ADVANCED

Round 9 advanced the stated contract. The anchor-budget variant no longer sits on
the eager-only side path: `anchor_mode`/`anchor_budget` are threaded through the
graph-safe selector, `deepseek_v2.py`, and `capture_decode_step`, and the startup
guard no longer rejects anchor under CUDA graph. AC-3 variant coverage is now met.

Do not close Loop 7. AC-4 lifted-budget decode, AC-6 perf/consolidation, and the
final strategic-gate supersession record remain original-plan acceptance work.

## Part 1: Goal Tracker Audit

| AC | Status | Evidence if met | Blocker if not met | Justification if deferred |
|----|--------|-----------------|--------------------|---------------------------|
| AC-1 | MET | R8 verified oracle-off byte-equivalence + zero allocation under CUDA-graph replay (`oracle_off_graph_replay_alloc.json`) and stride=1 dense reference (`oracle_stride_reference.json`), on top of the R4 fail-closed oracle and R0 AC-1.1 force-include evidence. | None. | Not deferred. |
| AC-2 | PARTIAL | R5/R7 provide DS-vs-DSA same-node recall matrices, CIs, material 16K hybrid uplift, within-budget parity, and MMLU re-anchor. | Final task20 decision record is still missing, and task19 consolidation must bring the final recall/perf/non-regression evidence together. | Not deferred. |
| AC-3 | MET | R9 completes the anchor-budget graph-safe path. `selection_kernel.py` uses one tensorized `_force_include_anchor` for eager and graph-safe paths; `retrieve_topk_graph_safe`, `deepseek_v2.py`, and `cuda_graph.py` thread anchor flags; validator/serve guard no longer force eager for anchor. Local review tests passed: `test_scorer_variants.py`, `test_ds_scorer_tp_determinism.py`, and `test_double_sparsity_unit.py`. | None. | Not deferred. |
| AC-4 | NOT MET | None yet beyond oracle-gate justification from task7. | task13-task17 remain active: no lifted-budget ABI, no opt-in decode path, no compact remap/tests, no production-hardening/disposition record. | Not deferred. |
| AC-5 | MET | R0/R1 baseline evidence records 64K served at mem0.7 with served/admission separated; tracker keeps task18 completed. | None. | Not deferred. |
| AC-6 | PARTIAL | R6/R8/R9 graph replay safety exists for selected pieces. | task19 remains active: no final conc-1/16 TTFT, decode TPS/req, memory, admission, graph replay, or Tier-1 non-regression report. | Not deferred. |

Forgotten items: none. The original plan tasks are represented in Active,
Completed, or the empty Deferred table. The tracker did have one drift item:
the R9 anchor follow-up was marked done while still sitting in Active; I moved it
to Completed and Verified.

Deferred items audit: the Explicitly Deferred table is empty. The learned/distilled
selector note is a DEC-5 scope boundary, not a deferred Loop-7 AC. No deferral
currently contradicts the Ultimate Goal.

Goal completion summary:

```text
Acceptance Criteria: 3/6 met (0 deferred)
Active Tasks: 7 remaining
Estimated remaining rounds: 3-5, depending on AC-4 implementation depth
Critical blockers: none
```

## Part 2: Mainline Drift Audit

The current round objective was clear and singular: port AC-3 anchor-budget
support to the graph-safe path. Claude has been advancing mainline ACs rather
than only clearing side issues: R6 graph-safe scorer, R7 binding AC-3 matrix, R8
AC-1 closure, and R9 anchor graph-safety are all direct acceptance work.

```text
Mainline Progress Verdict: ADVANCED
Blocking Side Issues: 0
Queued Side Issues: 3 open runtime/evidence cleanup items plus the DEC-5 learned-selector scope note
```

Blocking side issues: none found for the R9 objective.

Queued side issues:
1. Bound R9 anchor temporary work by effective budget/top-k before AC-6 perf/final merge.
2. Preserve/cite R8 raw oracle-sink provenance before task20.
3. Remove plan/workflow markers and stale variant comments before final merge.

## Part 3: Implementation Review

Accepted R9 work:

1. The graph-safe anchor integration is real.

   Evidence: `_force_include_anchor` is tensorized and called from both
   `retrieve_topk_via_labels` and `retrieve_topk_graph_safe`
   (`selection_kernel.py:895`, `selection_kernel.py:1058`,
   `selection_kernel.py:1496`). The graph-safe production call site passes
   `anchor_mode` and `anchor_budget` (`deepseek_v2.py:2332`), and the standalone
   capture helper does the same (`cuda_graph.py:342`).

2. The guard relaxation matches the implementation.

   Evidence: `ds_scorer_is_graph_safe()` now returns true for all non-learned
   variants (`selection_kernel.py:428`), the validator only separately rejects
   `recall_oracle` under CUDA graph (`validator.py:113`), and
   `serve_double_sparsity.sh` only auto-adds `--disable-cuda-graph` for
   `RECALL_ORACLE=1`. This matches the design boundary in
   `development/loop7/refined_plan_v1.md`: the non-learned AC-3 variants are
   opt-in/flag-gated and the default path remains unchanged. I also checked the
   public DSA environment-variable docs (`docs/references/environment_variables.md`)
   to keep the DSA/backend terminology aligned with repo docs.

3. Local verification passed.

   Commands run:
   - `pytest -q test/registered/unit/layers/attention/test_scorer_variants.py` -> `20 passed`
   - `pytest -q test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py` -> `2 passed`
   - `pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py` -> `290 passed, 24 warnings, 9 subtests`

4. Claim correction: the R9 finding doc overstates one evidence detail.

   `development/loop7/m6_anchor_graphsafe_finding.md` says the GPU eager-vs-graph
   matrix includes over-budget anchor coverage, but `test_graph_safe_matches_eager_all_variants`
   uses `AB = 16` with `K = 64`, and the CUDA replay test uses `AB = 8` with
   `K = 32`. Over-budget semantics are covered by CPU `_force_include_anchor`
   tests and the graph path uses the same helper, so this does not block AC-3,
   but the final evidence prose should not claim graph-matrix over-budget coverage
   unless a graph test is added.

5. Queued hardening: raw `anchor_budget` controls temporary tensor shape.

   `_force_include_anchor` sets `A = int(anchor_budget)` (`selection_kernel.py:918`)
   and allocates/generates `[bs, A]` temporaries before the effective tensor clamp
   (`min(anchor_budget, selected_count, seq_len)`) takes effect. A pathological
   opt-in `anchor_budget` can therefore allocate much more memory than the
   algorithm can use. Fix direction: clamp the tensor work shape to
   `min(anchor_budget, K, max_seq)` or validate/reject pathological budgets, and
   add an over-budget graph-path regression test.

## Part 4: Goal Tracker Update Requests

I updated `goal-tracker.md` mutable state:

- bumped Plan Version to 11 for Round 9 Review;
- added a Round 9 Review plan-evolution row;
- moved the AC-3 anchor graph-safe follow-up from Active to Completed and Verified;
- kept task13-task17, task19, and task20 active;
- resolved the MMLU `data_dir` queued item as R9-completed;
- added the raw-`anchor_budget` temporary-shape hardening item to Queued Side Issues.

No requested tracker change was rejected.

## Part 5: Progress Stagnation Check

Not stagnating. Recent rounds show distinct acceptance progress:

- R6: graph-safe scorer/head aggregation path.
- R7: binding graph-mode recall/MMLU matrix.
- R8: AC-1 oracle-off replay allocation + stride reference closure.
- R9: anchor-budget graph-safe port and AC-3 closure.

There are repeated reminders not to treat AC-4/AC-6/task20 as optional queue
items, but that is tracker discipline rather than circular development failure.

## Action Items

Mainline Gaps:
1. Complete AC-4/task13-task17: design and implement the explicit lifted-budget
   ABI/path/tests/disposition, or record the required disposition if hardening is
   deferred with evidence.
2. Complete AC-6/task19: final conc-1/16 TTFT, decode TPS/req, memory, admission,
   graph replay, and Tier-1 non-regression report.
3. Complete task20: final strategic-gate supersession decision record using the
   final M0/R7/R8/R9 evidence and AC-4 disposition.

Blocking Side Issues:
None.

Queued Side Issues:
1. Bound or validate `anchor_budget` temporary work shape; add over-budget graph
   regression coverage if the finding continues to claim it.
2. Preserve/cite R8 raw oracle-sink provenance before task20.
3. Remove plan/workflow markers and stale comments before final merge.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-10-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

### Post-Alignment Check Action Items

This round follows a Full Goal Alignment Check. Pay special attention to:
- **Forgotten Items**: Codex may have identified tasks that were being ignored. Address them.
- **AC Status**: If any Acceptance Criteria were marked NOT MET, prioritize work toward those.
- **Deferred Items**: If any deferrals were flagged as unjustified, un-defer them now.
- **Queued Issues**: Keep non-blocking follow-up work queued unless it now clearly blocks mainline progress.

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-10-summary.md

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
