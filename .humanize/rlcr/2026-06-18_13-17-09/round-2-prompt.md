Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop12/plan.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop12/plan.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/round-2-contract.md

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

## Goal Alignment Summary

ACs: 10/10 addressed | Forgotten items: 0 | Unjustified deferrals: 0

AC-8 is now accepted. The wrapper pins the generated-shared-prefix shape to the loop-11b single-group workload, the saved evidence completed exactly 256 requests, and the recomputed metrics meet the parity band. The round is still not complete because Claude claimed the P3 plan-marker cleanup and final sweeps were clean, but newly shipped DS files still contain workflow labels, and the provenance doc contains a false native-DSA band statement.

I updated the mutable section of `goal-tracker.md`: task13 is Completed and Verified, the stale AC-8 blocking issue is cleared, and task14/task15 remain Active for the doc/marker cleanup.

## Mainline Gaps

None at the AC level. The corrected AC-8 implementation and evidence satisfy the original workload and metric requirements:

- `/sgl-workspace/double-sparisty-v2/sglang/benchmarks/bench_double_sparsity.py:84-87` passes `--gsp-num-groups 1 --gsp-prompts-per-group <num_prompts>`.
- `/sgl-workspace/double-sparisty-v2/sglang/benchmarks/bench_double_sparsity.py:158-183` records `expected_prompts`, `actual_completed`, and `request_shape_ok`, and includes the shape check in `parity`.
- `/sgl-workspace/double-sparisty-v2/sglang/benchmarks/bench_double_sparsity.py:197-205` fails closed if the completed request count differs from `--num-prompts`.
- `development/loop12/m6m8_eval_r1.out` shows `gsp_num_groups=1`, `gsp_prompts_per_group=256`, `Total prompts: 256`, `Successful requests: 256`, and `PASS`.
- `development/loop12/perf_evidence/verdict.json` records `actual_completed=256`, `request_shape_ok=true`, `p50_decode_tps=35.053`, `p99_ttft_s=22.901`, and `parity=true`.

## Blocking Side Issues

None. The remaining issues do not block the AC-8 perf proof or DS runtime behavior.

## Queued Side Issues

### [P3] Plan-marker cleanup is still incomplete

Claude claimed the shipped-comment marker sweep was clean, but the branch still has plan/workflow labels in newly shipped files and user-facing validator messages:

- `/sgl-workspace/double-sparisty-v2/sglang/python/sglang/srt/layers/attention/double_sparsity/config.py:8` — `Option B`
- `/sgl-workspace/double-sparisty-v2/sglang/python/sglang/srt/layers/attention/double_sparsity/page_table_adapter.py:9` — `Option B`
- `/sgl-workspace/double-sparisty-v2/sglang/python/sglang/srt/layers/attention/double_sparsity/calibrate.py:428` — `Round 3`
- `/sgl-workspace/double-sparisty-v2/sglang/python/sglang/srt/layers/attention/double_sparsity/validator.py:218,251,262` — `Option B`
- `/sgl-workspace/double-sparisty-v2/sglang/python/sglang/srt/layers/attention/double_sparsity/validator.py:220,251` — `Tier-2.A`
- `/sgl-workspace/double-sparisty-v2/sglang/python/sglang/srt/layers/attention/double_sparsity/topk_kernel.py:78` — `round 1` (algorithmic radix wording, but it should be reworded if the final sweep treats `round N` as a workflow marker)

These do not change runtime behavior, so they are not an AC blocker, but they contradict the plan's code-style requirement and the Round 1 success criterion. The final close-out should not report COMPLETE until this sweep is actually clean.

Directive implementation plan:

1. Reword `Option B` as durable technical language such as `DSA indexer parity`, `FlashMLA sparse path`, or `DSA index-topk operating point`, depending on the local sentence.
2. Reword `Tier-2.A lifted-budget path` to `lifted-budget path`.
3. Reword `Round 3` in `calibrate.py` as the actual bug/fix: reshape MLA projections before slicing so V/RoPE columns from later heads are not selected.
4. Reword `round 1` in `topk_kernel.py` to `first radix pass` if the final marker sweep includes `round N`.
5. Remove the extra blank line reported by `git diff --check` at `python/sglang/srt/layers/attention/double_sparsity/metrics.py:281`.
6. Rerun the marker sweep over changed files, `git diff --check`, the import gate, and the three focused unit tests; then re-push.

### [P3] Provenance doc contradicts its own native-DSA table

`/sgl-workspace/double-sparisty-v2/sglang/benchmarks/DOUBLE_SPARSITY.md:96-104` reports native DSA P99 TTFT as `46.50 s` against a `≤ 30.1 s` band, then says "both DS and DSA meet the band once the workload shape is fixed." That statement is false as written and there is no saved corrected-shape native-DSA evidence supporting it.

Fix the prose to say the native-DSA column is same-base context only, while the accepted corrected-shape run is the DS run with `actual_completed=256`, `gsp_num_groups=1`, and `request_shape_ok=true`.

## Verification Performed

- Read `development/loop12/plan.md`, the Pensieve review pipeline and taste-review knowledge, the Round 0 summary/review, the Round 1 prompt/contract/summary, and `goal-tracker.md`.
- Checked v2 branch state: clean worktree, branch `double-sparsity-v2`, HEAD and remote both `f05326636c1ad1eab51a20ba61d479c2d95115a0`, merge-base `<BASE>=105e095e005d02a178fb6c5a23bd22ba644c90e4`.
- Verified the v2 diff remains 42 files with no `.pensieve/`, `.humanize/`, `development/`, or `SLOS.md`.
- Inspected stock `bench_serving` and `GeneratedSharedPrefixDataset`: the dataset produces `num_groups * prompts_per_group` requests, so the wrapper must pin both GSP fields and guard `actual_completed`.
- Recomputed the saved metrics from `development/loop12/perf_evidence/bench_result.json`: 256 completed requests, median decode TPS `35.0534277774`, P99 TTFT `22.9006561339s`, zero request errors.
- Ran `PYTHONPATH=python python3 -c "import sglang; import sglang.srt.layers.attention.dsa_backend; import sglang.srt.layers.attention.double_sparsity as ds; print('ok')"` in the v2 clone: passed.
- Ran `PYTHONPATH=python python3 -m pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py test/registered/unit/layers/attention/test_lifted_budget_decode.py test/registered/unit/managers/test_ds_abort_path.py`: 114 passed.
- Checked dropped-module refs in changed files: zero references to `oracle_artifact_sink`, `selection_recall_oracle`, `radix_fixture_capture`, `score_capture`, `selection_capture`, or `latent_capture`.
- Checked radix/dead-symbol refs: no `apply_radix_fixture_artifact`, radix recorders, `RADIX_FIXTURE_STATE_*`, `record_selection`, `ds_recall_oracle_enabled`, or `_maybe_record_recall_oracle`.

Round remains open.
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/round-2-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/round-2-summary.md

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
