Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop5/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop5/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-15-contract.md

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
Mainline Progress Verdict: ADVANCED

Goal Alignment Summary:
ACs: 10/11 met under mutable Plan Version 2, 0 deferred. AC-11 is executed with a recorded directional miss per DEC-7. AC-12 is MET under the user-authorized DS-fair re-scope; the original literal 4K/16K/64K NIAH parity gate remains preserved as beyond-budget characterization, not erased. Active original-plan evidence tasks: 0.

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `goal-tracker.md`, Round 11-13 summaries/reviews, Round 14 summary, recent commits, the re-scoped harness, the AC-12 helper regressions, AC-12 artifacts, `ac12_analysis.md`, `evidence_bundle.md`, `next_loop_issues.md`, and relevant DSA docs (`docs/advanced_features/attention_backend.md`, `docs/references/environment_variables.md`).

Verification rerun:

```bash
PYTHONPATH=python pytest \
  test/registered/unit/development/test_ac11_comparator.py \
  test/registered/unit/layers/attention/test_double_sparsity_unit.py \
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py \
  test/registered/unit/development/test_option_b_scripts.py \
  test/registered/unit/manual/test_ac12_helpers.py -q
# 411 passed, 24 warnings, 28 subtests

bash -n development/serve_double_sparsity.sh development/serve_native_nsa.sh
# pass
```

Tokenizer sanity check on the checked-in V3.2 tokenizer: 1024-word NIAH prompt -> 1097 raw tokens, 1536-word prompt -> 1658 raw tokens. The 64K server error shows the chat wrapper adds 4 tokens, so the current within-budget hard-gate prompts are genuinely below `index_topk=2048`.

## Acceptance Criteria Audit

| AC | Status | Evidence / Gap |
|----|--------|----------------|
| AC-0 | MET | Previously verified producer fix, hardware radix-capture meta, capture-disabled negative, and regression. |
| AC-4 | MET | Previously verified native-FP8 sharded calibration, full mask generation, loader validation, and SHA. |
| AC-1 | MET | Previously verified DS boot, `/get_server_info`, `/generate`, cluster model path, and invalid-mask fail-closed rejection. |
| AC-1.1 | MET | Previously verified genuine sparse decode on a prompt longer than `top_k` with `selected_tokens=2048` and `dense_fallback=0`. |
| AC-1b | MET | Chunked-prefill probe passed at the final radix-on operating point. |
| AC-6 | MET | Regular CUDA-graph capture/replay status recorded; piecewise graph disabled separately. |
| AC-8 / AC-9 | MET | Smoke DS/DSA benchmark pair plus comparator verified, radix-off both sides and labeled non-AC-11. |
| AC-10 | MET | No-env-override radix flip verified; both fixtures passed; final DS boots radix-on via artifact. |
| AC-11 | PARTIAL / EXECUTED DIRECTIONAL MISS | 3-trial radix-on sweep and comparator are valid; TPS/TTFT targets miss and are recorded per DEC-7 with follow-up. |
| AC-12 | MET under Plan Version 2 | Re-scoped DS-fair hard gates pass: MMLU 89%==89%, NIAH 1024/1536 100%==100%. Beyond-budget 4K/16K/64K remains recorded with `verdict=FAIL`; original pre-rescope artifacts are preserved. |
| AC-Q | MET | Corrected sequential quality smoke verified with all four gates passing. |

## Forgotten Items

No original task is missing from Active, Completed, or Deferred. Task1-task15 have executed evidence and the tracker now marks AC-12/task14 and task15 verified under Plan Version 2.

One documentation item is stale: `runs/20260528_dsv32_mvp/next_loop_issues.md:3-16` still says AC-12 is the only unmet criterion and lists "Re-scope AC-12" as a pending option. Round 14 chose that option, so this file now contradicts `ac12_analysis.md`, `evidence_bundle.md`, and the tracker.

## Deferred Items Audit

The Explicitly Deferred table is empty. DS long-context selector/kernel/KV-budget work is queued for the next loop, not deferred inside this loop. That does not contradict the Ultimate Goal after the user-authorized AC-12 re-scope, but the stale `next_loop_issues.md` wording should be corrected.

## Goal Completion Summary

Acceptance Criteria: 10/11 met under Plan Version 2, 0 deferred; AC-11 remains an executed directional miss.
Active Tasks: 0 remaining.
Estimated remaining rounds: 1 small documentation cleanup/review round if the handoff must be internally consistent.
Critical blockers: none.

## Mainline Drift Audit

The Round-14 mainline objective was clear and singular: apply the user-authorized AC-12 DS-fair re-scope and verify it on hardware. Claude advanced that objective rather than clearing side issues. Recent-round trend is not stagnant: Round 11 ran AC-12, Round 12 fixed missing 64K durable evidence, Round 13 proved the kernel/top-k constraint and decode-sound diagnosis, and Round 14 converted the user decision into a hard gate plus characterization.

Blocking Side Issues: 0
Queued Side Issues: 3

## Implementation Review

No high-signal implementation bug found in the re-scoped gate.

The core claims match the artifacts. `ac12_results/ac12_mmlu_5shot_20260529T190132Z.json` records DSA and DS both at 89.0%. `ac12_niah_1024_*` and `ac12_niah_1536_*` record 20/20 vs 20/20 with `gate_class=within_budget_hard` and `verdict=PASS`. The beyond-budget files record 4K 75%, 16K 5%, and 64K HTTP 400 with `ds_served=0` and `verdict=FAIL`, so the degradation is visible rather than hidden.

The budget rationale is supported. `ac12_ds_server_info.json` shows DS `double_sparsity_config.top_k=2048`, `dsa_decode_backend=flashmla_kv`, radix-on, and `max_total_num_tokens=53056`; DSA has `max_total_num_tokens=910784`. Round-13 evidence shows the validator refuses `top_k=8192` and the `flashmla_kv` decode kernel asserts `indices.shape[-1] == dsa_index_topk`. The DSA docs also identify DSA as V3.2's native sparse backend, and the environment docs record the DSA dense-prefill threshold defaulting to the model index top-k, 2048 for V3.2.

The new CPU regressions have teeth: the within-budget gate passes when DS returns the planted needle and fails when DS blanks it, while beyond-budget DS rejection is recorded without hard-failing the characterization path.

## Mainline Gaps

1. `next_loop_issues.md` must be reconciled with Round 14.

   The file still describes AC-12 as unmet and asks the user to decide whether to re-scope it (`runs/20260528_dsv32_mvp/next_loop_issues.md:3-16`). That was true in Round 13 but is stale after Round 14. Rewrite it so the chosen disposition is recorded and only the remaining DS long-context R&D, KV-budget, AC-11 follow-up, and strategic DS-on-DSA question remain.

## Blocking Side Issues

None.

## Queued Side Issues

1. AC-12 NIAH artifacts use `length_tokens` for requested word counts and set `within_budget` from `length_tokens <= INDEX_TOPK` (`test/manual/test_double_sparsity_v32.py:720-754`). Current evidence is safe by tokenizer sanity check, but the next harness touch should record actual chat input token counts and assert the within-budget hard gate from those counts.

2. Plan-specific terms were reintroduced in manual harness comments/docstrings (`test/manual/test_double_sparsity_v32.py:1-22`, `714-718`), and the pre-existing serve-header terms remain. This is hygiene only, not a gate blocker.

3. DS long-context R&D remains next-loop work: query-aware/learned selector, a decode kernel accepting `top_k > index_topk`, and smaller TokenLabelTable / KV budget to lift 64K admission and AC-11 TTFT.

## Goal Tracker Updates Applied

Updated only the mutable section of `goal-tracker.md`:

- Added a Round-14 review correction accepting the user-authorized DS-fair AC-12 re-scope.
- Moved AC-12/task14 and refreshed task15 evidence to Completed and Verified under Plan Version 2.
- Updated Active Tasks to show no active evidence tasks after Round 14.
- Reopened the non-blocking plan-term hygiene item.
- Added queued items for token-count artifact precision and stale `next_loop_issues.md`.

## Stagnation Check

Not stagnant. The last four rounds show concrete progress and changing evidence rather than circular retries: hardware gate execution, artifact-safety fix, kernel-budget diagnosis, and a user-authorized re-scoped gate with hardware pass.

No stagnation circuit breaker is triggered; the original-plan completion sentinel is intentionally withheld because AC-11's directional target is still a recorded miss and original literal AC-12 beyond-budget NIAH parity was re-scoped rather than literally satisfied.
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-15-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-15-summary.md

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
