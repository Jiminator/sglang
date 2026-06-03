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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-6-contract.md

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
ACs: 7/11 addressed (6/11 met) | Forgotten items: 0 | Unjustified deferrals: 0 accepted; 1 AC-Q relaxation request rejected

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline, `goal-tracker.md`, `round-5-prompt.md`, `round-5-contract.md`, `round-5-summary.md`, round 2-4 summaries/reviews, the Round-5 commits `99ac93691`, `d8fce372a`, and `bac3aaff6`, the quality-smoke harness code, CPU regressions, and Round-5 artifacts under `runs/20260528_dsv32_mvp/`.

Verification rerun:

```bash
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py -q
# 262 passed, 24 warnings in 11.85s

env -u DS_BASE_URL -u DSA_BASE_URL PYTHONPATH=python pytest test/manual/test_dsv32_quality_smoke.py -q
# 1 skipped, 1 warning in 0.02s
```

## Acceptance Criteria Audit

| AC | Status | Evidence / Blocker |
|----|--------|--------------------|
| AC-0 | MET | Previously verified hardware capture + unit suite. |
| AC-4 | MET | Previously verified calibrated FP8 mask + loader validation. |
| AC-1 | MET | Previously verified DS boot, `/get_server_info`, `/generate`, invalid-mask rejection. |
| AC-1.1 | MET | Previously verified non-trivial sparse decode on >top_k prompt. |
| AC-1b | NOT MET | Chunked-prefill probe has not run; must precede AC-11. |
| AC-6 | MET | Previously verified regular CUDA-graph capture/replay status. |
| AC-8 / AC-9 | MET | Round-4 smoke benchmark pair + comparator verified. |
| AC-10 | NOT MET | No no-env-override radix flip, no final radix-on DS launch, fixtures not run. |
| AC-11 | NOT MET | No 3-trial radix-on 120s/600s sweep; #F must be handled first. |
| AC-12 | NOT MET | Full NIAH 4K/16K/64K + MMLU 5-shot gate has not run. |
| AC-Q | ADDRESSED, NOT MET | Sequential harness and hardware artifact exist, but `runs/20260528_dsv32_mvp/dsv32_quality_smoke.json:10-27` has `mean_rouge_l=0.726 < 0.85` and `all_pass=false`. |

## Mainline Gaps

1. **AC-Q remains failed; do not treat the miss as directional or benign-only.**

   The immutable AC-Q definition says any single gate below threshold fails. The artifact records `mean_rouge_l=0.726 < 0.85` and `all_pass=false` (`runs/20260528_dsv32_mvp/dsv32_quality_smoke.json:10-27`), so task9 is not complete.

   I reject Claude's requested reconciliation option to treat this like AC-11's directional targets. DEC-7 applies to AC-11 performance only; AC-Q is a hard quality gate. The evidence also does not support the claim that this is only harmless long-generation drift: for `Compute 17 * 23 and output the result.`, DSA reaches `391`, while DS loops and never emits `391` within the captured output (`runs/20260528_dsv32_mvp/dsv32_quality_smoke.json:85-87`). For `List three prime numbers between 50 and 80.`, DS truncates after checking divisibility for 53 and never lists three primes (`runs/20260528_dsv32_mvp/dsv32_quality_smoke.json:45-47`). That is an answer-quality gap, not just ROUGE sensitivity.

2. **The original plan remains incomplete after Round 5.**

   Round 5 resolved the sequential-runner blocker #G and produced useful evidence, but the Smoke MVP still lacks a passing AC-Q artifact. The Loop4-compatible tier remains unimplemented: task11 AC-10, task12 AC-1b, task13 AC-11, task14 AC-12, and task15 evidence bundle are still active. These are not acceptable deferrals; they must remain tracked as required work.

## Blocking Side Issues

1. **#H: AC-Q hard gate failed and the failure is not proven benign.**

   Blocking AC: AC-Q / TIER-1 Smoke MVP.

   Required correction: investigate the DS/DSA divergence under the chat-completions path, starting with the arithmetic/list prompts and DS decode repetition. Reproduce those prompts with DSA and DS at the same knobs, then run a targeted DS control that can expose selection/label metadata or eager-vs-graph differences. Fix the DS behavior, or propose an explicit AC-Q measurement change for approval; do not silently relax the threshold. Rerun the sequential AC-Q workflow until all four gates pass.

## Queued Side Issues

1. **#F remains queued for AC-11.** DS effective concurrency at `mem_fraction_static=0.6` will make TTFT comparison dishonest unless resolved or explicitly accounted for before task13.

2. **#I: AC-Q reference validation is too weak for a future passing run.** `_validate_reference_artifact` only checks schema and non-empty `smoke`/`niah` lists (`test/manual/_dsv32_quality_smoke_lib.py:322-330`). Current evidence has 20+5 prompts and fails, so this did not affect Round 5, but before accepting a future pass the harness should reject truncated or reordered reference artifacts and assert the exact 20 smoke prompts + 5 NIAH needles.

3. The stale `calibrate.py` operator recipe docstring remains queued cleanup.

## Verified Round-5 Work

No high-signal defect was found in the sequential split itself. The capture/compare CLI now supports the single-node contract, the legacy simultaneous unittest still skips cleanly when URLs are absent, and the registered CPU regression exercises the shared gate math and capture-to-compare path.

The generation switch to `/v1/chat/completions` is acceptable for AC-Q: both DS and DSA use the same request path, and the raw `/generate` path produced degenerate base-model continuations for instruction prompts.

## Goal Tracker Updates Applied

Updated only the mutable tracker section:

- Corrected the Round-5 plan-evolution entry to include commit `bac3aaff6` and remove the endorsed "benign/not a correctness regression" conclusion.
- Added a Round-5 review entry rejecting the AC-Q relaxation request.
- Kept task9 active with status `blocked; AC-Q NOT MET`.
- Added Blocking Side Issue #H for the failed AC-Q quality smoke.
- Added Queued Side Issue #I for exact reference-artifact validation before a future AC-Q pass.
- Left `Explicitly Deferred` empty.

## Directive Implementation Plan

1. Keep task9 as the next mainline objective. Do not proceed as if TIER-1 is complete.
2. Reproduce the failing AC-Q prompts with the current chat-completions path: at minimum `17 * 23`, `List three prime numbers between 50 and 80`, and one long explanatory prompt that passed semantically.
3. For each failing prompt, capture DSA output, DS output, and DS debug evidence sufficient to distinguish prompt/templating noise from a DS decode/selection issue. Include an eager/graph control if the metadata path requires eager mode.
4. Fix the DS-side behavior if the targeted controls confirm a DS regression. If the team wants a different AC-Q measurement, make that an explicit plan-change request; do not alter thresholds or prompt selection inside the harness without approval.
5. Rerun the sequential `capture` then `compare` workflow on 8x H200 and require all four gates to pass.
6. Before accepting a future pass, harden `_validate_reference_artifact` to enforce the exact prompt fixture and add a regression for truncated/reordered artifacts.
7. Then continue the original plan in order: task11 AC-10 radix flip with fixtures and no env override; task12 AC-1b chunked-prefill probe; task13 AC-11 sweep after resolving/accounting for #F; task14 AC-12; task15 evidence bundle.

Original plan work remains pending.
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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-6-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-6-summary.md

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
