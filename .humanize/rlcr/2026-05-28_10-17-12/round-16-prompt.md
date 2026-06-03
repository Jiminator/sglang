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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-16-contract.md

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
ACs: 11/11 addressed; 10/11 met under mutable Plan Version 2, with AC-11 executed as a recorded directional miss | Forgotten items: 0 | Unjustified deferrals: 0 under Plan Version 2; original-plan completion still has residual follow-up work, so the COMPLETE sentinel remains withheld

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-15-prompt.md`, `round-15-contract.md`, `round-15-summary.md`, Round 12-14 summaries/reviews, `goal-tracker.md`, commit `f6faaadf3`, the changed handoff doc, the changed manual AC-12 harness comments, and the current AC-12 analysis/evidence bundle.

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

## Implementation Review

No high-signal Round-15 implementation bug found.

The stale handoff gap from Round 14 is resolved. `runs/20260528_dsv32_mvp/next_loop_issues.md:3-17` now records the AC-12 disposition as decided, states that AC-12 is MET under the user-authorized DS-fair re-scope, and keeps the beyond-budget 4K/16K/64K failures visible as characterization rather than erasing them. The previous "only unmet criterion" / "Re-scope AC-12" pending-option wording is gone.

The remaining items in the handoff are now the right ones for the accepted Plan Version 2 state: DS long-context selector/kernel R&D (`next_loop_issues.md:19-24`), TokenLabelTable/KV-budget work for 64K admission and AC-11 TTFT (`next_loop_issues.md:26-36`), token-count precision for the within-budget AC-12 artifacts (`next_loop_issues.md:38-43`), the DS-on-native-DSA strategic question (`next_loop_issues.md:45-50`), and cosmetic serve-header terms (`next_loop_issues.md:52-55`).

The manual harness comment cleanup matches Claude's claim. The Round-14 loop-process wording was removed from the module docstring and the within-budget comment (`test/manual/test_double_sparsity_v32.py:20-22`, `714-718`). This was comment-only; the test rerun confirms no behavior regression.

## Mainline Gaps

None for the Round-15 mainline objective. The handoff document is now internally consistent with `ac12_analysis.md`, `evidence_bundle.md`, and the tracker.

Residual original-plan completion gap remains and the stop word must not be emitted. The literal original AC-12 4K/16K/64K parity gate is not literally satisfied; it is accepted only through the user-authorized Plan Version 2 DS-fair re-scope, while the beyond-budget artifacts remain `verdict=FAIL`. AC-11 is executed but still a recorded directional TTFT miss per DEC-7. This is consistent with the tracker, but it means there is still follow-up work before anyone can claim all original targets are fully done with no residuals.

Directive implementation plan if the loop is forced back to literal original-target completion:

1. Implement a DS-owned flexible decode backend that does not reuse `flashmla_kv`'s fixed `index_topk` contract. It must accept the DS selector's configured `top_k`, map logical positions to physical KV slots through the existing page-table adapter, gather selected K/V rows, and run attention over that gathered set. Keep the current `flashmla_kv` path fail-closed when `top_k != index_topk`.
2. Change validation so `top_k > index_topk` is allowed only with that new DS-flex backend, never through the current mismatch override. Add registered validator/backend tests for accepted `top_k=8192/16384` on DS-flex and rejection on the current backend.
3. Reduce TokenLabelTable memory enough to admit the ~70K-token 64K NIAH prompt at a higher DS KV budget. The concrete target is a compact signature/table representation with selection-time expansion, then a `/get_server_info` artifact showing `max_total_num_tokens` above the 64K prompt length.
4. Rerun a staged hardware sweep: first NIAH 16K at `top_k=2048/8192/16384`, then 64K after admission is proven. If dense or near-dense selection still misses, treat it as a DS-flex decode bug and fix before updating AC-12 artifacts.
5. Rerun the full AC-12 literal gate and the AC-11 comparator at the new admitted operating point. Update `ac12_analysis.md`, `evidence_bundle.md`, and `next_loop_issues.md` only from the new hardware artifacts.

## Blocking Side Issues

None. The current Round-15 changes do not block the accepted Plan Version 2 state.

## Queued Side Issues

1. `test/manual/test_double_sparsity_v32.py:752-754` still records `length_tokens` as a word-count proxy and derives `within_budget` from that proxy. Current evidence is safe per the Round-14 tokenizer sanity check, but the next substantive harness change should record actual chat input tokens and gate from those counts.

2. Pre-existing plan-process comments remain in the serve scripts (`development/serve_double_sparsity.sh:7-12`, `development/serve_native_nsa.sh:5-14`) and a few older manual-harness comments (`test/manual/test_double_sparsity_v32.py:47`, `863-864`, `933`). These are cosmetic, not AC blockers.

3. DS long-context R&D and AC-11 TTFT follow-up remain valid next-loop work: query-aware/learned selector, a decode kernel accepting wider DS selection, and TokenLabelTable/KV-budget reduction.

## Goal Tracker Review

Claude's Goal Tracker Update Request is accepted. The mutable tracker already reflects the Round-15 state: Plan Version 2 unchanged, no active original-plan evidence tasks, stale `next_loop_issues.md` resolved, Round-14 manual-harness wording resolved, token-count precision and pre-existing serve-header wording still queued, and AC-11 still recorded as a directional miss. I made no tracker edit during this review.

Do not emit the stop sentinel: Round 15 advanced the narrow handoff-reconcile objective, but original-plan residuals and queued follow-up work remain.
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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-16-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-16-summary.md

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
