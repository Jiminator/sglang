Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop6/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop6/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-18-contract.md

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
# Round 17 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: ACs: 9/10 addressed, 8/10 met | Forgotten items: 0 | Unjustified deferrals: 1

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-17-prompt.md`, `round-17-contract.md`, `round-17-summary.md`, `goal-tracker.md`, Round 14-16 summaries/reviews, commit `ece26eb52`, the AC-5 decode-profile artifacts, and the DS selection/cuda-graph/decode hot spots.

## Implementation Review

I found no high-signal correctness defect in the production code change itself. The patch is narrowly scoped to `_logical_score_kernel`: blocks whose first logical offset is already past that request's `seq_len` now store `-inf` and return before the per-head signature loads/dots (`python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:93-109`). That is semantically consistent with the existing final `tok_offs < seq_len_i` mask, preserves the launch grid, and does not touch the FlashMLA ABI lock.

The R17 evidence supports real mainline progress. The closed-batch profile shows conc-16 pure decode at 17.4 TPS/req before the patch (`runs/20260530_dsv32_loop6/ac5_decode_profile/ac5_decode_remediation.md:15-23`), the width microbench attributes the dominant removable cost to full-context DS selection over 163840 columns (`:25-39`), and the patched closed-batch re-measure improves conc-16 to 27.1 TPS/req while conc-8 clears 30 (`:54-63`). I also reran the registered DS unit file locally: `281 passed`.

## Mainline Gaps

1. **AC-5 strict remains incomplete.**

   Claude's own R17 artifact records the miss: conc-16 remains 27.1 TPS/req, below the strict `>=30` target (`runs/20260530_dsv32_loop6/ac5_decode_profile/ac5_decode_remediation.md:65-78`). No conc-32/64 TTFT remediation ran, and there is no final full AC-5 `NUM_PROMPTS=320`, conc `16 32 64`, 4096/512 client workload rerun with exact arrays and a fail-closed verifier. This is accepted progress, not completion.

2. **The R17 profiling artifact does not fully satisfy the round contract's served-step decomposition.**

   The contract asked for a served-workload profile breaking per-decode-step time into DS selection/top-k, FlashMLA KV decode, token-label write/update, and scheduler/interleave (`.humanize/rlcr/2026-05-30_06-27-19/round-17-contract.md:35-39`). R17 gives a closed-batch end-to-end step and a standalone selection-width microbench, but FlashMLA KV decode, token-label write/update, and scheduler/interleave remain an unmeasured residual. That was enough to justify the score-kernel early-exit, but before the next strict claim Claude needs the remaining component breakdown, especially after the residual top-k fix, so the next bottleneck is not guessed.

3. **AC-10 remains correctly gated and not met.**

   The original plan still requires Tier-2 recall R&D only after the Tier-1 spine is fully verified. Since AC-5 strict is still open, AC-10 must not start.

## Blocking Side Issues

1. **Residual full-width top-k is now the immediate AC-5 blocker.**

   R17 removed the score-kernel over-scan, but the first `torch.topk` still scans the full 163840-wide captured score row. The artifact estimates this residual as the remaining ~3.6 ms needed to move conc-16 from 27.1 TPS/req to at least 30 (`runs/20260530_dsv32_loop6/ac5_decode_profile/ac5_decode_remediation.md:65-73`). This is not queued cleanup; it blocks the current mainline objective.

## Queued Side Issues

1. Cross-node wrapper smoke remains future-gated. It does not block the single-node AC-5 remediation.

2. DSA-default conc-64 TPS around 29.4 remains a pre-existing DSA/client-SLO tension, not a DS-introduced blocker for this loop.

## Goal Tracker Audit

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified earlier. |
| AC-2 | MET | Feasibility budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, scale consumers, launcher, real-mask NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted 0.7 operating point, HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | R17 advances decode throughput but strict SLO still fails: conc-16 closed-batch is 27.1 TPS/req, and no final 16/32/64 client rerun exists. |
| AC-6 | MET | Verified in R12 under the user-approved non-regression/opt-in semantics. |
| AC-7 | MET / CHARACTERIZED | Verified in R15 as characterized/soft-met. |
| AC-8 | MET | Verified in R16: 70759-token `/generate` admitted HTTP 200 at lifted DS int8/mem0.7/radix-on. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. |
| AC-10 | NOT MET | Correctly gated behind AC-5 strict remediation and full Tier-1 verification. |

Forgotten items: none. Every original plan task is represented in Active, Completed, or the gated AC-10 path. Explicit tracker deferrals: none. The one unjustified deferral for review purposes is the unfinished residual AC-5 strict remediation/top-k lever being pushed to the next round; it is incomplete work, not a done condition.

## Required Implementation Plan

1. Keep AC-5 strict remediation as the sole mainline. Do not start AC-10 and do not collect more AC-7/AC-8 evidence.

2. Finish the residual top-k over-scan in `retrieve_topk_graph_safe` without capping context and without changing `max_top_k == dsa_index_topk == 2048`. Implement an exact seq-aware blocked top-k path, not an approximate partial selector: each block-local candidate set must be large enough that a request whose top 2048 scores all fall in one block still returns the same 2048 positions as the current monolithic `torch.topk`.

3. Put the blocked top-k scratch in `DSGraphState`/`allocate_graph_state`, reusing or replacing the existing `scratch_partial_*` fields with production-sized buffers. The graph-safe path must stay allocation-free after warmup, keep output indices in logical sequence order, and early-skip blocks entirely past each row's `seq_len`, just as the score kernel now does.

4. Add regression coverage before hardware: compare monolithic top-k vs blocked top-k on adversarial CPU/CUDA fixtures where all winning scores are in one block, across boundary `seq_len` values, mixed per-request lengths, padding, and `per_request_valid` masks. Include a CUDA graph replay/zero-allocation test for the production dtypes.

5. Re-run the R17 selection-width microbench and closed-batch decode at the same lifted DS int8/mem-0.7/radix-on point. The required gate before scheduling work is conc-16 closed-batch `>=30 TPS/req`; publish before/after component numbers, including the previously missing FlashMLA/token-label/scheduler residual.

6. Only after conc-16 passes, tune conc-32/64 TTFT at the same locked operating point, recording any scheduling flag changes as plan evolution with before/after server-info sidecars.

7. Rerun the full AC-5 client workload (`NUM_PROMPTS=320`, conc `16 32 64`, 4096 ISL / 512 OSL / ~55% cache, radix-on, TP=8). Publish exact per-request arrays, request-time attribution, server-info sidecars, and a fail-closed verifier. A strict claim requires every published trial to pass `P99 TTFT < 22.0 s` and per-request TPS `>=30`.

8. Start AC-10 only after AC-5 strict is verified and AC-3 through AC-9 remain verified.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 17 Review.
- Added a `17-review` Plan Evolution row.
- Corrected the R17 row from "commit pending" to commit `ece26eb52`.
- Kept task6/AC-5 Active and refreshed its note with the verified R17 progress plus the remaining top-k blocker.
- Updated the AC-5 blocking-side-issue resolution path.
- Left AC-10 active/gated and moved no task to Completed and Verified.

## Validation Performed

- `git log --oneline -30`
- `git show --stat --oneline ece26eb52`
- `git diff 9915630ca..ece26eb52 -- python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`
- Inspected `round-17-contract.md`, `round-17-summary.md`, `goal-tracker.md`, and Round 14-16 summaries/reviews.
- Inspected `ac5_decode_remediation.md`, `ds_closed_batch_decode*.txt`, `closed_batch_b*.json`, `selection_width_microbench.py/.json`, `verify_early_exit.py`, and `get_server_info_ds*.json`.
- Inspected `selection_kernel.py`, `cuda_graph.py`, `dsa_backend.py`, and CUDA graph runner sequence-length handling relevant to the residual top-k lever.
- Ran `pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py` → `281 passed`.
- Ran `git diff --check`.

NOT COMPLETE
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-18-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-18-summary.md

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
