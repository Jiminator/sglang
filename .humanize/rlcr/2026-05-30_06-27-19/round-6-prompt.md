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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-6-contract.md

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
# Round 5 Review Result

Mainline Progress Verdict: ADVANCED

ACs: 5/10 addressed (4 met, 1 partial) | Forgotten items: 0 | Unjustified deferrals: 0

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-5-prompt.md`, `round-5-contract.md`, Round 2-4 summaries/reviews, `goal-tracker.md`, commit `91e9c20a3`, the AC-4 addendum and tracked mem-fraction artifacts under `runs/20260530_dsv32_loop6/`, plus the benchmark and harness entrypoints that remain in the original plan.

## Implementation Review

No high-signal Round-5 defect found. The R4 blocker is resolved:

- Commit `91e9c20a3` changes only acceptance artifacts under `runs/20260530_dsv32_loop6/`; no production code changed.
- `git diff --check 75e68053f..91e9c20a3` is clean, and the new NVML proof is tracked as `.txt`, not the ignored `.csv`.
- The HBM addendum records the measurement method and the accepted `torch_used`/NVML-per-process proxy for allocator stats that are not HTTP-exposed (`ac4_hbm_budget_addendum.md:11-22`), then closes the per-rank budget with named components plus residual for f=0.6/0.7/0.8 (`ac4_hbm_budget_addendum.md:24-41`).
- The no-OOM proof is now durable: client summary is 97/97 ok (`stress_0.7_client.txt:1-4`), server excerpt shows sustained decode at `#running-req: 32` with OOM count 0 (`stress_0.7_server_excerpt.txt:8-16`), and the 29-sample NVML series rises to a plateau where last == max (`nvml_timeseries_0.7.txt:1-29`).
- `/get_server_info` for the 0.7 run is tracked and records `mem_fraction_static=0.7`, `enable_double_sparsity=true`, `signature_dtype=int8`, and `max_total_num_tokens=396096`.

I am not reopening the exact `torch.cuda.memory_reserved()/memory_allocated()` wording from R4 because the Round-5 contract explicitly allowed `torch_used = total - avail` plus NVML per-process as the external allocator proxy when the serve API does not expose those calls. The addendum makes that limitation explicit and the budget closes.

## Mainline Gaps

1. The original Loop-6 plan is still incomplete after R5.

   This is not a Round-5 regression: R5 had one narrow AC-4 evidence objective and completed it. It still prevents a final `COMPLETE`: AC-5, AC-6 hardware proof, AC-7, AC-8, AC-9, and gated AC-10 remain pending in the tracker (`goal-tracker.md:126-131`).

   Directive next implementation plan:
   - Make task6/AC-5 the next single mainline objective.
   - Produce or validate a DS radix fixture state for the exact int8 lifted operating point, then boot DS with `SIGNATURE_DTYPE=int8 MEM_FRACTION_STATIC=0.7 RADIX_FIXTURE_ARTIFACT=<state> EXTRA_SERVER_ARGS="--enable-request-time-stats-logging" bash development/serve_double_sparsity.sh`.
   - Run the full client workload with `NUM_PROMPTS=320 CONCURRENCIES="16 32 64" TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 MODE=double_sparsity bash development/benchmark.sh`.
   - Copy all JSONL and `.meta.json` sidecars into `runs/20260530_dsv32_loop6/`; prove radix-on from sidecars/server args, and disclose every trial.
   - Write `client_slo_report.md` with strict `<22.0` TTFT P99 and `>=30` per-request TPS/req at conc 16/32/64. Hard pass requires every trial to pass; directional progress may use the pre-declared median rule only with the worst trial disclosed.
   - Attribute TTFT using the server request-time-stat logs (`queue_duration` vs `forward_duration`) together with bench JSONL `ttfts`/`itls`. If the fields are missing, record the run but do not claim the spine is validated.
   - Then complete task7/task8 using the same benchmark outputs where possible: DSA-default/no-DS-table hardware proof, DSA default SLO, and the AC-11 DS+DSA comparison with radix-on on both sides.
   - Complete task9 with a lifted-0.7 ~70K-token `/generate` probe, recording either HTTP 200 or a characterized new ceiling.
   - Complete task10 by editing `test/manual/test_double_sparsity_v32.py` so NIAH artifacts record actual `usage.prompt_tokens` as `input_tokens`, compute `within_budget` from that token count, and fail closed if usage is absent or inconsistent; then rerun on the live servers and copy artifacts.
   - Start task11/AC-10 only after AC-3 through AC-9 are verified.

## Blocking Side Issues

None for the current handoff. The prior AC-4 evidence blocker is resolved by `91e9c20a3`, and task6 is now ungated.

## Queued Side Issues

None newly found. The R4 trailing-whitespace issue was fixed; `git diff --check` is clean for the R5 range.

## Goal Alignment Check

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | `ds_on_v32_decision.md`, verified in earlier reviews. |
| AC-2 | MET | `footprint_feasibility.md`, binding int8 same-`label_dim` lever. |
| AC-3 | MET | Int8 table, scale-sidecar consumers, launcher, real-mask NIAH, and microbench verified by R3. |
| AC-4 | MET | R4 sweep plus R5 durable addendum and stress artifacts verified in this review. |
| AC-5 | NOT MET | Full client-SLO benchmark/report with radix-on proof and admission/prefill attribution has not run. |
| AC-6 | PARTIAL | Dev checks exist; hardware DSA-default/no-table and SLO proof remains pending. |
| AC-7 | NOT MET | Lifted-point AC-11 DS+DSA re-sweep remains pending. |
| AC-8 | NOT MET | Lifted-point ~70K-token servability probe remains pending. |
| AC-9 | NOT MET | Harness still uses word-count proxy; real `usage.prompt_tokens` gate and rerun remain pending. |
| AC-10 | NOT MET | Correctly gated behind full Tier-1 completion. |

Forgotten items: none. Every original plan task is still present in Active, Completed, or Deferred. Deferred items: none; AC-10 is gated/pending, not deferred. Plan evolution is valid: R5 only completed the AC-4 evidence addendum and did not change the operating point or weaken AC-5.

## Goal Tracker Update

Updated `.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md` mutable section:

- Plan version moved to Round 5 Review.
- Added an R5-review plan-evolution row verifying the AC-4 evidence addendum.
- Moved task5/AC-4 to Completed and Verified.
- Left task6 through task11 active and pending, with task6 now ungated as the next mainline task.

## Validation Performed

- `git show --stat 91e9c20a3`
- `git ls-files runs/20260530_dsv32_loop6`
- `git status --ignored --short runs/20260530_dsv32_loop6/memfraction_sweep_int8`
- `git diff --check 75e68053f..91e9c20a3`
- Inspected the AC-4 addendum, boot excerpts, `/get_server_info`, stress client/server logs, and NVML time series.
- Verified the NVML plateau with `awk`: 29 samples, last used = max used = 1041136 MiB.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-6-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-6-summary.md

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
