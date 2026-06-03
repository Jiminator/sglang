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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-20-contract.md

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
# Round 19 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: ACs: 9/10 addressed, 8/10 met | Forgotten items: 0 | Unjustified deferrals: 0

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-19-contract.md`, `round-19-summary.md`, `goal-tracker.md`, Round 16-18 summaries/reviews, commit `7f896b454`, `python/sglang/bench_serving.py`, benchmark tests, `docs/developer_guide/bench_serving.md`, and `development/CLIENT_SLOS.md`.

## Implementation Review

I found no high-signal correctness defect in the R19 patch itself.

The production change addresses the concrete R18 false-success path. In `async_request_sglang_generate`, an HTTP 200 stream that never emits a non-empty `text` chunk no longer leaves `output_len` at the requested value and marks the request successful; it now records `success=False`, `output_len=last_output_len` (0 for the R18 case), and an error (`python/sglang/bench_serving.py:696-716`). That directly fixes the fabricated "512 generated tokens" accounting described in the R19 summary.

The second guard in `calculate_metrics` raises when a streaming run has `completed > 0` but no ITLs and no first-token timings (`python/sglang/bench_serving.py:1049-1065`). That matches the benchmark docs' contract that `bench_serving` measures TTFT/ITL and can output per-request arrays via `--output-details` (`docs/developer_guide/bench_serving.md:5-9`, `:132-142`, `:160-172`). It also matches AC-5's need for exact per-request arrays and measured P99 TTFT/TPS, not inferred values.

Validation matched Claude's claim:

| Check | Result |
|-------|--------|
| `pytest -q test/registered/unit/development/test_bench_serving_timing.py` | 12 passed |
| `pytest -q test/registered/bench_fn/test_bench_serving_reasoning_stream.py` | 9 passed |
| `git diff --check 7f896b454^..7f896b454` | clean |

One limitation: the new regressions mainly exercise `calculate_metrics` with constructed outputs (`test_bench_serving_timing.py:616-682`), not a stubbed `async_request_sglang_generate` HTTP-200 empty stream. The code path is simple enough that I am not marking this as a blocking defect, but the next touch to this area should add direct request-function coverage.

## Mainline Gaps

1. **AC-5 is still not validated.**

   AC-5 requires the full client workload through `development/benchmark.sh`: 4096 ISL / 512 OSL, conc 16/32/64, TP=8, radix-on proof, exact arrays, strict numbers, and admission/prefill attribution (`development/loop6/refined_plan_v1.md:66-75`). R19 did not boot the server, did not rerun the full-context workload, did not publish new measured TTFT/TPS arrays, and did not publish attribution/component breakdown. This also misses the R19 contract's concrete success criteria 2 and 3 (`round-19-contract.md:35-41`).

2. **The live empty-stream root cause is still open.**

   R19 fixes the accounting failure mode, but Claude's own summary says the runtime cause of the empty stream still needs a small live reproduction before `bench_serving` can produce valid arrays. Since AC-5 depends on measured P99 TTFT and per-request details, this remains a blocking issue for the next full client run.

3. **The full-context conc-16 throughput blocker remains.**

   The original plan fixes the target at the full Option-B operating point and says the serve/bench flags are fixed except for the deliberate memory lever (`development/loop6/refined_plan_v1.md:5-7`, `:118-122`). R18's `--context-length 8192` result remains bounded-context characterization only. R19 did not implement the exact full-context blocked/top-k path or produce a full-context conc-16 `>=30 TPS/req` result. The requested owner decision is legitimate to surface, but it is not a tracker rescope until the owner explicitly accepts bounded context as the deployment target.

4. **AC-10 remains correctly gated and not met.**

   The original plan includes AC-10 only after the full Tier-1 spine lands (`development/loop6/refined_plan_v1.md:91-93`). With AC-5 still partial, AC-10 must not start, and the loop cannot complete.

## Goal Tracker Audit

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified in earlier reviews. |
| AC-2 | MET | Footprint budget and binding int8 lever verified in earlier reviews. |
| AC-3 | MET | Compact int8 table, scale consumers, launcher, real-mask NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted DS int8/mem-0.7 point, full HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | R19 fixed a measurement prerequisite, but no valid full-context client workload, fresh arrays, attribution, component breakdown, or full-context top-k remediation landed. |
| AC-6 | MET | Verified in R12 under the approved DSA-default non-regression / DS opt-in semantics. |
| AC-7 | MET / CHARACTERIZED | Verified in R15 as characterized/soft-met. |
| AC-8 | MET | Verified in R16 at the lifted full-context DS int8/mem-0.7 point. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. |
| AC-10 | NOT MET | Correctly gated behind AC-5 and full Tier-1 verification. |

Forgotten items: none. The original tasks are represented in Active, Completed, or the gated AC-10 path. I did not find a current tracker item marked complete without verification. The R18 bounded-context pass framing was already rejected in the R18 review and remains rejected here.

Deferred items: none in the Explicitly Deferred section. No deferral should be added for AC-5 or AC-10; both are incomplete work.

Goal completion summary:

```text
Acceptance Criteria: 8/10 met (0 deferred)
Active Tasks: 2 remaining (task6/AC-5, task11/AC-10)
Estimated remaining rounds: 4-6 without an explicit rescope; at least 2-3 even with a bounded-context rescope because measured client arrays and AC-10 still remain.
Critical blockers: live full-context bench streaming/root-cause, full-context client workload evidence, full-context conc-16 top-k/throughput path or explicit owner rescope, AC-10 gate.
```

## Mainline Drift Audit

R19's stated objective was clear and singular: AC-5 closure at the accepted full-context operating point with measured client evidence. The executed work was narrower than that contract, but it addressed a true blocking side issue identified in R18: the benchmark could fabricate successful 512-token completions from HTTP-200 empty streams.

This is mainline-serving progress, not a queued cleanup. It does not complete AC-5.

```text
Mainline Progress Verdict: ADVANCED
Blocking Side Issues: 3
Queued Side Issues: 2
```

Blocking side issues:

| Issue | Why blocking |
|-------|--------------|
| Live empty-stream/window-driver root cause | Without valid `ttfts`/`itls`/`generated_texts`, AC-5 cannot be measured. |
| Full-context top-k/decode TPS | Full-context conc-16 is still below the owner-selected strict TPS axis unless the exact top-k path lands or the owner explicitly accepts bounded context. |
| Full-context AC-5 run and verifier missing | The plan requires measured full workload evidence with sidecars and attribution. |

Queued side issues:

| Issue | Why queued |
|-------|------------|
| Cross-node wrapper smoke | Loop 6 SLO claim is single-node TP=8; this only gates future cross-node artifacts. |
| DSA-default conc-64 TPS around 29.4 | Pre-existing DSA/H200 ceiling; useful characterization, not a DS-introduced blocker for AC-6. |

## Progress Stagnation Check

I am not issuing STOP this round. R18 was a real stall, but R19 did address one of the two R18 blocking findings with production code and tests. That is meaningful forward movement.

The stagnation risk is now high. The residual full-context top-k/client-run issue has recurred since R17, and another round that only restates the bounded-context-vs-research-kernel tradeoff without either an explicit owner rescope or new full-context measured evidence should be treated as stagnation.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 19 Review.
- Added a `19-review` Plan Evolution row accepting the R19 fail-closed accounting fix but rejecting AC-5 closure.
- Updated task6/AC-5 Active status to mention the R19 prerequisite fix and the remaining live streaming/root-cause, full workload, attribution, and top-k gaps.
- Added a Blocking Side Issue for the remaining live full-context benchmark array/root-cause problem.
- Moved no task to Completed and Verified.
- Rejected the bounded-context-vs-research-kernel update as a tracker rescope until the owner explicitly changes the target.

## Required Action Items

### Mainline Gaps

1. Keep AC-5/task6 as the sole mainline until it is verified. Do not start AC-10.

2. Produce a small live reproduction of the full-context streaming empty-array failure after the R19 accounting patch. Fix the runtime/window-driver/abort cause so the benchmark produces real `ttfts`, `itls`, `generated_texts`, `output_lens`, and errors arrays.

3. Either implement the exact full-context blocked/top-k remediation required by the current plan, with adversarial and graph-safety regression coverage, or obtain an explicit owner rescope that accepts bounded context as the conc-16 deployment target. Without that rescope, `--context-length 8192` remains characterization only.

4. Rerun the full AC-5 client workload at the accepted operating point: `NUM_PROMPTS=320`, conc 16/32/64, 4096/512, radix-on, TP=8. Publish exact arrays, sidecars, measured attribution, component breakdown, and a fail-closed verifier that checks completion counts, output lengths, TTFT/TPS, ITL source, errors, and radix proof.

5. Start AC-10 only after AC-5 is verified and AC-3 through AC-9 remain verified.

### Blocking Side Issues

1. Treat the live streaming/root-cause failure as blocking until the full-context benchmark emits valid arrays.

2. Treat the bounded-context operating point as blocked from AC-5 completion unless an explicit owner rescope lands.

3. Treat any AC-5 report without measured attribution as invalid for spine validation.

### Queued Side Issues

1. Keep cross-node wrapper smoke queued for future remote artifacts.

2. Keep DSA-default conc-64 TPS tension queued as a separate DSA/client-SLO issue unless the owner makes it part of this loop.

## Validation Performed

- `git log --oneline -30`
- `git show --stat --oneline --decorate 7f896b454`
- Inspected `git show --unified=80 7f896b454 -- python/sglang/bench_serving.py`
- Inspected `git show --unified=80 7f896b454 -- test/registered/unit/development/test_bench_serving_timing.py`
- Inspected `round-19-contract.md`, `round-19-summary.md`, `goal-tracker.md`, and Round 16-18 summaries/reviews.
- Inspected `docs/developer_guide/bench_serving.md`, `development/CLIENT_SLOS.md`, and the original AC-5 plan text.
- Ran `pytest -q test/registered/unit/development/test_bench_serving_timing.py` -> 12 passed.
- Ran `pytest -q test/registered/bench_fn/test_bench_serving_reasoning_stream.py` -> 9 passed.
- Ran `git diff --check 7f896b454^..7f896b454` -> clean.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-20-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-20-summary.md

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
