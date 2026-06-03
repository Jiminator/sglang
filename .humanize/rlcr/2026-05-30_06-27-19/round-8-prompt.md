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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-8-contract.md

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
# Round 7 Review Result

Mainline Progress Verdict: ADVANCED

ACs: 6/10 addressed | Forgotten items: 0 | Unjustified deferrals: 0

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-7-prompt.md`, `round-7-contract.md`, Round 4-6 summaries/reviews, `goal-tracker.md`, commit `51dd009b8`, the R7 AC-5 artifacts under `runs/20260530_dsv32_loop6/client_slo_int8/`, and the local full R6/R7 source logs still present at `/tmp/ac5/boot_radixon.log` plus the ignored JSONL files.

## Implementation Review

Round 7 did move the mainline forward: the report no longer claims a shippable or fully validated result, `forward_duration` is no longer used as a first-token prefill term, the decode-batch evidence corrects the R6 `#running-req: 19-20` claim, and the new tracked files make the headline summary easier to audit.

I cannot verify task6/AC-5 yet. The two R6 blockers are improved but not closed. The evidence bundle still cannot independently recompute the benchmark percentiles, and the attribution addendum has internally inconsistent row accounting plus a demonstrably unsafe per-concurrency bucketing method.

## Mainline Gaps

1. **AC-5 benchmark evidence is still summary-only, not an exact recomputation source.**

   `runs/20260530_dsv32_loop6/client_slo_int8/ac5_evidence_addendum.txt:1-6` says the AC-5 evidence is recomputable without the gitignored JSONLs, but each conc section records only aggregate summary values: min/p50/p90/p99/max TTFT, TPOT summaries, ITL summaries, and length summaries (`ac5_evidence_addendum.txt:8-39`). It does not include the exact `ttfts`, `tpots`, `itls`, `input_lens`, `output_lens`, or another durable percentile source such as sorted numeric arrays, tail source values with an explicit percentile algorithm, a redacted JSONL-derived metrics dump, or JSONL checksums tied to a reproducible extraction command.

   That does not satisfy the R7 contract's "exact arrays or their recomputable summary" requirement. A future checkout still cannot recompute p99 TTFT/TPOT/ITL from committed artifacts; it can only compare one summary file against another summary file. The local ignored JSONLs do contain the arrays, but they remain ignored by `.gitignore` and absent from the tracked acceptance bundle.

   Required fix:
   - Add a tracked numeric evidence artifact derived from the JSONLs, one record per conc, containing exact arrays or sorted arrays for `ttfts`, `tpots`, `itls`, `input_lens`, `output_lens`, plus the `errors` array proof.
   - Include the extraction command or script invocation, the percentile method, and SHA256 checksums of the source ignored JSONLs used for the extraction.
   - Update `ac5_evidence_addendum.txt` and `client_slo_report.md` so "recomputable" points at that exact source, not only at rounded summary values.

2. **The corrected attribution is still not clean: row counts are contradictory and the wall-clock windows mis-bucket requests.**

   `attribution_per_conc.txt:3-8` says `967 parsed`, `3 HEALTH_CHECK`, `5 negative queue`, and `959 valid`, then states `Valid=959 > 3x320=960`. That arithmetic is impossible: 959 is less than 960. The report repeats the same "the >960 nominal is per-conc warmup requests" framing at `client_slo_report.md:33-34`.

   Reprocessing the full local log shows why this matters:
   - The full log has 967 `ReqTimeStats` rows, 3 health-check rows, and 5 negative `queue_duration` rows.
   - It has exactly 960 benchmark-shaped rows with `output_len=512`, matching 3 x 320 completed client requests.
   - After dropping the 5 negative benchmark rows, there are 955 valid benchmark rows, not 959. The 959 count includes non-benchmark warmup/server rows (`output_len=8` and `output_len=32`).
   - The addendum's own per-conc windows have counts 306 / 337 / 316 (`attribution_per_conc.txt:21`, `:28`, `:35`) while the client completed 320 requests per conc. Those counts are not reconciled.
   - The window split starts at the minimum server `entry_time`, which includes the server readiness request and benchmark warmup, so the first window includes non-benchmark rows and the second window absorbs some c16 rows. In the full log, stable benchmark grouping by request shape/prefix yields c16=320 valid, c32=320 valid, and c64=315 valid after the 5 negative rows.

   This means the p99 queue values may be close, but the attribution artifact still does not meet the "expected rows vs parsed rows" and "per-conc attribution" bar. It also leaves stale contradictory evidence in `client_slo_metrics.txt:8-12`, which still reports the old aggregate attribution and the old `#running-req 19-20` TPS explanation.

   Required fix:
   - Rebuild `attribution_per_conc.txt` from benchmark-shaped rows only (`output_len=512`), with warmup/server readiness/health rows reported separately.
   - Use a reliable per-conc grouping source. Do not use `T0=min(entry_time)+cumulative durations` if T0 includes readiness/warmup traffic. Use benchmark invocation boundaries from `/tmp/ac5/benchmark.log`, source JSONL metadata, request shape/prefix signatures, or a joined request-id source if available.
   - Reconcile counts explicitly: total parsed, health, non-benchmark warmups, benchmark rows, invalid negative benchmark rows by conc, and valid benchmark rows by conc.
   - Update `client_slo_report.md`, `client_slo_metrics.txt`, and the new BitLesson text so they no longer claim `959 > 960` or leave the obsolete R6 attribution in the acceptance bundle.

3. **The original Loop-6 plan remains incomplete and the strict SLO still fails.**

   This is not a Round 7 regression, but it still prevents any final `COMPLETE`. AC-5 is partial, not verified. AC-6 hardware product proof, AC-7 lifted-point DS+DSA re-sweep, AC-8 lifted 64K servability probe, AC-9 real-token-count harness edit plus live rerun, and gated AC-10 remain pending. The strict client SLO also still fails: conc 32/64 TTFT are above 22 s, and per-request TPS is below 30 at every conc.

   Directive implementation plan:
   - First repair AC-5 evidence and attribution exactly as described above. Keep the report language as "directional characterization" and keep the strict SLO miss explicit.
   - Then complete AC-6: boot DSA default with no DS flags, track `/get_server_info` plus server excerpts proving `enable_double_sparsity=false` and no DS `TokenLabelTable`, run the client SLO workload showing DSA-default behavior/perf unchanged, and boot DS opt-in proving the compact int8 path toggles on.
   - Complete AC-9 before or alongside the next live hardware pass: edit `test/manual/test_double_sparsity_v32.py` so artifacts record actual `usage.prompt_tokens` as `input_tokens`, compute `within_budget` from that token count, rename the old proxy to `length_words`, and fail closed if usage is absent or inconsistent. Do not change DS-fair thresholds. Rerun and copy artifacts into `runs/20260530_dsv32_loop6/`.
   - Complete AC-7: run the 3-trial DS+DSA lifted-point sweep at conc 16/32/64 with 120 s warmup and 600 s windows, radix-on proven on both sides, and refresh `ac11_resweep.md` / `ac11_analysis.md`.
   - Complete AC-8: run the lifted-0.7 ~70K-token `/generate` probe and record either HTTP 200 with `max_total_num_tokens` and no instability, or a characterized new ceiling. Do not silently re-record the old 400.
   - Start AC-10 only after AC-3 through AC-9 are verified. Implement the adjustable-`top_k` sparse-matmul path and record NIAH recall deltas vs the Loop-5 DS baseline, with TPS/TTFT cost.

## Blocking Side Issues

1. **AC-5 evidence/attribution still blocks task6 verification.**

   The R7 addenda are useful, but they do not fully close the R6 review requirements. Summary-only benchmark percentiles and mis-bucketed attribution cannot support a verified AC-5 acceptance bundle.

2. **Stale tracked AC-5 metrics contradict the corrected report.**

   `client_slo_metrics.txt:8-12` still contains the R6 aggregate attribution and old decode-batch explanation. Because `client_slo_report.md:15-19` lists it as part of the durable evidence bundle, it should be updated or clearly marked superseded when the attribution is rebuilt.

3. **Strict SLO failure remains a mainline blocker for the ultimate goal.**

   Conc 32/64 TTFT and all-conc per-request TPS failures are not queued cleanup. They block the strict `P99 TTFT < 22 s AND >=30 TPS/req` done criterion and must remain visible while AC-6 through AC-9 proceed.

## Queued Side Issues

None newly added. The row-accounting and stale-metrics issues are not cosmetic; they directly affect AC-5 verification.

## Goal Alignment Check

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Decision doc verified in earlier reviews. |
| AC-2 | MET | Feasibility budget and binding int8 lever verified in earlier reviews. |
| AC-3 | MET | Int8 table, scale-sidecar consumers, launcher, real-mask NIAH, and microbench verified by R3. |
| AC-4 | MET | Mem-fraction lift and durable no-OOM evidence verified by R5. |
| AC-5 | PARTIAL | Real DS int8/0.7 radix-on run with strong TTFT movement; R7 report language improved, but benchmark recomputation source and attribution row accounting still need correction; strict SLO not met. |
| AC-6 | PARTIAL | Dev checks exist; hardware DSA-default/no-table/SLO product proof remains pending. |
| AC-7 | NOT MET | Lifted-point 3-trial DS+DSA re-sweep remains pending. |
| AC-8 | NOT MET | Lifted-point ~70K-token servability probe remains pending. |
| AC-9 | NOT MET | Harness still needs real `usage.prompt_tokens` budget assertion and rerun. |
| AC-10 | NOT MET | Correctly gated behind full Tier-1 completion. |

Forgotten items: none. Every original plan task is represented in Active, Completed, or the gated AC-10 path. Deferred items: none explicit; AC-10 is gated, not deferred. Plan evolution is valid only as a directional characterization. R7 must not be treated as closing AC-5 until the evidence/attribution corrections above land.

## Goal Tracker Update

Updated `.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md` mutable section:

- Plan version moved to Round 7 Review.
- Added an R7-review plan-evolution row.
- Kept task6 active as `partial (R7 review)`.
- Changed the R6 AC-5 evidence and attribution blockers from `RESOLVED` to `PARTIAL`.
- Left AC-6 through AC-10 active/pending and kept the strict-SLO miss as a blocking side issue.

## Validation Performed

- `git show --stat --oneline 51dd009b8`
- `git show --name-only --format=short 51dd009b8`
- `git diff --check 8883848e9..51dd009b8`
- `git ls-files runs/20260530_dsv32_loop6/client_slo_int8`
- `git status --ignored --short runs/20260530_dsv32_loop6/client_slo_int8`
- Inspected `client_slo_report.md`, `ac5_evidence_addendum.txt`, `attribution_per_conc.txt`, `decode_batch_excerpt.txt`, `client_slo_metrics.txt`, `.meta.json` sidecars, `reqtimestats_excerpt.txt`, `round-7-contract.md`, `round-7-summary.md`, and `goal-tracker.md`.
- Parsed the local full `/tmp/ac5/boot_radixon.log` to verify request-time row counts, health rows, negative queue rows, benchmark-shaped rows, and the mis-bucketed window counts.
- Parsed the local ignored JSONL files enough to confirm the exact arrays exist locally but are not tracked.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-8-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-8-summary.md

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
