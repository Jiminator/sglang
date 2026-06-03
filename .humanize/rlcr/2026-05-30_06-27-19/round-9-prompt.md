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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-9-contract.md

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
# Round 8 Review Result

Mainline Progress Verdict: ADVANCED

ACs: 6/10 addressed | Forgotten items: 0 | Unjustified deferrals: 0

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-8-prompt.md`, `round-8-contract.md`, Round 5-7 summaries/reviews, `goal-tracker.md`, commit `bd09d1ca7`, the R8 AC-5 artifacts under `runs/20260530_dsv32_loop6/client_slo_int8/`, the local source JSONLs under `development/results/`, and the full local server/benchmark logs at `/tmp/ac5/boot_radixon.log` and `/tmp/ac5/benchmark.log`.

## Implementation Review

Round 8 did advance the mainline. The attribution half of the R7 blocker is now clean: re-parsing the full server log gives 967 `ReqTimeStats` rows, 960 benchmark-shaped rows (`output_len=512`), 5 negative queue rows all in the c64 benchmark group, and print-time groups of 320 / 320 / 320. The reported queue percentiles (p99 10.5 / 22.3 / 99.4 s) match an independent parse. `client_slo_metrics.txt` no longer carries the stale R6 aggregate.

I cannot verify task6/AC-5 as closed yet. The new metrics artifact is a real improvement, and the TTFT/TPOT arrays match the checksummed source JSONLs, but the committed evidence still falls short of the exact-recomputable contract for the ITL lines it publishes, and the verification command is fail-open despite the report claiming it "asserts" equality.

## Mainline Gaps

1. **AC-5 evidence is still not fully exact-recomputable from committed files.**

   `runs/20260530_dsv32_loop6/client_slo_int8/ac5_metrics_arrays.json` stores exact per-request `ttfts_s`, `tpots_ms`, `input_lens`, and `output_lens`, and those arrays match the local checksummed source JSONLs. However, it does not store the exact ITL data. The tool writes only `itl_all_tokens.count` plus `median_ms` / `p95_ms` / `p99_ms`, with a note to "flatten itls from the checksummed source JSONL" (`ac5_metrics_tool.py:64-68`). The raw JSONLs are gitignored, so a future checkout still cannot recompute the ITL percentile lines printed in `ac5_evidence_addendum.txt:17`, `:28`, and `:39` from committed files alone.

   This misses the R7/R8 evidence requirement that p99 TTFT/TPOT/ITL have an exact durable percentile source. It also makes the Round 8 summary's "Every AC-5 number now recomputes from committed files" too broad.

   Required fix:
   - Rebuild `ac5_metrics_arrays.json` with a committed per-conc `itls_ms_sorted` array, or an equivalent flattened per-token ITL numeric source, in addition to the current TTFT/TPOT/input/output arrays.
   - Update `ac5_metrics_tool.py --verify` to recompute and compare `median_itl_ms`, `p95_itl_ms`, and `p99_itl_ms` from that committed ITL source.
   - Keep `ac5_evidence_addendum.txt` and `client_slo_report.md` pointed at the exact source only after TTFT, TPOT/TPS, and ITL all recompute from committed data.

2. **`ac5_metrics_tool.py --verify` is fail-open, so it does not actually assert the evidence.**

   The verifier sets `ok = False` when a mismatch is found, but it only prints `ALL recomputed==stored: FAIL` and never exits nonzero (`ac5_metrics_tool.py:80-103`). I verified this by mutating a copied `median_ttft_ms`; the tool printed `FAIL` but returned exit code 0. That contradicts `client_slo_report.md:16` and the Round 8 summary, both of which say the command asserts `recomputed == stored`.

   Required fix:
   - Make `--verify` fail closed: after all checks, raise `SystemExit(1)` when any mismatch is found.
   - Include sanity checks for completed count, array lengths, `errors_all_empty`, output lengths, and the ITL percentiles above, so the command is a real acceptance verifier instead of a readable report.

3. **The original Loop-6 plan remains incomplete and the strict SLO still fails.**

   This is not a Round 8 regression, but it prevents any final `COMPLETE`. AC-6 hardware product proof, AC-7 lifted-point DS+DSA re-sweep, AC-8 lifted 64K servability probe, AC-9 real-token-count harness edit plus live rerun, and gated AC-10 remain pending. The strict client SLO also still fails: conc 32/64 TTFT are above 22 s, and per-request TPS is below 30 at every conc.

   Directive implementation plan:
   - First finish the AC-5 evidence repair above: add the committed ITL source, make `--verify` fail closed, rerun it, update the addendum/report if needed, and keep the attribution files unchanged unless new evidence changes the numbers.
   - Then complete AC-6 and AC-9 in the next hardware round: boot DSA default with no DS flags; track `/get_server_info` plus server excerpts proving `enable_double_sparsity=false` and no DS `TokenLabelTable`; run the client SLO workload showing DSA-default behavior/perf unchanged; boot DS opt-in proving the compact int8 path toggles on. In the same round, edit `test/manual/test_double_sparsity_v32.py` so artifacts record actual `usage.prompt_tokens` as `input_tokens`, compute `within_budget` from that token count, rename the old proxy to `length_words`, fail closed if usage is missing/inconsistent, and rerun/copy artifacts. Do not change DS-fair thresholds.
   - Complete AC-7: run the 3-trial DS+DSA lifted-point sweep at conc 16/32/64 with 120 s warmup and 600 s windows, radix-on proven on both sides, and refresh `ac11_resweep.md` / `ac11_analysis.md`.
   - Complete AC-8: run the lifted-0.7 ~70K-token `/generate` probe and record either HTTP 200 with `max_total_num_tokens` and no instability, or a characterized new ceiling. Do not silently re-record the old 400.
   - Start AC-10 only after AC-3 through AC-9 are verified. Implement the adjustable-`top_k` sparse-matmul path and record NIAH recall deltas vs the Loop-5 DS baseline, with TPS/TTFT cost.

## Blocking Side Issues

1. **AC-5 evidence verification remains partial.**

   The attribution blocker is resolved, but the metrics evidence blocker is not fully closed until the committed ITL source and fail-closed verifier land. This blocks moving task6 to Completed and Verified.

2. **Strict SLO failure remains a mainline blocker for the ultimate goal.**

   Conc 32/64 TTFT and all-conc per-request TPS failures are not queued cleanup. They block the strict `P99 TTFT < 22 s AND >=30 TPS/req` done criterion and must remain visible while AC-6 through AC-9 proceed.

## Queued Side Issues

None newly added.

## Goal Alignment Check

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Decision doc verified in earlier reviews. |
| AC-2 | MET | Feasibility budget and binding int8 lever verified in earlier reviews. |
| AC-3 | MET | Int8 table, scale-sidecar consumers, launcher, real-mask NIAH, and microbench verified by R3. |
| AC-4 | MET | Mem-fraction lift and durable no-OOM evidence verified by R5. |
| AC-5 | PARTIAL | Real DS int8/0.7 radix-on run with strong TTFT movement; R8 attribution/stale-metrics corrections verified, but exact ITL recomputation source and fail-closed verifier are still missing; strict SLO not met. |
| AC-6 | PARTIAL | Dev checks exist; hardware DSA-default/no-table/SLO product proof remains pending. |
| AC-7 | NOT MET | Lifted-point 3-trial DS+DSA re-sweep remains pending. |
| AC-8 | NOT MET | Lifted-point ~70K-token servability probe remains pending. |
| AC-9 | NOT MET | Harness still needs real `usage.prompt_tokens` budget assertion and rerun. |
| AC-10 | NOT MET | Correctly gated behind full Tier-1 completion. |

Forgotten items: none. Every original plan task is represented in Active, Completed, or the gated AC-10 path. Deferred items: none explicit; AC-10 is gated, not deferred. The plan evolution remains valid only as a directional characterization; it does not weaken the strict SLO or close the loop.

## Goal Tracker Update

Updated `.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md` mutable section:

- Plan version moved to Round 8 Review.
- Added an R8-review plan-evolution row.
- Kept task6 active as `partial (R8 review)`.
- Changed the AC-5 benchmark evidence blocker from `RESOLVED` to `PARTIAL` because ITL exact recomputation and fail-closed verification are still missing.
- Left the AC-5 attribution blocker as resolved, and left AC-6 through AC-10 active/pending.

## Validation Performed

- `git show --stat --oneline bd09d1ca7`
- `git show --name-only --format=short bd09d1ca7`
- `git diff --check 51dd009b8..bd09d1ca7`
- `git ls-files runs/20260530_dsv32_loop6/client_slo_int8`
- `git status --ignored --short runs/20260530_dsv32_loop6/client_slo_int8`
- `python3 runs/20260530_dsv32_loop6/client_slo_int8/ac5_metrics_tool.py --verify`
- Compared the committed TTFT/TPOT/input/output arrays and source SHA256s against the local source JSONLs in `development/results/`.
- Parsed `/tmp/ac5/boot_radixon.log` to verify total rows, benchmark rows, negative rows, print-time grouping, and queue/forward percentiles.
- Checked `/tmp/ac5/benchmark.log` for 320 completed requests per conc and run boundaries.
- Mutated a temporary copy of `ac5_metrics_arrays.json` to confirm `ac5_metrics_tool.py --verify` prints `FAIL` but exits 0.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-9-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-9-summary.md

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
