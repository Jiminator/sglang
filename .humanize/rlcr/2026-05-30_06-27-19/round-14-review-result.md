# Round 14 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: ACs: 8/10 addressed, 6/10 met | Forgotten items: 0 | Unjustified deferrals: 0

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-14-contract.md`, `round-14-summary.md`, `goal-tracker.md`, Round 11-13 summaries/reviews, commits `147b6d05f` and `99e51ad00`, the AC-7 artifacts, `development/benchmark_compare.py`, `development/CLIENT_SLOS.md`, and `development/loop6/runbook.md`.

## Implementation Review

Round 14 fixed the biggest AC-7 evidence contradiction. Clean `ac7_metrics_tool.py --verify` now recomputes the published AC-11 rows from committed data: DS achieved 15.998/31.996/46.983, TPS 17.711/11.546/9.796, and TTFT 12.838/25.491/100.836 s. `ac7_resweep_metrics.json` now stores effective `concurrency` rather than the `max_concurrency` cap, exact per-request TPS arrays, `ttfts_s`, completed/errors/duration fields, and full 64-character SHA256 strings. The AC-7 sidecars still support the intended local node0 operating points: DS int8/radix-on/mem0.7 and DSA-default/radix-on/mem0.85.

I cannot verify AC-7 as complete yet. The R14 contract required an AC-5/AC-6-grade fail-closed verifier and profiling reconciled to completed counts. The current verifier still accepts a recomputed value that visibly disagrees with the published 3-decimal report, and the profiling artifact is not visibly reconciled to the AC-7 completed-count/source evidence.

## Mainline Gaps

1. **AC-7 verifier is not fail-closed at the precision of the published report.**

   `benchmark_compare.py` renders the AC-11 table with 3 decimals, but `ac7_metrics_tool.py` uses `TOL = 0.05` as the comparison threshold (`runs/20260530_dsv32_loop6/ac7_resweep/ac7_metrics_tool.py:26`, checks at `:132-137`). That is not rounding slack for a 3-decimal report; it permits values that would print differently.

   Reproduction: in a temporary copy, I set `.conc["64"].DS[1].concurrency = 0`. The verifier printed `DS achieved=46.973 (md 46.983)` for conc 64 and still exited 0 with `PASS`. That contradicts the R14 contract's "exit 1 on mismatch" requirement (`round-14-contract.md:23-25`, `:37-39`) and the summary claim that a median-moving tamper exits 1.

   Required fix: compare the exact published rounded strings, or use a tolerance no looser than the printed precision, e.g. `<= 0.0005` for 3 decimals. Add tamper tests that actually change the rendered 3-decimal value and must exit 1.

2. **AC-7 verifier does not validate the provenance fields it now relies on.**

   The builder writes `sha256` for each raw JSONL (`ac7_metrics_tool.py:85`), but `verify()` only checks trial count, error count, output length, and `ttfts_s` length (`:115-122`). In a temporary copy, shortening a stored SHA to `deadbeef` still exited 0. Since R13 explicitly called out 16-hex digest prefixes as an evidence defect and R14 claims full SHA256 provenance, this should be sanity-checked.

   Required fix: have `--verify` assert every trial has a 64-hex `sha256`, required scalar fields are present and numeric, `completed > 0`, `duration_s` is above the AC-11 floor, and `len(per_req_gen_tps) == completed` if that array is the TPS source.

3. **The AC-7 profiling artifact is directionally useful, but not reconciled to completed counts as required.**

   The R14 contract required the request-time stats to be "reconciled to the AC-7 completed counts" (`round-14-contract.md:26-30`, `:37-40`). `queue_attribution.txt` gives row counts of 320/384/378 (`runs/20260530_dsv32_loop6/ac7_resweep/queue_attribution.txt:6-19`), while the committed AC-7 DS sweep medians are based on completed counts 256/320/320 for conc 16/32/64 in `ac7_resweep_metrics.json`. The text may come from a fresh profiling reproduction, but it does not show the source JSONL/sidecar for that profiling run or explain why those row counts reconcile to the AC-7 sweep rows.

   Required fix: commit either the profiling run's compact exact source/verifier or a small provenance table that names the profiling JSONL/log, its SHA, completed counts, time windows, and how the 320/384/378 valid rows map to the AC-7-methodology run. If this is a reproduction rather than the original AC-7 sweep, say that explicitly and stop claiming same completed-count reconciliation.

4. **`ac11_analysis.md` still contains a stale AC-5 attribution sentence.**

   The profiling section correctly says AC-5 WARMUP=0 is background (`ac11_analysis.md:43-47`), but the verdict later says the parity failure is "attributed via AC-5" (`runs/20260530_dsv32_loop6/ac7_resweep/ac11_analysis.md:62-68`). That is the exact wording pattern R13 rejected.

   Required fix: change that sentence to cite the AC-7-methodology artifacts (`decode_batch_ac7.txt` and `queue_attribution.txt`).

5. **The original Loop-6 work remains incomplete.**

   AC-5 remains directional-only and fails the strict DS client SLO; AC-8 is still pending; AC-10 remains gated behind full Tier-1 completion. This round cannot output `COMPLETE`.

## Blocking Side Issues

1. **Strict DS client SLO still blocks the Ultimate Goal.**

   AC-5 remains a directional result, not a shippable pass: conc-32/64 TTFT remain above `<22 s`, and DS per-request TPS is below `30 TPS/req` at every concurrency. This is the mainline blocker after the AC-7 evidence repair is made review-clean.

2. **Cross-node wrapper smoke remains partial for future cross-node artifacts.**

   This does not block the local node0 AC-7 artifact, per the R14 contract, but it must remain partial. Before publishing any future cross-node scripted benchmark artifact, capture a wrapper smoke proving `bench_serving --host <remote>` and the sidecar target the same host.

## Queued Side Issues

1. **DSA-default conc-64 TPS remains below the client threshold.**

   This stays queued under the R12 user decision because it reproduces the pre-existing DSA/H200 baseline and is not introduced by DS.

## Goal Tracker Audit

| AC | Status | Evidence / blocker | Deferred justification |
|----|--------|--------------------|------------------------|
| AC-1 | MET | Strategic decision doc verified earlier: `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`. | - |
| AC-2 | MET | Feasibility budget and binding int8 lever verified earlier. | - |
| AC-3 | MET | Compact int8 table, scale sidecar consumers, launcher, real-mask NIAH, and microbench verified earlier. | - |
| AC-4 | MET | Lifted 0.7 operating point, HBM budget, and no-OOM proof verified earlier. | - |
| AC-5 | PARTIAL | Evidence and attribution are verified, but strict DS SLO still fails. | - |
| AC-6 | MET | Verified in R12 under the user-approved non-regression/opt-in semantics. | - |
| AC-7 | PARTIAL | R14 fixed the main data contradiction, but the verifier is not fail-closed at published precision and profiling is not visibly reconciled to completed counts. | - |
| AC-8 | NOT MET | Lifted ~70K-token servability probe pending. | - |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. | - |
| AC-10 | NOT MET | Correctly gated behind full Tier-1 completion. | - |

Forgotten items: none. Every original task is represented in Active, Completed, or the gated AC-10 path. No tasks are marked complete in the tracker without review verification; task8 remains Active after this review.

Deferred items audit: no explicitly deferred tasks. The cross-node wrapper smoke is partial/future-gated, not an explicit deferral of an AC.

Goal Completion Summary:

```text
Acceptance Criteria: 6/10 met (0 deferred)
Active Tasks: 4 remaining (AC-5, AC-7 residual repair, AC-8, gated AC-10)
Estimated remaining rounds: 4+
Critical blockers: AC-7 verifier/profiling evidence residuals; AC-5 strict DS SLO miss; AC-8 not run; AC-10 gated
```

## Mainline Drift Audit

The round objective was clear and singular: repair AC-7 evidence. R14 advanced that objective materially by fixing the `max_concurrency` vs effective-`concurrency` contradiction and adding AC-7-methodology attribution artifacts. The remaining problems are evidence-quality residuals, not a pivot away from the plan.

Mainline Progress Verdict: ADVANCED

Blocking Side Issues: 2

Queued Side Issues: 1

True blocking side issues: AC-5 strict DS SLO miss for the Ultimate Goal; cross-node wrapper smoke before any future cross-node artifact. Queued side issue: pre-existing DSA-default conc-64 TPS tension. AC-8 and AC-10 are not side issues; they are remaining plan tasks.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 14 Review.
- Added an R14-review Plan Evolution row.
- Kept task8/AC-7 Active as partial rather than verified.
- Changed the AC-7 evidence-bundle issue from resolved to partial with the verifier/profiling residuals.
- Left AC-5, AC-8, and gated AC-10 active; left cross-node wrapper smoke partial and DSA conc-64 TPS queued.

## Stagnation Check

Development is not stalled. The last few rounds show repeated evidence-quality corrections, but they are not circular: R12 fixed host targeting and AC-6 evidence, R13 produced the AC-7 sweep, and R14 repaired the main AC-7 metrics contradiction and added profiling. The recurring weakness is verifier rigor, so the next step should be a narrow AC-7 evidence cleanup, not a circuit breaker.

## Required Implementation Plan

1. Tighten `ac7_metrics_tool.py --verify` so it fails on any recomputed value whose 3-decimal rendering differs from `ac11_resweep.md`. Add a tamper test that changes DS c64 achieved from `46.983` to a different rendered value and exits 1.
2. Add verifier sanity checks for provenance and metric-source shape: 64-hex SHA, required scalar fields, three trials per side/conc, `per_req_gen_tps` length equals `completed`, and AC-11 duration/workload floors.
3. Reconcile `queue_attribution.txt` to source evidence: commit the profiling run source summary or a provenance table with SHA, completed counts, windows, and row-count accounting. Explain the 320/384/378 rows versus the original AC-7 DS sweep completed counts.
4. Fix the stale `ac11_analysis.md` verdict sentence so AC-7 attribution cites AC-7-methodology artifacts, not AC-5.
5. After AC-7 is review-clean, complete AC-8, then return to the AC-5 strict SLO blocker. Start AC-10 only after AC-3 through AC-9 are verified.

## Validation Performed

- `git log --oneline -30`
- `git show --stat --oneline 147b6d05f 99e51ad00`
- Inspected `round-14-contract.md`, `round-14-summary.md`, `goal-tracker.md`, and Round 11-13 summaries/reviews.
- Inspected `ac7_metrics_tool.py`, `ac7_resweep_metrics.json`, `ac11_resweep.md`, `ac11_analysis.md`, `decode_batch_ac7.txt`, `queue_attribution.txt`, and AC-7 sidecars.
- Ran `python3 runs/20260530_dsv32_loop6/ac7_resweep/ac7_metrics_tool.py --verify`.
- Temporary-copy tamper tests: changed DS c64 trial concurrency so recomputed output printed `46.973 (md 46.983)` and observed exit 0; shortened a SHA to `deadbeef` and observed exit 0; dropped a `ttfts_s` element and observed exit 1.
- Inspected `development/benchmark_compare.py` to confirm the AC-11 report renders 3-decimal values and emits the profiling obligation.
- Inspected `development/CLIENT_SLOS.md` and `development/loop6/runbook.md` for SLO and Loop-6 scope.
- `git diff --check`

NOT COMPLETE
