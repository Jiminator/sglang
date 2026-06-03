# Round 15 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: ACs: 8/10 addressed, 7/10 met | Forgotten items: 0 | Unjustified deferrals: 0

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-15-contract.md`, `round-15-summary.md`, `goal-tracker.md`, Round 12-14 summaries/reviews, commit `40ccc4b63`, the AC-7 artifacts, `development/benchmark_compare.py`, and the AC-5 report/tracker state for the remaining strict-SLO blocker.

## Implementation Review

Round 15 closes the four R14 AC-7 evidence residuals. I found no remaining high-signal AC-7 evidence gap in the R15 changes.

`runs/20260530_dsv32_loop6/ac7_resweep/ac7_metrics_tool.py` now uses `TOL = 0.0005`, matching the 3-decimal published report precision, and adds the requested provenance/shape checks: 64-hex SHA, numeric required scalars, 3 trials per side/concurrency, `completed > 0`, `duration_s >= 600`, and `len(per_req_gen_tps) == len(ttfts_s) == completed`. Clean verification exits 0 and recomputes the published `ac11_resweep.md` achieved/TPS/TTFT rows.

I reproduced the three important tamper checks in a temp copy:

- Rendered-value drift: changed DS c64 median source so achieved recomputed as `46.987` vs published `46.983`; verifier exited 1.
- Provenance drift: changed a trial SHA to `deadbeef`; verifier exited 1.
- Shape drift: dropped a `per_req_gen_tps` element; verifier exited 1.

The committed AC-7 metric SHA256s also match the local raw `/tmp/ac7/results/*.jsonl` files available in this workspace. `queue_attribution.txt` now clearly labels the request-time-stats data as a separate AC-7-methodology reproduction run, includes source JSONL/log SHA256s and run windows, and reconciles `320/384/378` valid rows as measured completed `256/320/320` plus warmup-epoch rows `64/64/58`. `ac11_analysis.md` no longer has the stale "attributed via AC-5" verdict; the verdict cites `decode_batch_ac7.txt` and `queue_attribution.txt`.

## Mainline Gaps

1. **The original Loop-6 plan is still incomplete, even though R15's AC-7 objective is done.**

   AC-5 remains directional-only and fails the strict DS client SLO. The verified report records conc 32/64 P99 TTFT misses (`25.5 / 111.2 s > 22.0`) and per-request TPS misses at every concurrency (`17.6 / 11.5 / 9.3 < 30`). AC-8 has not run the lifted ~70K-token servability probe. AC-10 is still correctly gated behind full Tier-1 completion. This means the loop still cannot output `COMPLETE`.

## Blocking Side Issues

1. **Strict DS client SLO still blocks the Ultimate Goal.**

   AC-5 evidence is durable and verified, but the strict done-criterion is not met: DS does not yet serve every conc 16/32/64 at `P99 TTFT < 22.0 s` and `>=30 TPS/req`. The next remediation must target the measured scheduling/decode/admission operating point, not more AC-7 evidence churn.

## Queued Side Issues

1. **Cross-node wrapper smoke remains future-gated.**

   R13/R15 AC-7 artifacts are local node0 artifacts, so this does not invalidate them. Before publishing any future cross-node scripted benchmark artifact, run the wrapper with `HOST=<remote>` and capture both the `bench_serving` readiness banner for that host and the matching sidecar.

2. **DSA-default conc-64 TPS remains below the client threshold.**

   This remains queued under the R12 user decision: it reproduces the pre-existing DSA/H200 baseline and is not introduced by DS.

## Goal Tracker Audit

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified earlier. |
| AC-2 | MET | Feasibility budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, sidecar consumers, launcher, real-mask NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted 0.7 operating point, HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | Evidence/attribution verified; strict DS SLO still fails. |
| AC-6 | MET | Verified in R12 under the user-approved non-regression/opt-in semantics. |
| AC-7 | MET / CHARACTERIZED | R15 verifier/provenance/reconciliation cleanup verified in this review; DEC-9 soft characterization accepted. |
| AC-8 | NOT MET | Lifted ~70K-token servability probe pending. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. |
| AC-10 | NOT MET | Correctly gated behind full Tier-1 completion. |

Forgotten items: none. Every original task is represented in Active, Completed, or the gated AC-10 path. Deferred items: none. Plan evolution is valid: R15 is a data-only evidence cleanup for AC-7 and does not downgrade AC-5, AC-8, or AC-10.

## Required Implementation Plan

1. Complete AC-8 next: run the lifted ~70K-token `/generate` probe at DS int8/mem0.7/radix-on, record served `max_total_num_tokens`, HTTP result, and no OOM/instability; if it still fails, publish the characterized ceiling with the server-side reason.
2. Return to AC-5 strict SLO remediation: use the verified AC-5 and AC-7 attribution to implement the smallest scheduling/decode/operating-point change that can restore both `P99 TTFT < 22.0 s` and `>=30 TPS/req` at conc 16/32/64, then rerun the full client workload and publish exact arrays plus a fail-closed verifier.
3. Start AC-10 only after AC-3 through AC-9 are verified and the strict Loop-6 done-criterion is no longer open. Then implement the adjustable-`top_k` kernel or selector path and record NIAH 4K/16K/64K recall deltas plus TPS/TTFT cost.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 15 Review.
- Added an R15-review Plan Evolution row.
- Moved task8/AC-7 from Active to Completed and Verified.
- Marked the AC-7 evidence-bundle issue as RESOLVED and VERIFIED.
- Left AC-5, AC-8, and gated AC-10 active; left cross-node wrapper smoke future-gated and DSA conc-64 TPS queued.

## Validation Performed

- `git log --oneline -30`
- `git show --stat --oneline 40ccc4b63`
- Inspected `round-15-contract.md`, `round-15-summary.md`, `goal-tracker.md`, and Round 12-14 summaries/reviews.
- Inspected `ac7_metrics_tool.py`, `ac7_resweep_metrics.json`, `ac11_resweep.md`, `ac11_analysis.md`, `decode_batch_ac7.txt`, and `queue_attribution.txt`.
- Ran `python3 runs/20260530_dsv32_loop6/ac7_resweep/ac7_metrics_tool.py --verify`.
- Temporary-copy tamper tests: rendered-value drift, short SHA, and dropped `per_req_gen_tps` element all exited 1.
- Compared committed AC-7 source SHA256s against local `/tmp/ac7/results/*.jsonl`; mismatches = 0.
- Inspected AC-5 report lines documenting the remaining strict-SLO miss.
- `git diff --check`

NOT COMPLETE
