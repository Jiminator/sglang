# Round 14 Contract

## Mainline Objective (exactly one)
**Repair the AC-7 evidence bundle so it is exact-recomputable and the failing-row profiling
obligation is discharged under AC-7 methodology.** Codex's R13 review verified the AC-7 sweep
exists (18 sidecars, correct operating points) but rejected verification because: (1) the
"recomputable" `ac7_resweep_metrics.json` records `achieved=64` (the `max_concurrency` cap) while
the comparator headline is `46.983` (the JSONL `concurrency` effective field) — the source can't
recompute the headline; (2) not exact-recomputable (rounded summaries, 16-hex SHA prefixes, no
fail-closed verifier); (3) the profiling obligation was "discharged" by the AC-5 WARMUP=0/320/60
run, which is a different methodology than the AC-7 64/120/600 sweep. The raw 18 JSONLs + the
AC-7 DS decode-batch log are present (no full re-sweep needed). AC-5's directional verdict + the
open DS strict-SLO blocker stay tracked, not this round's objective.

## Target ACs (1)
- **AC-7** (`coding`, hardware-derived) — exact-recomputable evidence + AC-7-methodology profiling.

## Blocking Side Issues in Scope (Codex R13 review)
1. **Rebuild `ac7_resweep_metrics.json` as an exact source of truth** from the 18 raw JSONLs:
   per-trial **effective `concurrency`** (the comparator field, not `max_concurrency`), exact gate
   metrics (median_tpot→TPS, p99_ttft, achieved, completed, errors, duration, workload shape),
   **full 64-char SHA256** per JSONL, and computed medians that reproduce `ac11_resweep.md`.
2. **Fail-closed verifier** (`ac7_metrics_tool.py --verify`): recompute the comparator's per-conc
   achieved/TPS/TTFT medians from the committed JSON and assert they match `ac11_resweep.md` (exit 1
   on mismatch); AC-5/AC-6-grade tamper-resistance.
3. **Discharge the profiling obligation under AC-7 methodology.** Decode-batch evidence from the
   existing AC-7 DS log (per conc) + DS request-time stats captured at the AC-7 methodology (a short
   DS int8/0.7/radix-on run, 64-prompt, 120/600, with `--enable-request-time-stats-logging`),
   reconciled to the AC-7 completed counts; update `ac11_analysis.md` to cite this artifact (AC-5
   WARMUP=0 attribution demoted to background context).

## Queued / Out of Scope (NOT downgraded)
- **AC-5 DS strict-SLO miss** stays the open mainline blocker (remediation is a later round).
- **Cross-node wrapper smoke** stays PARTIAL (run only if a future cross-node scripted artifact is published; AC-7 is local).
- **DSA conc-64 TPS ~29.5** stays Queued (pre-existing). **AC-8** (~70K probe), gated **AC-10** — later. No FlashMLA decode-assert changes; DS-fair thresholds unchanged.

## Round Success Criteria
1. `ac7_resweep_metrics.json` rebuilt: effective `concurrency` per trial (DS c64 median 46.98 = comparator's 46.983), exact gate fields, full SHA256, medians; `ac7_metrics_tool.py --verify` recomputes the `ac11_resweep.md` rows from committed data and is fail-closed (demonstrated on a tamper).
2. AC-7-methodology profiling artifact: per-conc decode-batch (#running-req + gen throughput → per-req TPS) from the AC-7 DS log + DS request-time stats (queue vs forward) from a 64/120/600 DS run, reconciled to completed counts; `ac11_analysis.md` updated to cite it.
3. `git diff --check` clean; commit + push after each commit; goal-tracker updated (task8/AC-7); `round-14-summary.md` with BitLesson Delta; servers killed + GPUs freed.

## Out-of-Scope Guards
- No fabrication; the comparator FAIL verdict stands (AC-7 characterized/soft per DEC-9). Use `set +e` / `|| true` on pkills; absolute remote paths; no foreground `sleep` in harness Bash (per BL-remote-server-launch). Push between commits.
