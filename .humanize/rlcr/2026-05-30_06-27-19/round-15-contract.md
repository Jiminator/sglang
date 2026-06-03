# Round 15 Contract

## Mainline Objective (exactly one)
**Make the AC-7 evidence bundle review-clean** — close the four data-only residuals from Codex's
R14 review so the verifier is fail-closed at the published precision, validates its own
provenance, the profiling artifact is reconciled with explicit provenance, and the analysis cites
only AC-7-methodology attribution. No hardware (the 3-trial sweep + the request-time-stats
reproduction already exist; this is evidence cleanup). AC-5's directional verdict + the open DS
strict-SLO blocker stay tracked, not this round's objective.

## Target ACs (1)
- **AC-7** (`coding`, data-only) — review-clean exact-recomputable + reconciled evidence.

## Blocking Side Issues in Scope (Codex R14 review)
1. **Verifier precision.** `ac7_metrics_tool.py --verify` uses `TOL=0.05`; the comparator report
   renders 3 decimals. Tighten to ≤ 0.0005 (or compare rounded strings) so any value that renders
   differently from `ac11_resweep.md` exits 1. Add a tamper test that changes a *rendered* 3-decimal
   value (e.g. DS c64 achieved 46.983 → 46.5) and must exit 1.
2. **Verifier provenance/shape.** `--verify` must assert: every trial has a 64-hex `sha256`; required
   scalar fields present + numeric; 3 trials per side/conc; `completed > 0`; `duration_s` ≥ the
   AC-11 window floor (600 s); `len(per_req_gen_tps) == completed` (the TPS source) and
   `len(ttfts_s) == completed`. Demonstrate each new check fails closed.
3. **Profiling provenance + reconciliation.** `queue_attribution.txt` rows (320/384/378) come from a
   **separate request-time-stats reproduction run** (not the original AC-7 3-trial sweep, whose
   measured completed are 256/320/320). Commit a provenance table: profiling JSONL/log paths + SHA256,
   per-conc completed counts, run-window boundaries, and the row-count accounting (ReqTimeStats rows =
   warmup-epoch + measured requests, so > measured completed). State explicitly it is a reproduction
   that reproduces the AC-7 TTFT/achieved, not the original sweep.
4. **Stale AC-5 sentence.** Fix `ac11_analysis.md`'s verdict line "attributed via AC-5" → cite
   `decode_batch_ac7.txt` + `queue_attribution.txt` (AC-7 methodology).

## Queued / Out of Scope (NOT downgraded)
- **AC-5 DS strict-SLO miss** stays the open mainline blocker (remediation later).
- **Cross-node wrapper smoke** PARTIAL (future cross-node only). **DSA conc-64 TPS ~29.5** Queued.
- **AC-8** (~70K probe), gated **AC-10** — later rounds. No FlashMLA decode-assert changes; DS-fair thresholds unchanged.

## Round Success Criteria
1. `ac7_metrics_tool.py --verify`: TOL ≤ 0.0005; provenance (64-hex SHA) + shape (per_req_gen_tps/ttfts len == completed, 3 trials, completed>0, duration≥600) checks added; clean exits 0 PASS; demonstrated tamper tests (rendered-value change, bad SHA, length mismatch) each exit 1.
2. Tracked profiling provenance (JSONL/log paths + SHA256 + per-conc completed + windows + row-count accounting) reconciling `queue_attribution.txt`; explicitly labeled a reproduction.
3. `ac11_analysis.md` verdict cites AC-7-methodology artifacts (no "attributed via AC-5").
4. `git diff --check` clean; commit + push; goal-tracker updated (task8/AC-7); `round-15-summary.md` with BitLesson Delta.

## Out-of-Scope Guards
- No re-run/hardware; data-only. The comparator FAIL verdict + the characterized (DEC-9) AC-7 stand. No fabrication.
