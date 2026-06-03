# Round 15 Summary — AC-7 evidence made review-clean (verifier precision + provenance + reconciliation)

## Mainline objective (round contract)
Close the four data-only AC-7 evidence residuals from Codex's R14 review (no hardware): the
verifier wasn't fail-closed at the published 3-decimal precision, it didn't validate its own
provenance/shape, the profiling artifact wasn't reconciled to completed counts, and `ac11_analysis.md`
had a stale "attributed via AC-5" verdict sentence. AC-5's directional verdict + the open DS
strict-SLO blocker stay tracked, not this round's objective.

## What landed (commit 40ccc4b63, data-only)
1. **Verifier fail-closed at report precision + provenance/shape.** `ac7_metrics_tool.py`: `TOL`
   0.05 → **0.0005** (the comparator renders 3 decimals). Added sanity checks: every trial has a
   **64-hex `sha256`**, required scalars numeric, **3 trials/side/conc**, `completed > 0`,
   `duration_s ≥ 600` (window floor), and `len(per_req_gen_tps) == len(ttfts_s) == completed`.
   **Tamper tests now exit 1:** a median-moving concurrency (recomputes 46.987 ≠ published 46.983),
   a short SHA (`deadbeef`), and a dropped `per_req_gen_tps` element; **clean exits 0 PASS**
   (recomputes the `ac11_resweep.md` achieved/TPS/TTFT rows, DS+DSA, all conc).
2. **Profiling provenance + reconciliation.** `queue_attribution.txt` now states explicitly it is a
   **separate request-time-stats reproduction run** (the 3-trial sweep didn't enable the flag),
   identical methodology, and **reproduces** the AC-7 result (TTFT 12.8/25.4/100.8 s, achieved
   16/32/47). Added source JSONL SHA256 + DS-log SHA + per-conc window starts, and the row-count
   accounting: valid benchmark rows **320/384/378 = measured completed 256/320/320 + 120s-warmup-epoch
   64/64/58** (ReqTimeStats logs every request incl. warmup).
3. **`ac11_analysis.md` verdict** "attributed via AC-5" → cites `decode_batch_ac7.txt` +
   `queue_attribution.txt` (AC-7 methodology). The two remaining AC-5 mentions are corroboration /
   explicitly "background".

## Result
AC-7 evidence bundle is review-clean: exact-recomputable + fail-closed at published precision +
provenance-validated, profiling reconciled with explicit reproduction provenance, attribution cites
AC-7-methodology artifacts. The AC-7 result stands (characterized, DEC-9): **admission restored** (DS
achieved 16/32/47 = 100/100/73% vs Loop-5 14.5/24.6/35.7); DS-vs-DSA parity FAIL is a DEC-7
directional follow-up, not a footprint regression. The **AC-5 DS strict-SLO miss remains the open
mainline blocker**.

## Files Changed
- `runs/20260530_dsv32_loop6/ac7_resweep/`: `ac7_metrics_tool.py` (TOL + provenance/shape checks), `queue_attribution.txt` (provenance + reconciliation), `ac11_analysis.md` (verdict cites AC-7 artifacts).
- `.humanize/bitlesson.md` (durable-evidence lesson: tolerance ≤ published precision + validate SHA/shape provenance), goal-tracker (R15 row; task8/AC-7 done-characterized; evidence-bundle blocker → RESOLVED), round-15 contract/summary (gitignored loop state).

## Validation
- `ac7_metrics_tool.py --verify`: clean exit 0 PASS (recomputes published rows + sanity). Three tamper tests (rendered-value, bad SHA, length) each exit 1.
- `queue_attribution.txt` reconciliation: 320/384/378 = measured + warmup, source SHAs committed, reproduction labeled.
- `ac11_analysis.md`: no "attributed via AC-5" verdict; AC-5 only as corroboration/background.
- `git diff --check` clean; commit 40ccc4b63 pushed to `jimmy`; no hardware (data-only; no servers booted this round).

## Remaining Items
- **Open mainline blocker:** AC-5 DS strict client SLO (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). The AC-7 data confirms the root cause (admission-queue + DS throughput < DSA); the AC-5 remediation (smallest scheduling/decode/operating-point change) is the next focus after AC-8.
- **Cross-node wrapper smoke** PARTIAL (future cross-node only). **DSA conc-64 TPS ~29.5** Queued. **AC-8** (~70K probe), gated **AC-10** — later. No FlashMLA decode-assert changes; DS-fair thresholds unchanged.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260530-durable-tracked-acceptance-evidence
Notes: Extended clause (d): a fail-closed verifier's comparison TOLERANCE must be ≤ the PUBLISHED precision (R14's TOL=0.05 against a 3-decimal report let a value rendering 46.973 vs published 46.983 pass; R15 → ≤0.0005), and the verifier must VALIDATE the provenance/shape fields it relies on (full 64-hex SHA256 — a `deadbeef` SHA passed until checked; required scalars numeric; array length == completed), not just recompute the metric. Applied existing lessons: BL-20260530-clean-latency-attribution (per-conc queue bucketed by reliable `.meta.json` run windows + full row reconciliation incl. the warmup-epoch accounting), and the push-between-commits preference. No new lesson — same durable-acceptance-evidence family, sharpened across R13/R14/R15.
