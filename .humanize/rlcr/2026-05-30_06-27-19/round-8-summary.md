# Round 8 Summary — exact-recomputable AC-5 evidence + reconciled attribution

## Mainline objective (round contract)
Fully close the AC-5 evidence + attribution so task6/AC-5 is verifiable. Codex's R7 review found the two R6 blockers only *partially* resolved: (1) the evidence was summary-only (not exactly recomputable), and (2) the attribution had contradictory row accounting (`valid=959 > 3x320=960` — false arithmetic) plus a wall-clock window split that mis-bucketed the log (306/337/316 ≠ 320/conc), with the stale R6 aggregate still in `client_slo_metrics.txt`. Data-correction round on the existing R6 hardware run (no re-run; local JSONLs + full server log present and used). No production code.

## What landed (commit bd09d1ca7, pushed to `jimmy`)
1. **Exact recomputation source** — `client_slo_int8/ac5_metrics_arrays.json` + `ac5_metrics_tool.py`. Per conc, the **exact per-request arrays**: `ttfts` (s), `tpots` (ms = `sum(itls[i])/(output_lens[i]−1)`, the sglang formula — reproduces stored `median_tpot_ms` exactly), `input_lens`, `output_lens`, plus errors-all-empty, each source JSONL's **SHA256**, and the percentile method (`numpy.percentile`). `python3 ac5_metrics_tool.py --verify` recomputes every TTFT/TPOT/TPS percentile **from the committed JSON alone** and asserts **recomputed == stored → PASS at all conc** (TTFT p50/p99 and ITL p50/p95/p99 match the JSONL bit-for-bit; TPOT p50/p99 exact). Replaces the summary-only addendum as the recomputation source.
2. **Rebuilt attribution** — `attribution_per_conc.txt`. From **benchmark rows only** (`output_len=512`), grouped per conc by **request-completion print-time** (the `[HH:MM:SS]` server-log prefix), split at the 2 largest gaps → **320 / 320 / 320**, reproducing benchmark.log's 320-completed/conc. **Full row reconciliation (exact arithmetic):** 967 parsed = **3 HEALTH_CHECK + 4 warmup (`output_len` 8/32) + 960 benchmark**; **5 invalid negative-`queue_duration` rows dropped (all conc-64) → 955 valid**; per-conc valid **320 / 320 / 315**. Per-conc queue p50/p95/p99 (10.5 / 22.3 / 99.4 s p99), tail-to-tail post-admission residual (2.3 / 3.2 / 11.8 s), measured-vs-inferred (forward_duration = completion-time, context-only). The false `959>960` and the mis-bucketing are gone; entry_time gap-clustering and the `T0+cumulative-durations` split (both tried in R7) are explicitly rejected with the reason (within-run waves rival inter-run gaps; T0 anchored on readiness mis-buckets).
3. **De-staled metrics** — `client_slo_metrics.txt`: the R6 all-conc aggregate (`N=959`) + `#running-req 19-20` line replaced with the corrected per-conc attribution + the `decode_batch_excerpt.txt` per-req-TPS figures (batch 16/~32/~38 → 17.7/11.5/9.7).
4. **Consistency** — `client_slo_report.md` + `ac5_evidence_addendum.txt` now point "recomputable" at the exact source and drop the `959`/warmup framing; the report's attribution section carries the corrected reconciliation.

## Result (verdict unchanged; now exactly recomputable + internally consistent)
DIRECTIONAL — accepted progress, explicitly NOT shippable (DEC-3). conc-16 meets strict `<22 s` (12.8 s); conc-32/64 TTFT and all-conc per-req TPS miss the strict SLO and remain the **open mainline blocker**. Every AC-5 number now recomputes from committed files; the attribution is reconciled to the exact row counts.

## Files Changed
- `runs/20260530_dsv32_loop6/client_slo_int8/`: `ac5_metrics_arrays.json` (new, exact arrays), `ac5_metrics_tool.py` (new, build/verify), `attribution_per_conc.txt` (rebuilt), `client_slo_metrics.txt` (de-staled), `ac5_evidence_addendum.txt` (pointer to exact source).
- `runs/20260530_dsv32_loop6/client_slo_report.md` (evidence bundle + corrected attribution reconciliation).
- `.humanize/bitlesson.md` (updated `clean-latency-attribution`), goal-tracker, round-8 contract/summary (gitignored loop state).

## Validation
- `ac5_metrics_tool.py --verify`: recomputed == stored, **PASS** at conc 16/32/64 (TTFT/TPOT/TPS from committed JSON alone).
- Attribution reconciliation reproduces benchmark.log (3×320 completed) and Codex's authoritative per-conc valid counts (320/320/315); print-time grouping verified against the log.
- All new files tracked (`git check-ignore` → none); `git diff --check` clean; no stale `959`/`>960`/`cumulative-durations`/`wall-clock-windows` strings remain in the tracked AC-5 files.
- No re-run, no production code change; the R6 run's `.meta.json` radix-on sidecars unchanged.

## Remaining Items
- **Open mainline blocker:** strict client SLO still fails (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). Characterized (throughput/decode-batch, not footprint).
- **Next round = hardware (AC-6):** DSA-default product proof (DSA-default boot meets SLO unchanged, allocates no DS table; DS opt-in toggles the compact int8 path), pairing the **AC-9** code edit (`within_budget` from real `usage.prompt_tokens`, fail-closed, DS-fair thresholds UNCHANGED) + its live rerun.
- Then **AC-7** (3-trial DS+DSA lifted-point re-sweep), **AC-8** (~70K servability probe), gated **AC-10**. No FlashMLA decode-assert changes (AC-3.3).

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260530-clean-latency-attribution
Notes: Sharpened BL-20260530-clean-latency-attribution with the R7→R8 fix. R7 disclosed the negative-row filtering but still (a) asserted a false `959>960`, (b) mis-grouped per conc using `T0+cumulative-durations` (gave 306/337/316 ≠ 320/conc), and (c) left the evidence summary-only. R8's additions to the lesson: CLASSIFY rows by a stable shape signature before counting (benchmark rows = `output_len=512`; warmup = `output_len` 8/32; health = HEALTH_CHECK) and reconcile valid-vs-nominal with EXACT arithmetic (not a "warmup makes valid>nominal" narrative — the truth was 960 benchmark, 5 invalid, 955 valid); group per conc by a RELIABLE key (request-completion print-time split at the largest gaps), explicitly NOT entry_time gaps (within-run waves rival inter-run gaps) and NOT a readiness-anchored T0; validate the grouping reproduces the benchmark's known per-conc completion count; and commit the EXACT per-request arrays (or checksum + recompute script) so a reviewer recomputes from committed files, not summary-vs-summary. No new lesson added — this is the same failure family, made precise across rounds. Applied existing BL-20260530-durable-tracked-acceptance-evidence (exact arrays as tracked .json + a committed --verify script; git check-ignore + git diff --check before claiming done) and the push-between-rounds preference (pushed bd09d1ca7 to jimmy).
