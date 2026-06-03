# Round 14 Summary — AC-7 evidence repair (exact-recomputable + AC-7-methodology profiling)

## Mainline objective (round contract)
Repair the AC-7 evidence bundle so it is exact-recomputable and the failing-row profiling
obligation is discharged under AC-7 methodology. Codex's R13 review rejected AC-7 because:
(1) `ac7_resweep_metrics.json` recorded DS conc-64 `achieved=64` while the comparator headline is
46.983 (it stored the `max_concurrency` cap, not the effective `concurrency` field); (2) not
exact-recomputable (rounded summaries, 16-hex SHA prefixes, no fail-closed verifier); (3) the
profiling obligation was cited to the AC-5 WARMUP=0/320/60 run, not the AC-7 64/120/600 methodology.
The 18 raw JSONLs + the AC-7 DS decode-batch log were present, so no full re-sweep was needed.

## What landed
1. **Exact-recomputable metrics + fail-closed verifier (commit 147b6d05f, data-only).**
   `ac7_resweep_metrics.json` rebuilt from the 18 raw JSONLs with per-trial **effective
   `concurrency`** (the comparator's field — DS conc-64 median = **46.983**, fixing the prior `64`
   contradiction), exact per-request arrays (`ttfts_s`, `per_req_gen_tps = output_lens[i]/sum(itls[i])`),
   stored `p99_ttft_ms`, completed/errors/duration, and **full 64-char SHA256** per JSONL.
   `ac7_metrics_tool.py --verify` recomputes the `ac11_resweep.md` rows (achieved/TPS/TTFT, DS+DSA,
   all conc) from the committed JSON and is **fail-closed** — recomputes DS achieved 15.998/31.996/46.983,
   TPS 17.711/11.546/9.796, TTFT 12.838/25.491/100.836 s, all == the report; tamper tests
   (median-moving value, dropped array element) exit 1; clean exits 0 PASS.
2. **Profiling discharged at AC-7 methodology (commit 99e51ad00).**
   - `decode_batch_ac7.txt` — from the AC-7 3-trial sweep DS log: per-req decode TPS = gen/`#running-req`
     = **17.7 / 11.5 / 9.8 tok/s** at decode batch 16/32/~38, reconciling the comparator **TPS FAIL**.
   - `queue_attribution.txt` — a fresh DS int8/0.7/radix-on run at the **same methodology**
     (`num_prompts=64`, 120/600, `--enable-request-time-stats-logging`) that **reproduces AC-7**
     (TTFT 12.8/25.4/100.8 s, achieved 16/32/47); per-conc `queue_duration` p99 = **10.5 / 22.6 / 96.7 s**
     (bucketed by `.meta.json` run windows, 1082 valid rows) vs client TTFT → DS TTFT is
     **admission-queue-dominated** (DS drains the `request_rate=inf` flood-queue slower than DSA;
     conc-64 queue largest, matching the 47/64 achieved-concurrency deficit), reconciling the **TTFT FAIL**.
   - `ac11_analysis.md` updated to cite these AC-7-methodology artifacts; AC-5 WARMUP=0 demoted to background.

## Result
AC-7 evidence repaired and self-consistent. The **admission-restored headline** (DS achieved
16/32/47 = 100/100/73% vs Loop-5 14.5/24.6/35.7) now recomputes from committed data; the
**DS-vs-DSA parity FAIL** (TPS 0.31–0.38×, TTFT 18–49×) is attributed at AC-7 methodology
(decode-batch + request-time queue) as a **DEC-7 directional** follow-up — not a footprint
regression; AC-7 is soft/characterized (DEC-9). The **AC-5 DS strict-SLO miss remains the open
mainline blocker**.

## Files Changed
- `runs/20260530_dsv32_loop6/ac7_resweep/`: `ac7_metrics_tool.py` (new, build/verify), `ac7_resweep_metrics.json` (rebuilt exact), `decode_batch_ac7.txt` + `queue_attribution.txt` (new profiling), `ac11_analysis.md` (cites AC-7-methodology profiling).
- `.humanize/bitlesson.md` (durable-evidence lesson +clause (e): recompute the consumer's exact field, verify against the published artifact), goal-tracker (R14 row; task8/AC-7 done-characterized; AC-7 evidence-bundle blocker → RESOLVED), round-14 contract/summary (gitignored loop state).

## Validation
- `ac7_metrics_tool.py --verify`: recomputed == `ac11_resweep.md` (DS+DSA achieved/TPS/TTFT, all conc) + sanity PASS; tamper tests exit 1.
- The request-time-stats DS run reproduced AC-7 (TTFT 12.8/25.4/100.8 s, achieved 16/32/47) — same regime; per-conc `queue_duration` bucketed by `.meta.json` windows (queue ≤ TTFT, residuals valid).
- `git diff --check` clean; commits 147b6d05f + 99e51ad00 pushed to `jimmy`; node0 GPUs freed; serve script unchanged (used its existing `EXTRA_SERVER_ARGS` for the flag).

## Remaining Items
- **Open mainline blocker:** AC-5 DS strict client SLO (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). The AC-7 data confirms the root cause (admission-queue + DS throughput < DSA); the AC-5 remediation (smallest scheduling/decode/operating-point change) is the next focus.
- **Cross-node wrapper smoke** stays PARTIAL (run only before a future cross-node scripted artifact; AC-7 was local). **DSA conc-64 TPS ~29.5** Queued. **AC-8** (~70K probe), gated **AC-10** — later. No FlashMLA decode-assert changes; DS-fair thresholds unchanged.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260530-durable-tracked-acceptance-evidence
Notes: Added clause (e): when the published number comes from a downstream CONSUMER (here `benchmark_compare.py`), the recomputable source must store the EXACT field/formula the consumer uses and the verifier must recompute the consumer's PUBLISHED value — not an adjacent-looking field. R13 stored `max_concurrency` (cap=64) as "achieved" while the comparator's achieved is the JSONL `concurrency` (effective=46.983), so the "recomputable source" silently couldn't reproduce the headline; R14 fixed it by storing effective `concurrency` + a fail-closed verifier that recomputes the `ac11_resweep.md` rows and asserts equality (tamper → exit 1). Applied existing lessons: BL-20260530-cold-flood-not-steady-state-slo (num_prompts=64 steady-state for the request-time-stats run), BL-20260530-clean-latency-attribution (per-conc queue bucketed by reliable `.meta.json` run windows, not entry/print-time gap clustering which mis-bucketed first), BL-20260530-remote-server-launch (`set +e`/`|| true`, node0 boots), and the push-between-commits preference.
