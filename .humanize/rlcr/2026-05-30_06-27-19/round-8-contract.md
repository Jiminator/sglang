# Round 8 Contract

## Mainline Objective (exactly one)
**Fully close the AC-5 evidence + attribution so task6/AC-5 is verifiable.** Codex's R7 review found the two R6 blockers only *partially* resolved: (1) the evidence addendum is summary-only (not an exact percentile-recomputation source), and (2) the attribution has contradictory row accounting (`valid=959 > 3x320=960` — false arithmetic) plus a wall-clock window split that mis-buckets the log (306/337/316 ≠ 320/conc), and the stale R6 aggregate attribution still sits in `client_slo_metrics.txt`. This is a **data-correction round on the existing R6 hardware run** (no re-run — the local JSONLs + full server log are present; verified this round). No production code.

## Target ACs (1–2)
- **AC-5** (`coding`, hardware-derived data) — make the benchmark numbers exactly recomputable from tracked files and the attribution clean + internally consistent.

## Blocking Side Issues in Scope (Codex R7 review)
1. **Evidence is summary-only.** `ac5_evidence_addendum.txt` records rounded percentile summaries; a future checkout cannot recompute p99 TTFT/TPOT/ITL. Fix: add a tracked **exact numeric artifact** (`ac5_metrics_arrays.json`) with per conc the exact arrays `ttfts` (s), per-request `tpots` (ms, = `mean(itls[i])`), `input_lens`, `output_lens`, the `errors` proof (all-empty), the **SHA256 of each source JSONL**, the **extraction command**, the **percentile method**, and a **recomputed-vs-JSONL-stored validation**. Full per-token ITL recomputable from the checksummed source via the committed command (ITL summary stored too).
2. **Attribution row accounting is contradictory + mis-bucketed.** Rebuild `attribution_per_conc.txt` from **benchmark-shaped rows only** (`output_len=512`), grouped per conc by **request-completion print-time** (verified this round: 2 largest gaps → 320/320/320; negatives 0/0/5 → valid 320/320/315). Reconcile counts explicitly: 967 parsed = 3 HEALTH_CHECK + 4 warmup (`output_len` 8/32) + 960 benchmark; 5 invalid negative-queue (all conc-64); 955 valid (320/320/315). Per-conc queue p50/p95/p99 over valid rows; tail-to-tail residual; measured-vs-inferred. No `959>960`.
3. **Stale tracked metrics.** `client_slo_metrics.txt` still has the R6 all-conc aggregate attribution (`N=959`) and the obsolete `#running-req 19-20` TPS line. Update it to the corrected per-conc attribution (or mark superseded + point to the addenda).
4. **Consistency cleanup.** Update `ac5_evidence_addendum.txt` + `client_slo_report.md` so "recomputable" points at the exact source and drop the `959`/warmup-miscount framing; fix the BitLesson `clean-latency-attribution` text that referenced the 959 count.

## Queued / Out of Scope (explicitly NOT downgraded)
- **Strict-SLO failure stays a visible mainline blocker** (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). Not fixable by data work; remains tracked.
- **AC-6** (DSA-default product proof), **AC-7** (3-trial re-sweep), **AC-8** (~70K probe), **AC-9** (real-token within-budget edit + live rerun), gated **AC-10** — the **next round is the hardware round (AC-6, pairing AC-9's code edit + rerun)**. No FlashMLA decode-assert changes (AC-3.3); do not change DS-fair thresholds (AC-9).

## Round Success Criteria
1. Tracked `ac5_metrics_arrays.json`: per conc exact `ttfts`/`tpots`/`input_lens`/`output_lens` arrays + errors-all-empty proof + source SHA256 + extraction command + percentile method + recomputed==stored validation. Every headline AC-5 number recomputes from committed files.
2. Rebuilt `attribution_per_conc.txt`: benchmark-row filtering + print-time grouping (320/320/320 → valid 320/320/315), fully reconciled counts, per-conc queue p50/p95/p99, tail-to-tail residual, honest measured-vs-inferred; **no contradictory arithmetic**.
3. `client_slo_metrics.txt` corrected (per-conc attribution; obsolete aggregate + `#running-req 19-20` removed/superseded).
4. `ac5_evidence_addendum.txt` + `client_slo_report.md` reference the exact source; `959`/warmup framing removed; BitLesson text fixed.
5. `git diff --check` clean; commit + push to `jimmy`; goal-tracker updated (blockers → resolved-pending-verify; task6); `round-8-summary.md` with BitLesson Delta.

## Out-of-Scope Guards
- No re-run; no production code change; directional verdict unchanged — only durable + internally consistent.
- Do not weaken the strict SLO or mark the loop done; the SLO miss stays a mainline blocker.
- Next round must be hardware (AC-6) so this is not a third consecutive evidence-only round.
