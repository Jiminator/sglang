# Round 9 Contract

## Mainline Objective (exactly one)
**Finish the AC-5 evidence repair so task6/AC-5 is fully exact-recomputable and self-asserting.** Codex's R8 review verified the attribution half (row reconciliation, print-time grouping, queue percentiles) and the TTFT/TPOT/length arrays, but found two precise residual gaps: (1) **ITL is not exact-recomputable from committed files** — `ac5_metrics_arrays.json` stores only the ITL summary (count + median/p95/p99) with a note to flatten from the gitignored JSONL, so the ITL lines in `ac5_evidence_addendum.txt` cannot be recomputed from committed data; (2) **`ac5_metrics_tool.py --verify` is fail-open** — it prints `FAIL` but exits 0 (confirmed by Codex mutating a copy), contradicting the report's "asserts recomputed == stored" claim. Data-only round on the existing R6 hardware run (no re-run). No production code.

## Target ACs (1–2)
- **AC-5** (`coding`, hardware-derived data) — close the exact-ITL-recomputation + fail-closed-verifier gaps.

## Blocking Side Issues in Scope (Codex R8 review)
1. **Commit an exact ITL source.** Add a tracked per-conc flattened per-token ITL numeric array (`ac5_itl_flat_ms.json`, sorted ascending ms, 4-decimal — verified to reproduce stored `median_itl_ms`/`p95_itl_ms`/`p99_itl_ms` exactly). Reference it from `ac5_metrics_arrays.json`.
2. **Make `--verify` a real acceptance verifier (fail-closed).** It must recompute median/p95/p99 ITL from the committed ITL source and TTFT/TPOT/TPS from the arrays, compare all to stored, run sanity checks (completed count == array lengths == 320, `errors_all_empty`, all `output_lens == 512`, ITL flat count), and **`raise SystemExit(1)` on any mismatch** (currently it only prints FAIL and exits 0).
3. **Point docs at the exact source only after all three recompute.** Update `ac5_evidence_addendum.txt` / `client_slo_report.md` so "recomputable" covers TTFT, TPOT/TPS, **and** ITL; correct the Round-8 over-broad "every AC-5 number recomputes" wording.

## Queued / Out of Scope (explicitly NOT downgraded)
- **Strict-SLO failure stays a visible mainline blocker** (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). Not fixable by data work; remains tracked.
- **AC-6** (DSA-default product proof), **AC-7** (3-trial re-sweep), **AC-8** (~70K probe), **AC-9** (real-token within-budget edit + live rerun), gated **AC-10** — the **next round is the hardware round (AC-6 + AC-9 code edit + rerun)**; this is the last AC-5-evidence-only round. No FlashMLA decode-assert changes (AC-3.3); do not change DS-fair thresholds (AC-9).

## Round Success Criteria
1. Tracked `ac5_itl_flat_ms.json`: per conc flattened per-token ITL (ms) that reproduces stored ITL percentiles exactly.
2. `ac5_metrics_tool.py --verify` recomputes TTFT/TPOT/TPS **and ITL** from committed files, runs the sanity checks, and **exits nonzero on any mismatch** (demonstrated: a mutated copy makes it exit 1).
3. `ac5_metrics_arrays.json` references the ITL source; `ac5_evidence_addendum.txt` + `client_slo_report.md` reflect that TTFT/TPOT/TPS/ITL all recompute from committed data (drop the over-broad wording).
4. `git diff --check` clean; commit + push to `jimmy`; goal-tracker updated (AC-5 evidence blocker → resolved-pending-verify; task6); `round-9-summary.md` with BitLesson Delta.

## Out-of-Scope Guards
- No re-run; no production code change; directional verdict unchanged — only the evidence is made exact + self-asserting.
- Do not weaken the strict SLO or mark the loop done; the SLO miss stays a mainline blocker.
- Next round must be hardware (AC-6/AC-9) — no fourth consecutive evidence-only round.
