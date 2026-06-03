# Round 9 Summary — exact ITL source + fail-closed AC-5 verifier

## Mainline objective (round contract)
Finish the AC-5 evidence repair so task6/AC-5 is fully exact-recomputable and self-asserting. Codex's R8 review verified the attribution half (row reconciliation, print-time grouping, queue percentiles) and the TTFT/TPOT/length arrays, but found two precise residuals: (1) **ITL not exact-recomputable** from committed files (only the summary was stored; the per-token array sat in the gitignored JSONL); (2) **`--verify` fail-open** (printed `FAIL` but exited 0). Data-only round on the existing R6 hardware run (no re-run). No production code.

## What landed (commit 57f86b66f, pushed to `jimmy`)
1. **Exact ITL source** — `client_slo_int8/ac5_itl_flat_ms.json` (4.5 MB, tracked): per conc the **flattened per-token ITL** (ms, sorted, 4-decimal). `np.percentile` of it reproduces the stored `median_itl_ms`/`p95_itl_ms`/`p99_itl_ms` **exactly** (c16 56.573/58.386/58.695, c32 87.005/87.600/87.944, c64 103.140/103.801/105.268). Referenced from `ac5_metrics_arrays.json` (`itl_source`).
2. **Fail-closed verifier** — `ac5_metrics_tool.py --verify` is now a real acceptance verifier: it recomputes **TTFT, TPOT/TPS, and ITL** percentiles from the committed files alone, runs sanity checks (array lengths == completed == 320, `errors_all_empty`, all `output_lens==512`, ITL flat count), and **`raise SystemExit(1)` on any mismatch**. Demonstrated fail-closure by mutating copies:
   - tampered stored `median_ttft_ms` → `FAIL` + **exit 1**
   - shifted every ITL value +50 ms → ITL `FAIL` + **exit 1**
   - dropped one `ttfts` element → sanity length `FAIL` + **exit 1**
   - clean committed files → **exit 0**, `ALL recomputed==stored + sanity checks: PASS` at all conc.
3. **Docs** — `client_slo_report.md` + `ac5_evidence_addendum.txt` now state TTFT/TPOT/TPS **and ITL** all recompute from committed data via the fail-closed verifier (dropping the R8 over-broad "every number" wording for the precise claim). Attribution files unchanged.

## Result (verdict unchanged; evidence now exact + self-asserting)
DIRECTIONAL — accepted progress, explicitly NOT shippable (DEC-3). Every reported AC-5 percentile (TTFT/TPOT/TPS/ITL) recomputes from committed files, asserted fail-closed; the attribution is reconciled (R8). conc-16 meets strict `<22 s`; conc-32/64 TTFT and all-conc per-req TPS miss the strict SLO and remain the **open mainline blocker**.

## Files Changed
- `runs/20260530_dsv32_loop6/client_slo_int8/`: `ac5_itl_flat_ms.json` (new, exact per-token ITL), `ac5_metrics_tool.py` (build writes ITL source; `--verify` recomputes ITL + sanity checks + fail-closed), `ac5_metrics_arrays.json` (references ITL source; `itl_flat_count`), `ac5_evidence_addendum.txt` (header).
- `runs/20260530_dsv32_loop6/client_slo_report.md` (evidence bundle wording).
- `.humanize/bitlesson.md` (updated `durable-tracked-acceptance-evidence`), goal-tracker, round-9 contract/summary (gitignored loop state).

## Validation
- `ac5_metrics_tool.py --verify` on committed files: **exit 0**, recomputed==stored + sanity PASS at conc 16/32/64.
- Fail-closure demonstrated: three independent tampers (stored value / ITL values / array length) each → **exit 1**.
- New files tracked (`git check-ignore` → none); `git diff --check` clean.
- No re-run, no production code change; the R6 run's `.meta.json` radix-on sidecars + the R8 attribution files unchanged.

## Remaining Items
- **Open mainline blocker:** strict client SLO still fails (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). Characterized (throughput/decode-batch, not footprint).
- **Next round = hardware (AC-6 + AC-9):** boot DSA-default (prove `enable_double_sparsity=false`, no DS `TokenLabelTable`, SLO unchanged) + DS opt-in (compact int8 path toggles on); in the same round edit `test/manual/test_double_sparsity_v32.py` to assert `within_budget` from real `usage.prompt_tokens` (`input_tokens`, rename proxy→`length_words`, fail-closed, **DS-fair thresholds UNCHANGED**) + live rerun + copy artifacts.
- Then **AC-7** (3-trial DS+DSA lifted-point re-sweep), **AC-8** (~70K servability probe), gated **AC-10**. No FlashMLA decode-assert changes (AC-3.3).

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260530-durable-tracked-acceptance-evidence
Notes: Extended the durable-evidence lesson with the R7→R9 benchmark-percentile findings: (c) commit the EXACT numeric source for every PUBLISHED percentile — do not publish a metric (here ITL) whose exact per-token source you did not commit; (d) a recompute/verify script must be FAIL-CLOSED (`SystemExit(1)` on any mismatch) with sanity checks (counts, array lengths, constant-field assertions) — a verifier that prints FAIL but exits 0 is just a readable report, and a reviewer WILL mutate a copy to test it (Codex did, and the fail-open script returned 0); demonstrate fail-closure by mutating a copy. Updated its Validation/Source to the R7(summary-only)→R8(exact TTFT/TPOT but ITL-summary + fail-open)→R9(exact ITL + fail-closed verifier) progression and noted the cost (a 4-round re-review because each pass left one gap). No new lesson — same durable-acceptance-evidence family, made precise. Also reinforces BL-20260530-clean-latency-attribution (clause g, commit exact arrays + recompute script). Pushed 57f86b66f to jimmy per the push-between-rounds preference.
