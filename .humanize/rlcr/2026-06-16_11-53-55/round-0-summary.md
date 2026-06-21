# Round 0 Summary — loop11b: finish loop 11's M4 verdict on a fresh 8×H200

**Outcome: the loop is COMPLETE — all 11 mainline tasks done, all milestones met, an honest measured
verdict delivered.** 32 commits on `dev/double-sparsity-standalone` (local; not pushed — see Remaining).

## What Was Implemented

**The headline verdict (the loop's reason to exist):** Table-free Double Sparsity on GLM-5.1-FP8 MEETS the
client SLO (decode-TPS p50 ≥ 30, P99 TTFT < 22 s) at concurrency 16 and 32, and FAILS at concurrency 64 (DS
decode-TPS 26.98 < 30 AND P99 TTFT 25.12 s > 22 s). Native DSA also fails at conc 64 (26.22 TPS, 33.32 s
TTFT) — the 30-TPS decode floor is the binding constraint for BOTH at high concurrency. At the matched
op-point DS is competitive-to-better than DSA: decode-TPS ratio 0.98–1.03, P99 TTFT ratio 0.46–0.75 (DS lower
at every conc), per-step TPOT 0.97–1.02. A documented FAIL at conc-64 is a complete result.
Report: `development/loop11b/runs/20260616_mb/DS_absolute_verdict.md`; ledger: `development/loop11b/results.md`.

**M-A op-point re-establishment (AC-0/5/6/7) — COMPLETE.**
- Regenerated the GLM-5.1 channel mask. CAUGHT + FIXED a recipe error: the plan/AC-0.1 recipe carried the
  DeepSeek-V3.2 values (`--dtype bfloat16 --label-dim 16`), which served −5.2pp recall at L4096; the
  GLM-native loop8 DEC-3 recipe is `--dtype fp8_e4m3 --label-dim 32`. Corrected mask `content_sha256=35155ac4…`
  serves; recall matches the frozen baseline (L4096 58.045% = baseline). FP8 dry-run gate passed; provenance.json
  records both hashes + command + env + the recipe correction.
- Landed DEC-1: validator pins `channel_mask_content_sha256` (tensor-content) instead of path + full-file SHA
  (+2 portability unit tests; 13/13 fixture tests green; orphan `_sha256_file` removed).
- Re-minted radix-on via the full DEC-12 battery, all PASS: GATE A (recall on-vs-off equiv num=60), GATE B
  (cross-rank identity 8 ranks byte-identical + no-dense-fallback), GATE C (production-reuse edge: boundary/
  partial@2752/evict within ±0.5pp; nearfull +1.5692pp recorded out-of-contract). Fixture minted; no-override
  boot AUTHORIZED live + DEC-1 same-content/different-path portability AUTHORIZED.
- Capacity: DS table-free @0.8 token_capacity 504640 (= loop11 ref), no TokenLabelTable; DSA-native @0.8
  410560 (= loop11 ref) — AC-7 un-regressed by the shared-surface change.

**M-B M4 verdict (AC-2/3/4/9) — COMPLETE.** Comparator tweaked for DEC-4 (trial floor 3→2) + DEC-6 (exit
gates the absolute SLO; DS/DSA ratio report-only). Production-envelope locked sweep (2 trials, 600 s, conc
16/32/64, radix-on both); DSA re-run at the matched 64/64 op-point. AC-4 per-step tax PASS (TPOT ratio ≤ 1.10).
Same-memory op-point deferred-and-recorded (plan lower bound).

**M-C productionize (AC-UX) — COMPLETE.** RUNBOOK.md (zero→serving DS), Category-A fixes (serve/calibrate/
config/benchmark de-DeepSeek; loop8 throughput warning reconciled to the measured verdict; CLIENT_SLOS→SLOS),
Category-B CLI help (server_args.py). No flag rename / JSON-schema change (DEC-5).

**Close-out (AC-8) — COMPLETE.** results.md regenerated (rewrite-over-append); evidence preflight PASS (all
verdict-bearing artifacts tracked; fixture hash = served mask); queue.md final.

## Files Changed
- Code: `double_sparsity/validator.py` (DEC-1), `benchmark_compare.py` (DEC-4/DEC-6), `serve_native_nsa.sh`
  (op-point caps + de-DeepSeek), `serve_double_sparsity.sh` (warning reconcile + de-DeepSeek), `calibrate.py` /
  `config.py` / `server_args.py` (de-DeepSeek docs/help). Tests: `test_double_sparsity_unit.py` (+2 DEC-1 tests).
- Harness/evidence (committed): `development/loop11b/` — run_calibrate.sh, build_corpus.py, RUNBOOK.md,
  results.md, queue.md, draft.md, plan.md; `runs/20260616_ma` (provenance.json, capacity_ds_evidence.md, mint
  runners + probe verdicts + server_info); `runs/20260616_mb` (sweep.sh, dsa_rerun.sh, tax_guard.sh,
  extract_verdict.py, DS_absolute_verdict.md, verdict_matched.json, .meta sidecars). Bulky .jsonl/.log/mask/.pt
  blobs gitignored (reproducible from committed runners + recorded hashes).

## Validation
- 13/13 `radix_fixture` unit tests pass (incl. +2 new DEC-1 same-content/different-path + different-content).
- Live: DEC-12 GATE A/B/C all PASS; no-override + altpath authorization boots PASS; DS recall = frozen baseline
  (matched population); DS capacity 504640 / DSA AC-7 410560 match; locked sweep produced the verdict above.

## Remaining Items
- NOT PUSHED: `origin` is the PUBLIC `sgl-project/sglang` upstream; pushing experimental loop11b commits there
  needs explicit owner authorization. Commits are LOCAL (RLCR keeps commits local by design; AC-8 "push" is
  intentionally deferred to the owner given the remote is the public upstream).
- Same-memory op-point (both 0.8) deferred-and-recorded (plan lower bound; production-envelope is the published verdict).
- Queued tooling gaps (SI-1/2/5/6 in queue.md): bench_serving doesn't dump per-request cached_tokens (prefix-reuse
  is ~55% by GSP construction, not per-request measured), no `total_tokens` no-op field, aggregate throughput not
  carried by the comparator. None affect the gated per-request verdict; logged as follow-ups.
- AC-4 measured via the sweep's conc-64 TPOT ratio (bench_one_batch unsuitable: skips the DS validator → mask
  unbound, and OOMs on a single non-chunked bs×4096 prefill).

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260616-glm-ds-calib-recipe
Notes: Added a lesson capturing the GLM DS calibration recipe (DEC-3 `label_dim=32` + `fp8_e4m3`, NOT the V3.2
values) and the rule to verify served recall vs the frozen baseline after any mask regen — a wrong recipe
silently degraded recall ~5pp while radix on-vs-off equivalence still passed.
