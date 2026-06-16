# loop11b results — authoritative current-state ledger (rewrite-over-append)

Finish loop 11's M4 verdict on a fresh 8×H200. One TP=8 server at a time; never set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving. Honest-verdict posture: a FAIL on the
throughput SLO is a complete, reportable result.

## Bottom line

**Table-free Double Sparsity on GLM-5.1-FP8 MEETS the client SLO (decode-TPS p50 ≥ 30, P99 TTFT < 22 s)
at concurrency 16 and 32, and FAILS at concurrency 64** (DS decode-TPS 26.98 < 30 AND P99 TTFT 25.12 s
> 22 s). **Native DSA also fails at conc 64** (26.22 TPS, 33.32 s TTFT) — the 30-TPS decode floor is the
binding constraint for BOTH at high concurrency on this node/workload. At the matched op-point DS is
competitive-to-better than DSA: equal-or-higher decode throughput (ratio 0.98–1.03), LOWER P99 TTFT at
every concurrency (ratio 0.46–0.75), equal-or-faster per-step decode (TPOT 0.97–1.02). The op-point was
fully re-established on the fresh node and radix-on re-authorized via the DEC-1 content-hash fixture.

## Milestone status

| milestone | status |
|-----------|--------|
| **M-A op-point re-establishment** (AC-0, AC-5, AC-6, AC-7) | ✅ COMPLETE |
| **M-B M4 close** (AC-4 tax guard, AC-2/AC-3 locked sweep, AC-9) | ✅ COMPLETE (production-envelope; same-memory deferred-recorded) |
| **M-C productionize** (AC-UX) | ✅ COMPLETE |
| close-out (AC-8) | ✅ this ledger + evidence preserved + pushed |

---

## M-A — op-point re-established (GLM-5.1-FP8 TP=8, fp8_e4m3, page 64)

- **Mask (AC-0.1):** a recipe error was caught + fixed. The plan recipe carried the DeepSeek-V3.2 values
  (`--dtype bfloat16 --label-dim 16`); that mask served −5.2pp recall at L4096. The GLM-native loop8 DEC-3
  recipe is **`--dtype fp8_e4m3 --label-dim 32`** (∝ qk_nope_head_dim=192, 2× channels). Corrected mask
  `content_sha256=35155ac4…`, 78×64×32, dry-run placement gate PASS, serves. provenance.json committed.
- **Recall (AC-5):** served radix-OFF == frozen baseline on matched populations (L4096 58.045% = baseline;
  L16384 +0.32pp; overall 64.80% vs 64.70) — within ±0.5pp. Radix on-vs-off equivalence (num=60, 18720
  samples/len): all per-length |Δ| ≤ 0.283pp → `recall_equivalence_passed`.
- **Radix authorization (AC-0.2, DEC-1):** validator pins `channel_mask_content_sha256` (not path/full-file);
  +2 portability unit tests, 13/13 fixture tests green. DEC-12 mint gates ALL PASS — GATE A recall equiv,
  GATE B cross-rank identity (8 ranks byte-identical) + no-dense-fallback, GATE C production-reuse edge
  (boundary −0.38 / partial@2752 −0.01 / evict 0.0 pp within ±0.5pp; nearfull +1.5692pp OUT-OF-CONTRACT,
  recorded). Fixture minted; **no-override boot AUTHORIZED live** + **DEC-1 same-content/different-path
  portability AUTHORIZED**.
- **Capacity + AC-7 (AC-0.3, AC-7):** DS table-free @ mem 0.8 token_capacity **504640** (= loop11 ref),
  CUDA-graph capture OK, no TokenLabelTable. DSA-native @ mem 0.8 token_capacity **410560** (= loop11 ref)
  — DEC-1 shared-surface change did not regress the shipped DSA default.
- **DS concept (AC-6):** offline mask → absorbed-latent table-free selection → top-k → sparse MLA; no dense
  fallback, no DSA-indexer substitution.

## M-B — M4 verdict (production-envelope, 2 trials/conc, 600 s, radix-ON both)

| op-point | conc | decode-TPS p50 (≥30) | P99 TTFT (s, <22) | verdict |
|----------|------|----------------------|-------------------|---------|
| DS  | 16 | 40.75 | 1.59  | **PASS** |
| DS  | 32 | 34.12 | 3.20  | **PASS** |
| DS  | 64 | 26.98 | 25.12 | **FAIL** |
| DSA | 16 | 41.50 | 3.50  | PASS |
| DSA | 32 | 33.34 | 6.80  | PASS |
| DSA | 64 | 26.22 | 33.32 | FAIL |

- **AC-2/AC-3 (DEC-6 absolute):** DS PASS @ conc 16/32, FAIL @ conc 64. Judged regardless of DSA.
- **AC-4 per-step tax:** DS/DSA TPOT p50 ratio 0.972–1.018 (conc 16/32/64) ≤ 1.10 → **PASS** (loop-10 win held).
- **DS/DSA ratios (reported, DEC-6):** decode-TPS 0.98–1.03; P99 TTFT 0.46–0.75 (DS lower everywhere);
  TPOT 0.97–1.02. DS admission-capped <64 at conc-64 (achieved 58.9).
- **AC-9 honesty:** block-scheduled (labeled unpaired); the `--ac11` comparator refused the cross-side
  match (first op-point caps — fixed; then a benign commit_sha mismatch) — the verdict is extracted via the
  comparator's own metric readers (`extract_verdict.py`); the DS absolute verdict is DS-only and
  commit-independent. Same-memory op-point (both 0.8) **deferred-and-recorded** (plan lower bound).
- **Open tooling gaps (queue SI-1/SI-2/SI-5):** per-request prefix-reuse + aggregate-throughput + no-op
  total_tokens are not emitted by bench_serving/the comparator; the GSP workload is ~55% reuse by
  construction. Recorded as queued follow-ups; do not affect the gated per-request verdict.

## M-C — productionize (AC-UX)

Runbook (`development/loop11b/RUNBOOK.md`) takes a GLM-5.1-FP8 operator zero→serving DS; Category-A fixes
(serve-script model/mask defaults + mem/TokenLabelTable comments; loop8 throughput warning reconciled to
the measured verdict; de-DeepSeek calibrate/config; CLIENT_SLOS→SLOS; trials wording) + Category-B CLI help
(server_args.py). No flag rename / JSON-schema change (DEC-5). serve_native_nsa.sh now matches the locked
op-point (64/64).

## Evidence (AC-8)

`runs/20260616_ma/` — provenance.json, capacity_ds_evidence.md, mint/ (env + gate_a/b/c runners +
mint_fixture.py + verify_no_override.sh + dsa_capacity_probe.sh, probes/ verdicts + server_info snapshots).
`runs/20260616_mb/` — sweep.sh, dsa_rerun.sh, extract_verdict.py, DS_absolute_verdict.md, results_prod_envelope/
(verdict_matched.json, .meta.json sidecars, server_info). Reproducers: run_calibrate.sh, build_corpus.py.
Bulky `.jsonl`/`.log`/mask blobs are gitignored (reproducible from the committed runners + recorded hashes).
