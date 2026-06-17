# loop11b results — authoritative current-state ledger (rewrite-over-append)

Finish loop 11's M4 verdict on a fresh 8×H200. One TP=8 server at a time; never set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving. Honest-verdict posture: a FAIL on the
throughput SLO is a complete, reportable result.

## Bottom line

**Table-free Double Sparsity on GLM-5.1-FP8 MEETS the client SLO (decode-TPS p50 ≥ 30, P99 TTFT < 22 s)
at concurrency 16 and 32, and FAILS at concurrency 64** (DS decode-TPS 26.98 < 30 AND P99 TTFT 25.08 s
≥ 22 s). **Native DSA also fails at conc 64** (26.13–26.20 TPS, 33.2 s TTFT) — the 30-TPS decode floor is
the binding constraint for BOTH at high concurrency on this node/workload. At BOTH op-points (production-
envelope and same-memory) DS is competitive-to-better than DSA: equal-or-higher decode throughput
(ratio 0.98–1.03), LOWER P99 TTFT at every concurrency (ratio 0.44–0.76), and ≤ 6 % per-step decode tax
(dedicated probe). The op-point was re-established on the fresh node and radix-on re-authorized via the
DEC-1 content-hash fixture. The R1 verdict is from `--ac11` comparator-ACCEPTED artifacts at one frozen
HEAD (commit_sha 99ac584ac).

## Milestone status

| milestone | status |
|-----------|--------|
| **M-A op-point re-establishment** (AC-0, AC-5, AC-6, AC-7) | ✅ COMPLETE |
| **M-B M4 close** (AC-2/3/4/9) | ✅ COMPLETE — BOTH op-points comparator-ACCEPTED, conc-64 nominal (peak 63) |
| **M-C productionize** (AC-UX) | ✅ COMPLETE |
| close-out (AC-8) | ✅ this ledger + evidence hashes committed; push pending owner direction (see below) |

## R1 correction (what Round 0 got wrong)

Round 0 claimed M-B publishable; Codex rejected it. R1 fixes, all landed:
- **The `--ac11` comparator now ACCEPTS** both op-points (was refused: op-point caps, then commit_sha). DS +
  DSA re-run from ONE frozen HEAD → production_envelope rc=3, same_memory rc=3 (honest absolute DS FAIL@64).
- **Real DS bug found + fixed (`R1_DS_CRASH_FINDING.md`):** the DS per-request error-abort path called
  `req.check_finished()`, renamed upstream (#25725) to `update_finish_state()` → AttributeError crashed the
  whole TP=8 server when the DS selector sanitized a row. Fixed. It is a semantic merge conflict that escaped
  detection because the abort path only runs on a selector error, which representative load never triggered.
- **conc-64 reaches nominal:** peak running-req 63 (≥ 61); achieved 58.9 is a real DS smaller-KV-pool effect.
- **AC-4 is now the spec'd dedicated probe** (not sweep-derived TPOT).
- **AC-5 no-op + reuse evidence** emitted per trial (B1 bench_serving change).

## M-A — op-point re-established (GLM-5.1-FP8 TP=8, fp8_e4m3, page 64)

- **Mask (AC-0.1):** a recipe error was caught + fixed. The plan recipe carried the DeepSeek-V3.2 values
  (`--dtype bfloat16 --label-dim 16`); that mask served −5.2pp recall at L4096. The GLM-native loop8 DEC-3
  recipe is **`--dtype fp8_e4m3 --label-dim 32`** (∝ qk_nope_head_dim=192, 2× channels). Corrected mask
  `content_sha256=35155ac4…`, 78×64×32, dry-run placement gate PASS, serves. provenance.json committed.
- **Recall (AC-5):** served radix-OFF == frozen baseline on matched populations (L4096 58.045% = baseline;
  L16384 +0.32pp; overall 64.80% vs 64.70) — within ±0.5pp. Radix on-vs-off equivalence ≤ 0.283pp.
- **Radix authorization (AC-0.2, DEC-1):** validator pins `channel_mask_content_sha256` (not path/full-file);
  +2 portability tests, 13/13 fixture tests green. DEC-12 mint gates ALL PASS — GATE A recall equiv,
  GATE B cross-rank identity (8 ranks byte-identical) + no-dense-fallback, GATE C production-reuse edge
  (boundary −0.38 / partial −0.01 / evict 0.0 pp; nearfull +1.5692pp OUT-OF-CONTRACT, recorded). Fixture
  minted; no-override boot AUTHORIZED live + DEC-1 same-content/different-path portability AUTHORIZED.
- **Capacity + AC-7:** DS table-free @ mem 0.8 token_capacity **504640** (= loop11 ref; reconfirmed at the
  ld32 no-override boot), CUDA-graph capture OK, no TokenLabelTable. DSA-native @ mem 0.8 **410560** (= ref)
  — DEC-1 shared-surface change did not regress the shipped DSA default.

## M-B — M4 verdict (R1, both op-points, 2 trials/conc, 600 s, radix-ON, comparator-ACCEPTED)

DS absolute client SLO (DEC-2/DEC-6: decode-TPS p50 ≥ 30 AND P99 TTFT < 22 s, judged regardless of DSA):

| conc | DS decode-TPS p50 | DS P99 TTFT | DS SLO |
|------|-------------------|-------------|--------|
| 16 | 40.73 | 1.59 s | **PASS** |
| 32 | 34.13 | 2.99 s | **PASS** |
| 64 | 26.98 | 25.08 s | **FAIL** (TPS < 30 AND TTFT ≥ 22) |

DS vs DSA directional (REPORTED, DEC-6) — both comparator-ACCEPTED:
- production_envelope (DS0.8/DSA0.85): TPS ratio 0.977 / 1.015 / 1.032; TTFT ratio 0.456 / 0.441 / 0.755.
- same_memory (DS0.8/DSA0.8): TPS ratio 0.983 / 1.020 / 1.030; TTFT ratio 0.462 / 0.436 / 0.754.
- **AC-4 per-step tax (dedicated distinct-prefix probe, mem 0.8 both sides):** bs64 1.056, bs30 1.057
  — both ≤ 1.10 PASS; bs30 31 850 µs ≪ 380 000. Loop-10 per-step parity held.
- **AC-2/3 admission:** peak running-req 63 (nominal reached); achieved 58.9 = real DS pool effect at mem 0.8.
- **AC-5/AC-9 no-op + reuse:** ~54 % measured prefix reuse per trial; 0 dense_fallback (all 6 trials);
  sparse selection structural (top_k 2048 < 4096 ctx) + 4303 DS decode batches. Per-request DS meta_info
  aggregate is unwired for GLM (`Glm4MoeAttention`/dsa-backend never reaches DeepseekV2's publisher) —
  documented observability gap + recommended backend-side fix (`ac5_no_op_evidence.md`).

Headline: `runs/20260616_mb/R1_HEADLINE_VERDICT.md`.

## M-C — productionize (AC-UX)

Runbook (`RUNBOOK.md`) takes a GLM-5.1-FP8 operator zero→serving DS; Category-A fixes (serve-script
model/mask defaults + mem/TokenLabelTable comments; loop8 throughput warning reconciled; de-DeepSeek
calibrate/config; CLIENT_SLOS→SLOS; trials wording; plan-vocabulary stripped from operator output) +
Category-B CLI help. No flag rename / JSON-schema change (DEC-5). serve_native_nsa.sh matches the locked
op-point (64/64).

## Evidence (AC-8)

`runs/20260616_ma/` — provenance.json, capacity_ds_evidence.md (ld32 504640 reconfirm), mint/ (gate runners
+ probes/ verdicts + server_info).
`runs/20260616_mb/` — R1_HEADLINE_VERDICT.md, R1_DS_CRASH_FINDING.md, ac5_no_op_evidence.md; ds_only.sh /
dsa_tax.sh / mb_v2.sh (runners); results_v2/ (ac11_{production_envelope,same_memory}.{md,json} comparator-
accepted; ds080,dsa080,dsa085 .meta.json + .evidence.json sidecars; tax/log_*.txt; server_info;
EVIDENCE_SHA256.txt content hashes; crash_evidence_r1/). Bulky `.jsonl`/`.log` blobs are gitignored
(reproducible from the committed runners + recorded hashes). Verdict commits: 8fbe848ed (sweep+comparators),
9d2c4253d (headline+AC-4); crash-fix 99ac584ac.

## Push status (AC-8)

All commits are LOCAL on `dev/double-sparsity-standalone`. `origin` is the PUBLIC `sgl-project/sglang`
upstream; pushing experimental loop11b artifacts there needs explicit owner authorization (or an owner-
designated fork/branch). **Push is therefore pending owner direction** — recorded, not silently skipped.
