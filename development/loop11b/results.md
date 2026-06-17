# loop11b results — authoritative current-state ledger (rewrite-over-append)

Finish loop 11's M4 verdict on a fresh 8×H200. One TP=8 server at a time; never set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving. Honest-verdict posture: a FAIL on the
throughput SLO is a complete, reportable result.

## Bottom line

**Table-free Double Sparsity on GLM-5.1-FP8 MEETS the client SLO (decode-TPS p50 ≥ 30, P99 TTFT < 22 s)
at concurrency 16 and 32, and FAILS at concurrency 64** (DS decode-TPS 26.91 < 30 AND P99 TTFT 25.11 s
≥ 22 s). **Native DSA also fails at conc 64** (26.20 TPS, 33.2 s TTFT) — the 30-TPS decode floor is
the binding constraint for BOTH at high concurrency on this node/workload. At BOTH op-points (production-
envelope and same-memory) DS is competitive-to-better than DSA: equal-or-higher decode throughput
(ratio 0.98–1.03), LOWER P99 TTFT at every concurrency (ratio 0.44–0.76), and ≤ 6 % per-step decode tax
(dedicated probe). The op-point was re-established on the fresh node and radix-on re-authorized via the
DEC-1 content-hash fixture. The published verdict is from `--ac11` comparator-ACCEPTED artifacts at one
frozen HEAD (commit_sha 8df44a59c), with every DS trial's no-op evidence PASSING (`results_r3/`).

## Milestone status

| milestone | status |
|-----------|--------|
| **M-A op-point re-establishment** (AC-0, AC-5, AC-6, AC-7) | ✅ COMPLETE |
| **M-B M4 close** (AC-2/3/4/9) | ✅ COMPLETE — BOTH op-points comparator-ACCEPTED, conc-64 nominal (peak 63) |
| **M-B AC-5 no-op** | ✅ RESOLVED (R3) — GLM/dsa-backend DS summary wired + `total_tokens` corrected (true seq-len, not the rate-inverse); all 6 DS trials `trial_evidence.py` PASS with a consistency gate |
| **M-C productionize** (AC-UX) | ✅ COMPLETE |
| close-out (AC-8) | 🔶 ACTIVE — ledgers regenerated + raw evidence committed losslessly (`.gz` + hashes + REPRODUCE, replay-validated rc=3); **NOT complete until push**: pending owner remote/waiver (see below) |

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

## M-B — M4 verdict (R3 re-run, both op-points, 2 trials/conc, 600 s, radix-ON, comparator-ACCEPTED)

DS absolute client SLO (DEC-2/DEC-6: decode-TPS p50 ≥ 30 AND P99 TTFT < 22 s, judged regardless of DSA):

| conc | DS decode-TPS p50 | DS P99 TTFT | DS SLO |
|------|-------------------|-------------|--------|
| 16 | 40.70 | 1.58 s | **PASS** |
| 32 | 34.05 | 3.00 s | **PASS** |
| 64 | 26.91 | 25.11 s | **FAIL** (TPS < 30 AND TTFT ≥ 22) |

DS vs DSA directional (REPORTED, DEC-6) — both comparator-ACCEPTED (rc=3) at one frozen HEAD 8df44a59c:
- production_envelope (DS0.8/DSA0.85): DS ≥ DSA throughput (ratio ~0.98–1.03), lower P99 TTFT every conc.
- same_memory (DS0.8/DSA0.8): comparator-accepted rc=3; DS ≥ DSA throughput, lower TTFT every conc.
- **AC-4 per-step tax (dedicated distinct-prefix probe, mem 0.8 both sides):** bs64/bs30 DS/DSA ITL ratio
  ≈ 1.06 — both ≤ 1.10 PASS; bs30 ≪ 380 000 µs. Loop-10 per-step parity held.
- **AC-2/3 admission:** peak running-req 63 (nominal reached); achieved 58.9 = real DS pool effect at mem 0.8.
- **AC-5/AC-9 no-op + reuse (RESOLVED in R3):** every published DS trial carries the per-request DS summary —
  `dense_fallback_total = 0`, `selected_tokens_mean 2048 < total_tokens_mean ~4765` (the TRUE sequence-length
  total; R2 mislabeled it ~3590 by inverting the wrong fraction — fixed). `trial_evidence.py` PASSES on all 6
  with a strengthened consistency gate (refuses if the aggregate disagrees with the per-request arrays or
  `sparsity_rate != 1 - selected/total`). Reuse ~54 % measured per trial.

Headline: `runs/20260616_mb/R1_HEADLINE_VERDICT.md` (numbers reproduce within noise; R3 = results_r3/).

## M-C — productionize (AC-UX)

Runbook (`RUNBOOK.md`) takes a GLM-5.1-FP8 operator zero→serving DS; Category-A fixes (serve-script
model/mask defaults + mem/TokenLabelTable comments; loop8 throughput warning reconciled; de-DeepSeek
calibrate/config; CLIENT_SLOS→SLOS; trials wording; plan-vocabulary stripped from operator output) +
Category-B CLI help. No flag rename / JSON-schema change (DEC-5). serve_native_nsa.sh matches the locked
op-point (64/64).

## Evidence (AC-8) — the verdict reproduces from committed artifacts alone

PUBLISHABLE evidence is `runs/20260616_mb/results_r3/` (R3, HEAD 8df44a59c; supersedes results_r2, which
mislabeled total_tokens). `REPRODUCE.md` gives the exact decompress + comparator + trial_evidence commands;
re-running the comparator from the decompressed `.gz` + `.meta.json` was VALIDATED to reproduce
production_envelope rc=3 / FAIL@64 / DS 26.91 TPS exactly.
- `results_r3/{ds080,dsa080,dsa085}/`: per-trial **`*.jsonl.gz`** (raw bench inputs, committed losslessly) +
  `*.meta.json` (commit_sha/op-point/aggregate sidecars) + `ds080/*.evidence.json` (per-trial no-op PASS).
- `results_r3/tax/` `*.jsonl.gz` + `log_*.txt`; `ac11_{production_envelope,same_memory}.{md,json}` (rc=3);
  `serve_*.log.gz` (per-boot logs); `server_info_*.json`; `mb_r3.log.gz` (run order); `EVIDENCE_SHA256.txt`
  (raw + .gz hashes); `../mb_r3.sh` (command ledger).
- `runs/20260616_ma/` — provenance.json, capacity_ds_evidence.md (ld32 504640 reconfirm), mint/ probes.
- Prior-round narrative: R1_HEADLINE_VERDICT.md, R1_DS_CRASH_FINDING.md (the check_finished crash-fix),
  ac5_no_op_evidence.md (superseded by the wired publisher); results_r2/SUPERSEDED.md.
- Verdict commits: AC-5 publisher b0e448b1; total_tokens fix 8df44a59c; R3 evidence c805b4be5; crash-fix 99ac584ac.

## Push status (AC-8) — THE ONE REMAINING ITEM (owner decision)

Everything else in AC-8 is done: ledgers are one current state, raw evidence is committed losslessly and the
verdict replays from committed artifacts (rc=3). **Close-out is NOT marked complete because the push is not
done.** All commits are LOCAL on `dev/double-sparsity-standalone`; the ONLY configured remote is `origin =`
PUBLIC `github.com/sgl-project/sglang`, and there is no fork/owner remote. Pushing experimental loop11b
artifacts (incl. ~84 MB of compressed raw evidence) to the public upstream is an irreversible outward action
that needs owner authorization, and a destination cannot be fabricated. **Resolve exactly one way:**
(a) provide an owner-approved fork/remote+branch → `git push <remote> dev/double-sparsity-standalone`, or
(b) record an explicit written owner waiver here. Surfaced, not silently skipped.
