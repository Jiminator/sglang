# loop11b R2 — reproduce the M-B verdict from committed artifacts alone

All verdict inputs are committed here losslessly. Raw bench JSONLs and per-boot serve logs are stored
gzip-compressed (`*.jsonl.gz` / `*.log.gz`); `EVIDENCE_SHA256.txt` carries the SHA-256 of BOTH the raw
(decompressed) file and the committed `.gz`, so a reviewer can verify the decompression is exact.

Frozen HEAD: `b0e448b1` (commit_sha stamped in every `*.meta.json`). One TP=8 server at a time.

## 1. Decompress the raw comparator inputs + logs
    cd development/loop11b/runs/20260616_mb/results_r2
    gunzip -k ds080/*.jsonl.gz dsa080/*.jsonl.gz dsa085/*.jsonl.gz tax/*.jsonl.gz *.log.gz

## 2. Verify the decompression is exact (raw hashes in EVIDENCE_SHA256.txt)
    sha256sum -c <(grep -vE '\.gz$' EVIDENCE_SHA256.txt)     # the .jsonl / .log lines

## 3. Re-run the comparator from the decompressed artifacts (+ the committed .meta.json sidecars)
    python3 ../../../benchmark_compare.py --ac11 \
      --ac11-baseline-results dsa085/native_nsa_gsp_isl4096_osl512_c*_t*.jsonl \
      --ac11-ds-results ds080/double_sparsity_gsp_isl4096_osl512_c*_t*.jsonl \
      --output /tmp/pe.md --json-output /tmp/pe.json
    # production_envelope -> rc=3 (absolute DS SLO FAIL@64); same against dsa080/ -> same_memory rc=3.
VALIDATED: re-running from the gunzipped `.gz` + `.meta.json` reproduces production_envelope rc=3,
client_slo_verdict=FAIL, conc-64 DS decode-TPS 26.92 (== `ac11_production_envelope.json`).

## 4. Per-trial no-op evidence (AC-5)
    for j in ds080/double_sparsity_*.jsonl; do python3 ../trial_evidence.py "$j"; done
    # all 6 -> verdict PASS: dense_fallback_total==0, selected_tokens_mean 2048 < total_tokens_mean, reuse ~54%.

## What is committed (stable names)
- `ds080/`,`dsa080/`,`dsa085/`: per-trial `*.jsonl.gz` (raw bench inputs) + `*.meta.json` (lossless sidecars:
  commit_sha, server_args/op-point, seed, aggregate stats) + `ds080/*.evidence.json` (per-trial no-op verdict).
- `tax/`: `*.jsonl.gz` + `log_*.txt` (AC-4 dedicated per-step probe; DS vs DSA at fixed bs64/bs30).
- `ac11_{production_envelope,same_memory}.{md,json}`: comparator outputs (rc=3 both).
- `serve_{ds080,dsa080,dsa085}.log.gz`: per-boot serve logs. `server_info_*.json`: /server_info snapshots.
- `mb_r2.log.gz`: the orchestrator log = run order + per-phase peaks/health. `../mb_r2.sh`: the command ledger.
- `EVIDENCE_SHA256.txt`: raw + .gz content hashes.

## Full regeneration from scratch (optional)
`bash ../mb_r2.sh` (boots DS@0.8 + DSA@0.8 + DSA@0.85 from the env in `../../20260616_ma/mint/env.sh`,
sweep-first, distinct-prefix tax, comparators, per-trial evidence). ~4 h on a fresh 8×H200.
