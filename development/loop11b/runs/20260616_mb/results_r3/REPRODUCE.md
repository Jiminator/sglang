# loop11b R3 — reproduce the M-B verdict from committed artifacts alone (corrected total_tokens)

This is the PUBLISHED verdict evidence. It supersedes `../results_r2/` (R3 fixes the AC-5
`total_tokens_mean` aggregate — see `../R1_DS_CRASH_FINDING.md` / round summaries). Raw bench JSONLs and
serve logs are committed gzip-compressed (`*.jsonl.gz` / `*.log.gz`); `EVIDENCE_SHA256.txt` carries the
SHA-256 of BOTH the raw (decompressed) file and the committed `.gz`.

Frozen HEAD: `8df44a59c` (commit_sha in every `*.meta.json`). One TP=8 server at a time.

## 1. Decompress the raw comparator inputs + logs
    cd development/loop11b/runs/20260616_mb/results_r3
    gunzip -k ds080/*.jsonl.gz dsa080/*.jsonl.gz dsa085/*.jsonl.gz tax/*.jsonl.gz *.log.gz

## 2. Verify the decompression is exact
    sha256sum -c <(grep -vE '\.gz$' EVIDENCE_SHA256.txt)

## 3. Re-run the comparator from the decompressed artifacts (+ committed .meta.json sidecars)
    python3 ../../../benchmark_compare.py --ac11 \
      --ac11-baseline-results dsa085/native_nsa_gsp_isl4096_osl512_c*_t*.jsonl \
      --ac11-ds-results ds080/double_sparsity_gsp_isl4096_osl512_c*_t*.jsonl \
      --output /tmp/pe.md --json-output /tmp/pe.json     # production_envelope -> rc=3
    # same against dsa080/ -> same_memory rc=3.
VALIDATED: re-run from the gunzipped `.gz` + `.meta.json` reproduces production_envelope rc=3,
client_slo_verdict=FAIL, conc-64 DS decode-TPS 26.91 (== `ac11_production_envelope.json`).

## 4. Per-trial no-op evidence (AC-5) — now with the consistency gate
    for j in ds080/double_sparsity_*.jsonl; do python3 ../trial_evidence.py "$j"; done
    # all 6 -> verdict PASS: dense_fallback_total==0, selected_tokens_mean 2048 < total_tokens_mean ~4765
    # (the TRUE sequence-length total; trial_evidence now REFUSES if the aggregate disagrees with the
    #  per-request total_tokens array or sparsity_rate != 1 - selected/total).

## Verdict (unchanged conclusion)
DS PASS@conc16 (40.70 TPS / 1.58 s) + conc32 (34.05 / 3.00), FAIL@conc64 (26.91 < 30, 25.11 s ≥ 22).
DSA also fails @64. Both comparators rc=3. Competitive-to-better than DSA at both op-points.

## What is committed (stable names) — same layout as the validated R2 package
`{ds080,dsa080,dsa085}/` per-trial `*.jsonl.gz` + `*.meta.json` + `ds080/*.evidence.json`;
`tax/*.jsonl.gz` + `log_*.txt`; `ac11_{production_envelope,same_memory}.{md,json}` (rc=3);
`serve_*.log.gz`; `server_info_*.json`; `mb_r3.log.gz` (run order); `EVIDENCE_SHA256.txt`;
`../mb_r3.sh` (command ledger). Full regeneration: `bash ../mb_r3.sh` (~4 h, fresh 8×H200).
