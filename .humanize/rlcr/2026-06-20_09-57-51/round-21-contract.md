# Round 21 Contract

Round 20 was ADVANCED; Codex VERIFIED AC-3.1 (AC-3 now MET). Remaining: AC-4 (serial cells +
artifact-backed selected-vs-total) and AC-8 (final writeup — explicitly AFTER AC-4). This round is the AC-4
evidence-table close-out (Codex required-plan item #1); AC-8 is the next round.

## Mainline Objective (exactly one)
**Close out the AC-4 per-arm evidence table: fill the missing strict serial GSM8K cells and replace the
static selected-vs-total literals with an artifact-backed, fail-closed probe.** Then the table has no blank
serial cells for the core arms and selected-vs-total is provenance-backed + ledger-guarded.

## Target ACs
- **AC-4** (primary; overlaps AC-1 serial cells): per-arm serial+batched GSM8K + evidence-backed
  selected-vs-total for the core DS arms.

## Blocking Side Issues (these ARE the mainline)
- `build_ledger.py` wires `ds={...}` as STATIC literals for the core DS arms (production_ds/ref_faithful/
  ref_cosine) and several serial `.out` labels are unwired (dsa_noradix both, production_ds sparse,
  ref_faithful both, ref_cosine both). The table must be driven from real run `.out` + a selected-vs-total
  artifact, with a fail-closed ledger gate.

## Queued Side Issues (documented, OUT OF SCOPE this round)
- AC-8 final root-cause writeup (next round, after AC-4 passes).
- Plan-term comment cleanup (`AC-*`/`H3` in retained diagnostics); reference selector CUDA-graph safety
  outside loop13; `ac4_garbage_counters.py --arm <non-prod>` default CAPDIR footgun.

## Approach
1. **Serial GSM8K cells** (one TP=8 server at a time, no PYTHONPATH, completion API, teardown to 0 MiB
   after each), via a sequential background orchestration that reuses the guarded harness:
   - `dsa_noradix`: `serve.sh dsa_noradix`; `THREADS=1 REGIME=both run_gsm8k.sh dsa_noradix_serial` (graph).
   - production DS: `serve.sh ds`; `THREADS=1 REGIME=sparse run_gsm8k.sh ds_serial` (dense_serial already
     committed at 0.655; graph).
   - `ref_faithful`: `serve.sh ref_faithful`; `THREADS=1 REGIME=both run_gsm8k.sh ref_faithful_serial`
     (eager — slow).
   - `ref_cosine`: `serve.sh ref_cosine`; `THREADS=1 REGIME=both run_gsm8k.sh ref_cosine_serial` (eager).
2. **Selected-vs-total artifact**: a fail-closed probe `ac4_selected_vs_total_probe.py --arm NAME` that (the
   arm's server up) sends a dense (<top_k) and a sparse (>top_k) `/generate`, reads
   `meta_info["double_sparsity"]` (`selected_tokens`/`total_tokens`/`dense_fallback`), and records
   `evidence/ac4_selected_vs_total.json[arm][regime]`. Asserts dense `selected==total`, sparse
   `selected<total`, `dense_fallback==0` per arm; run during each DS arm's uptime (production_ds,
   ref_faithful, ref_cosine). Atomic write.
3. **Ledger**: wire the new serial `.out` labels into the ARMS table; replace the static `ds={...}` literals
   for the core DS arms with values loaded from `ac4_selected_vs_total.json`; add
   `validate_selected_vs_total_artifact()` (dense selected==total, sparse selected<total, dense_fallback==0,
   exact arm labels) and a guard that REJECTS a blank serial cell for the AC-4 core arms once filled.
   Regenerate `evidence_table.md` / per-arm JSONs / `run_meta.json` / `findings.md`.

## Concrete Success Criteria
1. Serial cells filled in `evidence_table.md` for dsa_noradix (dense+sparse), production_ds (sparse),
   ref_faithful (dense+sparse), ref_cosine (dense+sparse) — sourced from real `.out` files. One TP=8 server
   at a time, each torn down to 0 MiB; no PYTHONPATH; no `.out`-raw-prompt leakage.
2. `evidence/ac4_selected_vs_total.json` records per-arm/per-regime `selected_tokens`/`total_tokens`/
   `dense_fallback` for production_ds, ref_faithful, ref_cosine, fail-closed on the dense==total /
   sparse<total / dense_fallback==0 invariants; `build_ledger.py` drives the table's selected-vs-total from
   it and `validate_selected_vs_total_artifact()` aborts on a missing/invalid artifact (verified).
2b. The ledger fails closed on a blank serial cell for any AC-4 core arm once the artifact requires it.
3. Tests pass; provenance consistent. Commit; round-21-summary with BitLesson Delta + Goal Tracker Update
   Request. No selection/adapter FIX. No exit by lying / editing loop state / cancel-rlcr-loop.
