# Round 11 Contract

Round 10 was ADVANCED, but my R10 "DS effective" table column introduced a real provenance bug: it
renders the resolved config DEFAULTS (`selector_width_buckets=[5120]`, `score_reduce_dtype=bf16`) for
the REFERENCE arms, even though the reference selector path BYPASSES those knobs (it runs exact fp32
dequant + full-width `torch.topk`, no bf16 reduce, no radix, no selector-width bucketing —
`deepseek_v2.py:2137-2143`, `config.py:132-136`). So the table falsely shows `ref_faithful`/`ref_cosine`/
`ref_cosine_noinc` as `W[5120] · bf16`. Codex says fix this CPU issue first, before the GPU close-out.

## Mainline Objective (exactly one)
**Make the per-arm AC-4 selector surface reflect ACTUAL selector behavior, not dormant config defaults:**
add a path-aware `ds_selector_behavior` per DS arm, render THAT in `evidence_table.md`, and guard
reference arms from ever showing production width/reduce as "used".

## Target ACs
- **AC-4** (primary): per-arm selector width / score-reduce / top-k / scoring reflect what the selector
  actually does (production vs reference).
- **AC-6 / AC-8** (secondary): the AC-4 surface the writeup relies on is now behavior-accurate.

## Blocking Side Issues (this CPU bug — it IS the mainline)
- **The "DS effective" column conflates resolved config defaults with reference-arm behavior.**
  `effective_ds_config` is fine as config-object provenance, but for `selector_impl` starting
  `reference_` the width/reduce/radix/fp8 knobs are dormant. Emit a separate `ds_selector_behavior` per
  DS arm: production → resolved width/reduce/`blocked-radix`/`fp8 absorbed`/`raw-dot`; reference → `full
  (no bucketing)` / `none (per-rank-local fp32, no cross-TP reduce)` / `exact torch.topk` / `exact fp32
  dequant` / raw-dot|cosine. Render `ds_selector_behavior` in the table. Add a fail-closed assertion: a
  `reference_*` arm's `ds_selector_behavior` must NOT show `[5120]`/`bf16`/`fp32` width-or-reduce as used.

## Queued Side Issues (documented, OUT OF SCOPE — GPU/instrumentation close-out, next rounds)
- AC-2.1 forced-all physical-slot assertions + AC-4 length-cap garbage counters — guarded
  `logical_to_physical`→`transform_index_page_table_decode` adapter instrumentation + GPU run (shared
  physical-slot boundary).
- AC-3.1 captured-row materialized fp32 `K_label` selected-index equality — resident-latent capture +
  offline compute.
- AC-2.4 recall-oracle@2048 (NIAH-only) — GPU run, labeled corroboration.
- AC-4 remaining serial cells (DSA-radix serial, production DS sparse serial); selected-vs-total gaps.
- AC-8 final writeup (after the above).
- Plan-term comment cleanup; reference-mode fail-closed.

## Concrete Success Criteria
1. Every DS arm JSON has `ds_selector_behavior` with path-aware width / score_reduce / topk / scoring /
   scorer / head_agg. `effective_ds_config` is retained (config-object provenance) but the TABLE renders
   `ds_selector_behavior`.
2. `evidence_table.md`: production arms show their resolved width/reduce (e.g. `[5120] · bf16`, or
   `bf16→fp32` for ds_reduce_fp32); reference arms show `full · none(exact-fp32) · exact-topk` — NOT
   `W[5120] · bf16`. The table column header reflects "behavior".
3. A fail-closed assertion in `build_ledger.py` rejects any `reference_*` arm whose `ds_selector_behavior`
   reports production width/reduce as used (verified it fires when re-broken).
4. ROOT_CAUSE.md / findings.md (where they describe the reference arms' selector behavior) are consistent
   — reference arms are exact-fp32/full-width/no-reduce. Tests pass; provenance consistent. Commit;
   round-11-summary with BitLesson Delta + Goal Tracker Update Request. CPU-only this round; no
   selection/adapter fix; no exit by lying / editing loop state / cancel-rlcr-loop.
