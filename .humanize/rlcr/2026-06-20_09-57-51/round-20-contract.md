# Round 20 Contract

Round 19 was ADVANCED; Codex VERIFIED AC-2.4 (now in Completed) and AC-2 is MET. Remaining for COMPLETE:
AC-3.1 captured-row materialized-K equality, AC-4 serial cells + selected-vs-total, AC-8 writeup (last).
This round takes AC-3.1 — a discrete, named artifact Codex explicitly flagged (the committed
`ac3_1_materialized_k.json` is a SYNTHETIC CPU proof; the plan wants it on CAPTURED decode rows).

Feasibility (low math risk): the served reference selectors are `reference_rawdot_select` →
`absorbed_latent_score_logical_fp8` (raw) and `reference_cosine_select` →
`absorbed_latent_cosine_logical_fp8(normalize=False)` (the materialized per-head K_label numerator); both
take the IDENTICAL arg set, and their top-k selection equality is exactly what the PASSING unit test
`test_materialized_raw_equals_absorbed_raw` proves. So the captured-row proof = dump those args at the
`reference_rawdot_select` call site for a few decode rows, then OFFLINE (CPU, like the unit test) call BOTH
functions on the SAME captured inputs and assert top-2048 selected-index equality. No new math, no
reconstruction — the reducer is the unit test fed real captured tensors.

## Mainline Objective (exactly one)
**Produce the AC-3.1 CAPTURED decode-row materialized fp32 `K_label` selected-index equality artifact**: a
guarded (default-off) capture of the `reference_rawdot_select` inputs on a served `ref_faithful` run, a
fail-closed CPU reducer that recomputes absorbed raw-dot vs materialized-cosine-numerator top-2048 on the
captured rows and asserts per-row selected-index equality @2048, persisted to
`evidence/ac3_1_materialized_k_selected_index_equality.json` and wired into the ledger with a fail-closed
presence/shape check.

## Target ACs
- **AC-3.1** (primary): the fp32 absorbed raw-dot reference == offline materialized fp32 `K_label` score
  (selected-index equality @2048) on CAPTURED decode rows (not synthetic).

## Blocking Side Issues (these ARE the mainline)
- The existing `latent_capture` dumps only `latent_fp8`/`scales`/`req_to_token` — not the query `w_sel`
  `channel_selection`/`channel_weights`/`written`/`seq_lens` needed to re-call the scoring functions. A new
  guarded capture of the full `reference_rawdot_select` arg set is required (config-borne, default-off,
  eager-only, byte-identical when off).

## Queued Side Issues (documented, OUT OF SCOPE this round)
- AC-4 serial cells (production_ds sparse serial + dsa_noradix serial graph-mode; ref_faithful/ref_cosine
  serial eager-slow) + selected-vs-total verification (the values are already populated; make them
  capture-backed).
- AC-8 final root-cause writeup (after AC-3.1 + AC-4).
- `ac4_garbage_counters.py --arm <non-production>` default CAPDIR reuse footgun; plan-term comment cleanup.

## Approach
1. Add a config-borne `materialized_k_capture: bool = False` flag (all 4 config places). A capture module
   `materialized_k_capture.py` dumps the `reference_rawdot_select` args (queries, latent_fp8, latent_scales,
   w_sel, channel_selection, channel_weights, req_pool_indices, req_to_token, seq_lens, max_seq_len,
   max_top_k, written, head_agg) per (rank, req, layer, step) for the first few rows — host-side `.pt`,
   dir via env, capped count. Hook it in `_reference_selector_topk` inside the existing
   `not is_current_stream_capturing()` guard, BEFORE `reference_rawdot_select`; copy-only, so the served
   selection is unchanged whether or not it fires.
2. `serve.sh ref_faithful_matk` = ref_faithful config + `materialized_k_capture:true`, eager, LAUNCH_CWD as
   needed. One TP=8 server, a SMALL dense (5-shot) + sparse (24-shot) capture (a handful of rows is enough —
   the identity is exact), teardown to 0 MiB.
3. `ac3_1_materialized_k_equality.py`: load each captured arg set, call `absorbed_latent_score_logical_fp8`
   (raw) and `absorbed_latent_cosine_logical_fp8(normalize=False)` (materialized) on CPU, take
   `select_topk_sequence_order(..., top_k=2048)` of each, assert per-row selected-index SET equality; record
   per-regime row counts, max score diff, and the equality verdict. Fail-closed: nonzero exit on zero rows,
   missing fields, or ANY selected-index mismatch. Write
   `evidence/ac3_1_materialized_k_selected_index_equality.json`.
4. `build_ledger.py`: add a fail-closed presence/shape check (dense+sparse rows>0, all_equal true, source
   basename) and reference it in run_meta + footer. Keep the synthetic `ac3_1_materialized_k.json` as the
   algebra note but mark it superseded-by the captured artifact. `.gitignore` the new capture dir.

## Concrete Success Criteria
1. `materialized_k_capture` flag wired in all 4 config places; default off; reference tests still pass +
   `py_compile` clean (production byte-identical when off). `serve.sh ref_faithful_matk` exists.
2. `evidence/ac3_1_materialized_k_selected_index_equality.json` records, on CAPTURED rows (dense+sparse),
   per-row selected-index equality @2048 between absorbed raw-dot and materialized fp32 K_label
   (cosine-numerator), with a non-zero row count and an explicit all_equal verdict. Fail-closed (verified
   exit 2 on empty / a synthetic mismatch).
3. `build_ledger.py` references the artifact and fail-closes if it is absent/empty/not-all-equal; the
   synthetic proof is marked superseded. `findings.md` records the captured-row AC-3.1 result.
4. Tests pass; provenance consistent. One TP=8 server, eager, torn down to 0 MiB. Commit; round-20-summary
   with BitLesson Delta + Goal Tracker Update Request. No selection/adapter FIX (diagnostic/guarded only).
   No exit by lying / editing loop state / cancel-rlcr-loop.
