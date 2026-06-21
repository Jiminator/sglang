# Round 9 Contract

Round 8 was ADVANCED. Codex's Round-9 priority is explicit: **reconcile the evidence generators first
(CPU-only) before any new GPU capture.** The new AC-2.2 / sample-ID artifacts are valid, but the
generated package contradicts them, the head-agg exoneration overclaims, and per-arm DS launch configs
are incomplete.

## Mainline Objective (exactly one)
**Make the generated evidence package internally consistent and complete:** (a) every DS arm records its
full `--double-sparsity-config` launch config in the ledger (AC-1/AC-4); (b) AC-2.2 reads SETTLED
everywhere with a corrected, non-overclaimed exoneration; with fail-closed guards so the contradictions
cannot recur.

## Target ACs
- **AC-1 / AC-4** (primary): full per-arm DS launch config / server args in the ledger.
- **AC-2.2 / AC-6** (secondary): consistent settled wording across all generated surfaces.

## Blocking Side Issues (truly block a trustworthy package / final writeup)
- **Per-arm `server_args` omit `--double-sparsity-config`.** `serve.sh` launches DS modes with
  `--double-sparsity-config "$DS_CONFIG"`, but `build_ledger.py` records only abbreviated extras, and the
  arm JSONs have no structured `ds_config` (production_ds) or only a 3-field one (ds_reduce_fp32). AC-1
  requires full server args; AC-4 is the per-arm table. Fix: construct the canonical per-arm DS config
  matching serve.sh for EVERY DS mode (top_k, page_size, channel_mask_path, device_buffer_size,
  scorer_norm, head_agg, anchor_mode, anchor_budget, lifted-budget flags, selector_impl,
  reference_include_current, capture flags, score_reduce_dtype where applicable); include
  `--double-sparsity-config <json>` in `server_args` AND a complete structured `ds_config`. Add a
  fail-closed assertion: any arm with `--enable-double-sparsity` must have `--double-sparsity-config` in
  `server_args` and a complete `ds_config`.
- **AC-2.2 generated contradictions.** `cheap_controls.json.summary` still carries the old 78-row +
  `AC_2_2_served_sum_matches_post_reduce_all=false` + "trust only if..." note (a peer verdict next to the
  SETTLED `_status`); `ac6_bisection_matrix.json` leg 1 still says "still PRELIMINARY". Reconcile both to
  the settled `head_agg_tp_semantics.json` (move the old fields to a superseded key / reference the
  artifact). Add a fail-closed check: once `head_agg_tp_semantics.json` has
  `capture_validation_sum_pre_eq_post == "702/702"`, no generated surface may contain `PRELIMINARY` or
  `served_sum_matches_post_reduce_all=false`.
- **AC-2.2 exoneration overclaim.** `findings.md` (+ `head_agg_tp_semantics.json` + tracker) say "cosine
  recovers under both aggregations" — but only RAW-DOT was measured under both (production cross-TP-SUM
  0.000 vs reference per-rank-local 0.013); cosine was measured ONLY on the reference (local) path. Rewrite
  to the MEASURED claim: raw-dot collapses under both aggregations ⇒ the cross-TP aggregation difference is
  bounded ≤~1.3 pp (second-order, like fp8); the accuracy driver is scorer + current-slot. Do NOT claim
  cosine recovers under production-SUM (unmeasured; no production cosine kernel — AC-6 leg 6 blocker).
  Refine AC-6 matrix leg 1: within-rank head_agg=max is matched, but cross-TP aggregation (SUM vs
  reference-local) is a real second-order difference (≤1.3 pp), not simply "not-a-differing-variable".

## Queued Side Issues (documented, OUT OF SCOPE this round — need GPU/instrumentation)
- AC-2.1 forced-all physical-slot assertions (`forced_all_assertions.json`); AC-4 length-cap garbage
  counters — both need guarded `logical_to_physical` adapter instrumentation + a GPU run.
- AC-3.1 captured-row materialized fp32 `K_label` equality — needs the resident latent captured + offline compute.
- AC-2.4 recall-oracle@2048 (NIAH-only) — GPU run.
- AC-8 final writeup — after the above.
- Plan-term comment cleanup; reference-mode fail-closed.

## Concrete Success Criteria
1. Every DS arm JSON has `server_args` containing `--double-sparsity-config <json>` AND a complete
   `ds_config` matching serve.sh; `build_ledger.py` asserts it (verified the assert fires when a config key
   is dropped). Non-DS arms (dsa, dsa_noradix) unaffected. Table/run_meta regenerated; provenance consistent.
2. `cheap_controls.json.summary` carries the 702/702 + SUM-vs-MAX (0.679) / SUM-vs-MEAN (1.0) AC-2.2
   result; the old 78-row fields moved under a superseded key. `ac6_bisection_matrix.json` leg 1
   references the settled artifact (no "PRELIMINARY"). A fail-closed check forbids `PRELIMINARY` /
   `served_sum_matches_post_reduce_all=false` once the artifact is 702/702 (verified it fires).
3. The exoneration wording in `head_agg_tp_semantics.json`, `findings.md`, `cheap_controls.json._status`,
   and the tracker is corrected to the measured claim (raw-dot collapses under both; cross-TP aggregation
   ≤~1.3 pp second-order; cosine-under-SUM not claimed).
4. Tests pass; provenance consistent. Commit; round-9-summary with BitLesson Delta + Goal Tracker Update
   Request. CPU-only this round; no selection/adapter fix; no exit by lying / editing loop state /
   cancel-rlcr-loop.
