# Round 8 Summary

Mainline: **advance original-plan close-out** — settle the two offline-computable evidence items
(AC-2.2, AC-4 sample IDs) on a corrected, self-consistent ledger. Round 7 was ADVANCED; Codex flagged a
real metadata bug (#1 P1) + the close-out backlog. Entirely CPU this round (no server launched).

## Work Completed
- **Blocking fix — `ds_reduce_fp32` ledger metadata** (Codex #1 P1). When the arm switched to graph mode
  in R7, `build_ledger.py` still hard-coded `--disable-cuda-graph`, so the arm JSON recorded
  `cuda_graph: "off"` — contradicting the actual graph-enabled run (`serve_ds_reduce_fp32.log`:
  `disable_cuda_graph=False`, decode `cuda graph: True`) and making the single-variable arm look
  multi-variable. Fixed `extra` to match `serve.sh` (`--disable-radix-cache --enable-double-sparsity`);
  regenerated arm JSON/table/run_meta (`cuda_graph: "on (piecewise off)"`); recorded
  `ds_config={score_reduce_dtype: fp32, ...}`; added a **fail-closed consistency check** (server_args must
  not contain `--disable-cuda-graph`, cuda_graph graph-enabled, ds_config has `score_reduce_dtype=fp32`)
  — verified it asserts when the extra is re-broken.
- **AC-2.2 SETTLED — TP head-aggregation micro-test** (`ac2_2_head_agg.py` → `head_agg_tp_semantics.json`),
  offline from the validated per-rank `pre_reduce_scores` (702 8-rank groups; `sum(pre)==post` **702/702**,
  resolving the long-standing PRELIMINARY blocker). Served cross-TP **SUM** (= `reduce_token_scores`) vs
  **global-MAX** over heads: median Jaccard **0.679** (78/702 identical) → the served `head_agg="max"` +
  SUM is **not** a global max over heads (the plan's negative test). SUM vs **global-MEAN**: Jaccard 1.0
  (scale-only). **Exoneration:** `build_absorbed_projection` uses `num_local_heads` and the reference path
  does NO cross-TP reduce (verified — no `reduce_token_scores`/all-reduce in `_reference_selector_topk`),
  so production (SUM) and the reference (per-rank-local) use *different* head aggregation — yet cosine
  recovers under both and raw-dot collapses under both (production-SUM 0.000 ≈ reference-local 0.013). So
  cross-TP head aggregation is **not** the accuracy driver (consistent with AC-6).
- **AC-4 sample IDs/order** (`ac4_sample_ids.py` → `gsm8k_sample_ids.json`). The stock
  `simple_eval_gsm8k` loader is deterministic (no seed/shuffle) → re-derived the exact ordered eval
  slices (dense `lines[5:205]`, sparse `lines[24:174]`) with per-example `(line, question sha16)` +
  `test.jsonl` sha256; all arms share the identical set. Wired into the ledger
  (`gsm8k.sample_ids_artifact`); removed sample IDs from `fields_not_instrumented` (the prior "seed-42
  slice" note was wrong — selection is deterministic). Only garbage counters remain not-instrumented.

## Files Changed (committed `752752f6d`)
- NEW: `development/loop13/ac2_2_head_agg.py`, `ac4_sample_ids.py`,
  `evidence/head_agg_tp_semantics.json`, `evidence/gsm8k_sample_ids.json`.
- MODIFIED: `build_ledger.py` (ds_reduce_fp32 extra + ds_config + consistency check + sample_ids wiring +
  footer text), `evidence/findings.md` (AC-2.2 section), `evidence/cheap_controls.json` (AC-2.2 status),
  `evidence/evidence_table.md`, `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json`.

## Validation
- `build_ledger.py` → provenance consistent (blob `0d914406af8b`); ds_reduce_fp32 arm `cuda_graph: "on
  (piecewise off)"`, no `--disable-cuda-graph`, `ds_config` records `score_reduce_dtype=fp32`; the
  consistency check **asserts (exit 1)** when the extra is re-broken.
- `ac2_2_head_agg.py` → 702 groups, `sum(pre)==post` 702/702, SUM-vs-MAX median Jaccard 0.679, exit 0.
- `ac4_sample_ids.py` → test.jsonl sha, dense [5:205]/200, sparse [24:174]/150, exit 0.
- Full suite re-run: `test_reference_selectors` (5/5), `verify_ac2_3`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac6_bisection_matrix` — all exit 0.
- No `.pt`/`.humanize` committed. No selection/adapter fix landed. CPU-only (GPUs idle throughout).

## Remaining Items (for AC-8 COMPLETE)
- **AC-2.1** forced-all physical-slot assertions (`forced_all_assertions.json`) — needs guarded
  instrumentation of the `logical_to_physical`→`transform_index_page_table_decode` adapter (dump physical
  slots + `req_to_token`, assert no dup/`-1`/unwritten/out-of-range, adapter errors 0). GPU + capture.
- **AC-2.4** recall-oracle@2048 — NIAH-only instrument; run the NIAH dense/sparse oracle as corroboration. GPU.
- **AC-3.1** captured-row materialized fp32 `K_label` selected-index equality — needs the resident latent
  captured alongside scores, then offline materialize + compare top-2048. GPU capture + offline.
- **AC-4** length-cap garbage counters — same adapter instrumentation as AC-2.1.
- **AC-8** final root-cause writeup — after the above.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-ledger-metadata-tracks-serve-mode
- Notes: Added a lesson that a ledger/evidence generator which HAND-CODES per-arm server args / cuda_graph
  state separately from the serve script will drift when the serve mode changes (here eager→graph), silently
  contradicting the actual run and making a single-variable arm look multi-variable. Fix: match the
  generator to serve.sh, record the actual config knobs, and add a fail-closed check tying the recorded
  args/graph-state to the serve mode — verified to actually fire when re-broken.

## Goal Tracker Update Request

### Requested Changes:
- Close **R7-review blocking: ds_reduce_fp32 wrong CUDA-graph metadata** — fixed + guarded (cuda_graph
  graph-enabled, ds_config records fp32, fail-closed check verified).
- Mark **AC-2.2 (task3) DONE** — `head_agg_tp_semantics.json`: served SUM ≠ global-max (Jaccard 0.679),
  exonerated as the bottleneck; the PRELIMINARY `sum==post` blocker resolved (702/702).
- Mark **AC-4 sample IDs/order done** (task9 partial→advanced) — `gsm8k_sample_ids.json` wired into the
  ledger; only garbage counters remain not-instrumented.
- Plan Evolution Round-8 row added.

### Justification:
The metadata bug was a genuine AC-1/AC-4/AC-6 integrity defect (now fixed and guarded against recurrence).
AC-2.2 was settleable offline once the captures were validated (`sum(pre)==post` 702/702), and the
exoneration uses verified code facts (`num_local_heads`, no reference cross-TP reduce) + the measured GSM8K
numbers. AC-4 sample IDs are deterministic and thus exactly re-derivable. The remaining close-out items
(AC-2.1/2.4/3.1/4-garbage/8) each require GPU capture or adapter instrumentation and are the explicit
next sequence toward AC-8 COMPLETE.
