# Round 9 Summary

Mainline: **make the generated evidence package internally consistent and complete.** Round 8 was
ADVANCED; Codex flagged that the new artifacts were valid but the generated package contradicted them.
This is the CPU-only reconciliation Codex asked to do **before** any new GPU capture. No server launched.

## Work Completed
- **Full per-arm DS launch config** (AC-1/AC-4). `build_ledger.py` now emits the canonical
  `--double-sparsity-config` for **every** DS arm (a `DS_BASE` + per-arm `DS_OVERRIDES` matching
  `serve.sh` exactly) in `server_args` **and** as a complete structured `ds_config`. Previously only
  abbreviated extras were recorded (production_ds had no config; ds_reduce_fp32 had a 3-field stub).
  Added a **fail-closed assertion**: any `--enable-double-sparsity` arm must have
  `--double-sparsity-config` in `server_args` **and** a `ds_config` with all required keys — verified it
  asserts when the config append is removed. Non-DS arms (dsa/dsa_noradix) carry no DS config.
- **AC-2.2 reconciled across all generated surfaces.**
  - `cheap_controls.json`: the stale `summary` fields (78 rows, `served_sum_matches_post_reduce_all=false`,
    old note) moved under `superseded_round2_head_agg_test`; `summary` now carries the settled
    **702/702** + SUM-vs-MAX **0.679** / SUM-vs-MEAN **1.0** result. `_status` overclaim removed.
  - `ac6_bisection_matrix.py` leg 1: no longer "still PRELIMINARY"; references the settled
    `head_agg_tp_semantics.json`; **reclassified MEASURED** — within-rank `head_agg="max"` is matched,
    but the cross-TP aggregation (production SUM vs reference per-rank-local) is a real **second-order**
    difference (≤~1.3 pp on raw-dot, like fp8/reduce).
- **Exoneration overclaim FIXED** (in `head_agg_tp_semantics.json` via `ac2_2_head_agg.py`, `findings.md`,
  `cheap_controls.json._status`, and the tracker). The prior "cosine recovers under both aggregations"
  was unsupported — only **raw-dot** was measured under both (production cross-TP-SUM 0.000 vs reference
  per-rank-local 0.013 ⇒ ≤1.3 pp); cosine was measured **only** on the reference path, and there is no
  production cosine kernel (AC-6 leg 6 blocker), so **cosine-under-production-SUM is not claimed**. The
  measured statement: raw-dot collapses under both aggregations ⇒ the aggregation is not the driver; the
  scorer (+92.7 pp) and current-slot are.
- **Fail-closed AC-2.2 consistency guard** added in the matrix path: once `head_agg_tp_semantics.json`
  validates all groups (`sum(pre)==post`), no generated surface may contain `PRELIMINARY`, a
  `served_sum_matches_*` field in `summary`, or the cosine-under-both overclaim — verified it fires.

## Files Changed (committed `5d48cbd0d`)
- `build_ledger.py` (DS_BASE/DS_OVERRIDES + `--double-sparsity-config` in server_args + structured
  `ds_config` + DS-config assertion), `ac2_2_head_agg.py` (narrowed exoneration), `ac6_bisection_matrix.py`
  (leg 1 reclassified + AC-2.2 consistency guard), `evidence/findings.md`, `evidence/cheap_controls.json`,
  `evidence/head_agg_tp_semantics.json`, `evidence/ac6_bisection_matrix.json`, `evidence/evidence_table.md`,
  `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json`.

## Validation
- `build_ledger.py` → provenance consistent (blob `b8f4ea711941`); DS arms have
  `--double-sparsity-config` + complete `ds_config`; the DS-config assertion **asserts (exit 1)** when the
  config append is removed; the ds_reduce_fp32 cuda_graph/score_reduce_dtype checks still hold.
- `ac6_bisection_matrix.py` → verdicts measured[1,2,3,7]/retired[4,5]/blocked[6]; AC-2.2 consistency guard
  passes, and **asserts (exit 2)** when `PRELIMINARY` is re-injected.
- Full suite: `test_reference_selectors` (5/5), `verify_ac2_3`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids` — all exit 0.
- Forbidden-string scan: `served_sum_matches`/`PRELIMINARY`/`cosine recovers under both` appear only under
  the labeled `superseded_round2_head_agg_test` key, not in any active verdict surface.
- CPU-only (GPUs idle throughout). No `.pt`/`.humanize` committed. No selection/adapter fix.

## Remaining Items (for AC-8 COMPLETE — all need GPU/instrumentation)
- **AC-2.1** forced-all physical-slot assertions (`forced_all_assertions.json`) + **AC-4** length-cap
  garbage counters — both need guarded `logical_to_physical`→`transform_index_page_table_decode` adapter
  instrumentation + a GPU run (shared physical-slot boundary).
- **AC-3.1** captured-row materialized fp32 `K_label` selected-index equality — needs the resident latent
  captured alongside scores, then offline materialize + compare top-2048.
- **AC-2.4** recall-oracle@2048 — NIAH-only instrument; GPU run, labeled corroboration.
- **AC-8** final root-cause writeup — after the above; must avoid unsupported aggregation claims, no fix.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-reconcile-generated-surfaces
- Notes: Added a lesson that when one evidence artifact becomes authoritative, EVERY generated surface that
  restates the same fact must be reconciled in the same round (move superseded numbers under a labeled
  `superseded_*` key, put the settled result in the authoritative `summary`, update the matrix/findings),
  and a fail-closed cross-surface guard should forbid the stale token once the artifact validates.
  Corollary captured: never claim a result for an UNMEASURED cell — state the measured bound and mark the
  unmeasured cell (cosine-under-production-SUM) as not-claimed.

## Goal Tracker Update Request

### Requested Changes:
- Close **R8-review blocking: head-agg generated contradictions** — reconciled (summary/matrix/findings),
  exoneration narrowed, fail-closed guard added.
- Close **R8-review blocking: incomplete per-arm DS config/server args** — every DS arm now records the
  full `--double-sparsity-config` + structured `ds_config`, guarded.
- Mark **AC-2.2 (task3) DONE** (reconciled + guarded); **AC-1 (task1)** advanced (full DS config; only
  serial cells remain).
- Plan Evolution Round-9 row added.

### Justification:
These were Codex's two explicit Round-9 priorities, both CPU-only ("reconcile evidence generators first…
before new GPU capture"). The package is now internally consistent — one authoritative AC-2.2 verdict, a
fail-closed guard against recurrence, and every DS arm reconstructable from its recorded launch config —
and the head-agg exoneration is narrowed to exactly what was measured. The remaining close-out items
(AC-2.1/2.4/3.1/4-garbage/8) each require GPU capture or adapter instrumentation and are the next sequence
toward AC-8 COMPLETE.
