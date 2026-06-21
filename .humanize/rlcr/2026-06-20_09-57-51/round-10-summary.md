# Round 10 Summary

Mainline: **finish the evidence-package consistency/completeness** — Codex's three remaining CPU-only
gaps (the explicit "fix these quickly, then move to GPU/instrumentation"). All three closed + guarded.
CPU-only round; no server launched.

## Work Completed
- **One head-aggregation classification across the package** (AC-6/AC-8). R9 reclassified matrix leg 1
  to MEASURED (cross-TP second-order), but `build_ledger.py`'s generated table footer and `findings.md`
  still said `head_agg NOT-a-differing-variable`. Updated both to the matrix classification (within-rank
  `head_agg="max"` matched; cross-TP SUM vs reference-per-rank-local is a **measured second-order
  ≤1.3 pp** difference); regenerated `evidence_table.md`. Extended the AC-2.2 consistency guard to scan
  `evidence_table.md` + `findings.md` for the stale wording once leg 1 is measured — **verified it fires**.
- **Effective per-arm DS config** (AC-1/AC-4). Added `DS_DEFAULTS` (the full resolved
  `DoubleSparsityConfig`: `selector_width_buckets=[5120]`, `selector_width_overflow_policy=full_fallback`,
  `score_reduce_dtype=bf16`, `selector_impl=production`, capture flags `false`, …). Every DS arm now
  records `effective_ds_config` (all **20** fields: defaults + `channel_mask_path` + launch overrides)
  alongside the literal `ds_config` launch JSON (kept for provenance). Strengthened the assertion to
  require the AC-4-relevant effective keys (`selector_width_buckets`, `score_reduce_dtype`,
  `selector_impl`, `head_agg`, `scorer_norm`) — **verified it fires** when one is dropped. Added a
  "DS effective (impl·width·reduce·scorer·head-agg)" column to the table.
- **cheap_controls stale top-level rows superseded** (AC-2.2). Moved the old top-level analyze_captures
  fields (`n_score_groups:78`, the 78-row `head_agg_test` array carrying `served_sum_matches_post_reduce`,
  and `selected_index_equivalence`/`join`/`n_selection_records`) under the `superseded_round2_head_agg_test`
  / `superseded_round3_join_summary` keys. Top level is now just `top_k` + `summary` + `_status` +
  `superseded_*`. Extended the guard so row-level `served_sum_matches` is allowed **only** under a
  `superseded_*` section — **verified it fires**.

## Files Changed (committed `75158e505`)
- `build_ledger.py` (DS_DEFAULTS + effective_ds_config + effective-key assertion + "DS effective" table
  column + head-agg footer fix), `ac6_bisection_matrix.py` (guard extended to table/findings/cheap_controls
  top-level), `evidence/findings.md` (head-agg line), `evidence/cheap_controls.json` (top-level rows
  superseded), `evidence/evidence_table.md`, `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json`.

## Validation
- Full suite — `test_reference_selectors` (5/5), `verify_ac2_3`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `ac6_bisection_matrix` — all exit 0.
- `build_ledger.py` → provenance consistent (blob `d8d93f638b6b`); effective-config assertion fires when
  an AC-4 key is dropped; ds_config launch + cuda_graph + score_reduce_dtype checks still hold.
- AC-2.2 guard fires on: stale head-agg wording in findings.md, and top-level `served_sum_matches` in
  cheap_controls — verified both, restored clean.
- Repo-wide scan: **no** `head_agg NOT-a-differing-variable` / `still PRELIMINARY` / active
  `served_sum_matches` in any active surface (only under labeled `superseded_*`).
- CPU-only (GPUs idle). No `.pt`/`.humanize` committed. No selection/adapter fix.

## Remaining Items (for AC-8 COMPLETE — all GPU/instrumentation)
- **AC-2.1** forced-all physical-slot assertions (`forced_all_assertions.json`) + **AC-4** length-cap
  garbage counters — guarded `logical_to_physical`→`transform_index_page_table_decode` adapter
  instrumentation + a GPU run (shared physical-slot boundary).
- **AC-3.1** captured-row materialized fp32 `K_label` selected-index equality — resident-latent capture
  + offline materialize/compare at top-2048.
- **AC-2.4** recall-oracle@2048 — NIAH-only instrument; GPU run, labeled corroboration.
- **AC-4** remaining serial cells (DSA-radix serial, production DS sparse serial); selected-vs-total gaps.
- **AC-8** final root-cause writeup — after the above.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-effective-config-not-launch-overrides
- Notes: Added a lesson that a machine-readable ledger must record the EFFECTIVE runtime config (every
  dataclass field resolved = defaults + overrides), not just the literal launch-override JSON — a reader
  can't compare arms without the resolved defaults, and the assertion must require the AC-relevant
  effective keys, not only the launch keys. Keep the literal launch JSON too (it's the reproducible
  command); the effective config is the comparison surface.

## Goal Tracker Update Request

### Requested Changes:
- Close **R8/R9-review blocking: head-agg generated contradictions** — evidence_table.md + findings.md now
  match the matrix; guard extended to those surfaces + cheap_controls top-level (all verified to fire).
- Close **R9-review blocking: ds_config is literal-only, not effective** — `effective_ds_config` (20
  fields) per DS arm + effective-key assertion + table column.
- Mark **AC-2.2 (task3) DONE** (fully reconciled); **AC-1/AC-4 (task1/task9)** advanced (effective config
  done; only serial cells / garbage counters remain); **AC-6 (task11)** matrix evidence internally
  consistent.
- Plan Evolution Round-10 row added.

### Justification:
These were Codex's three explicit Round-9 gaps, all CPU-only consistency/completeness. The package is now
internally non-contradictory (one head-agg classification, guarded across matrix + table + findings +
cheap_controls), and every DS arm records its full effective runtime config for machine-readable AC-4
comparison. The remaining close-out items (AC-2.1/2.4/3.1/4-garbage/serial/8) each require GPU capture or
adapter instrumentation and are the next sequence toward AC-8 COMPLETE — no further broad CPU-only polish
is needed.
