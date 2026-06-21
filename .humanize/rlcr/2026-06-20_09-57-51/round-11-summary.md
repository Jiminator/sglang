# Round 11 Summary

Mainline: **fix the AC-4 selector-behavior mismatch** my R10 change introduced — the one CPU blocker
Codex asked to fix before the GPU close-out. CPU-only round; no server launched.

## Work Completed
- **Bug (mine, R10):** the "DS effective" table column rendered the resolved config DEFAULTS
  (`selector_width_buckets=[5120]`, `score_reduce_dtype=bf16`) for the REFERENCE arms — but the reference
  selector path (`selector_impl=reference_*`) **bypasses** those knobs: it dequantizes to fp32 and runs
  the exact absorbed channel-dot + full-width `torch.topk`, with no fp8-in-register, no bf16 reduce, no
  radix kernel, no selector-width bucketing (`deepseek_v2.py:2137-2143`, `config.py:132-136`). So the
  table falsely showed `ref_faithful`/`ref_cosine`/`ref_cosine_noinc` as `W[5120] · bf16`.
- **Fix:** split config-object provenance from actual selector behavior.
  - Added `ds_selector_behavior_for(arm)` — path-aware on `selector_impl`. **production** → resolved
    width/reduce + `blocked/radix` top-k + `fp8 absorbed in-register` + `raw-dot (scorer_norm=off)`.
    **reference_*** → `full (no bucketing)` / `none (per-rank-local fp32; no cross-TP reduce)` /
    `exact torch.topk` / `exact fp32 dequant` / raw-dot|cosine, with a note that the production knobs are
    bypassed.
  - Each DS arm JSON now records `ds_selector_behavior` alongside `effective_ds_config` (kept as
    config-object provenance, now noted that a set knob isn't necessarily used).
  - `evidence_table.md` renders `ds_selector_behavior` (path·width·reduce·topk·scorer·head-agg): production
    arms show `[5120] · bf16` (or `fp32` for ds_reduce_fp32) · `blocked/radix`; reference arms show
    `full · none · exact torch.topk`. Column header renamed to "DS selector behavior".
  - **Fail-closed assertion:** a `reference_*` arm's `ds_selector_behavior` must NOT show production
    `5120`/`bf16` as used — **verified it fires** when re-broken. `findings.md`/`ROOT_CAUSE.md` already
    describe the reference arms as exact-fp32/full-width/no-reduce (no contradiction; confirmed by scan).

## Files Changed (committed `482ff8083`)
- `build_ledger.py` (ds_selector_behavior_for + rec field + table renders behavior + header + reference-arm
  bypass assertion), `evidence/evidence_table.md`, `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json`.

## Validation
- `build_ledger.py` → provenance consistent (blob `4f83d15605ca`); reference-arm behavior guard **asserts
  (exit 1)** when a reference arm is made to show `bf16`; effective-config + ds_reduce_fp32 + DS-config
  assertions still hold.
- Table check: production_ds `prod · [5120] · bf16 · blocked/radix · raw-dot · max`; ds_reduce_fp32
  `prod · [5120] · fp32 · …`; ref_* `ref · full (no bucketing) · none … · exact torch.topk · raw-dot|cosine`.
  No reference arm shows production width/reduce.
- Full suite — `test_reference_selectors` (5/5), `verify_ac2_3`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `ac6_bisection_matrix` — all exit 0.
- CPU-only (GPUs idle). No `.pt`/`.humanize` committed. No selection/adapter fix.

## Remaining Items (for AC-8 COMPLETE — all GPU/instrumentation)
- **AC-2.1** forced-all physical-slot assertions (`forced_all_assertions.json`) + **AC-4** length-cap
  garbage counters — guarded `logical_to_physical`→`transform_index_page_table_decode` adapter
  instrumentation + a GPU run (shared physical-slot boundary).
- **AC-3.1** captured-row materialized fp32 `K_label` selected-index equality — resident-latent capture +
  offline materialize/compare at top-2048.
- **AC-2.4** recall-oracle@2048 — NIAH-only instrument; GPU run, labeled corroboration.
- **AC-4** remaining serial cells (DSA-radix serial, production DS sparse serial); selected-vs-total gaps.
- **AC-8** final root-cause writeup — after the above.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260621-effective-config-not-launch-overrides
- Notes: Added the R11 corollary — a resolved config OBJECT is not the runtime BEHAVIOR: a knob being set
  doesn't mean the dispatch path uses it. For an AC-4 comparison/display surface, derive behavior from the
  dispatch key (`selector_impl`), not the config object (production → resolved knobs used; reference →
  full/none/exact-topk, knobs bypassed); display the behavior view, keep the config object as provenance
  only, and guard that a bypassed knob can never be shown as "used".

## Goal Tracker Update Request

### Requested Changes:
- Close **R10-review blocking: "DS effective" reports dormant defaults as reference-arm behavior** —
  fixed via `ds_selector_behavior` + table + the reference-arm bypass guard (verified fires).
- Mark **AC-4 (task9)** advanced: config-object + behavior surfaces both correct; remaining AC-4 gaps are
  GPU/instrumentation (garbage counters, serial cells, selected-vs-total).
- Plan Evolution Round-11 row added.

### Justification:
This was Codex's single new CPU blocker — a provenance bug I introduced in R10 that would mislead the
AC-4/AC-8 production-vs-reference comparison (reference arms shown as if they used the production width
ladder + bf16 reduce). The behavior surface now derives from the actual dispatch path and is guarded, so
the table reflects what each selector truly does. All remaining close-out items (AC-2.1/2.4/3.1/4-garbage/
serial/8) require GPU capture or adapter instrumentation and are the next sequence toward AC-8 COMPLETE.
