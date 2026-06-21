# Round 12 Summary

Mainline: **fix the `ds_forced_all` selector-behavior surface** — the last AC-4 behavior-provenance
item (the same class Codex flagged in R10/R11, narrower). CPU-only round; no server launched.

## Work Completed
- **Bug:** `ds_selector_behavior_for()` branched only on `selector_impl`, so `ds_forced_all` (which has
  `forced_all_dense_control=true`) rendered as plain `prod · [5120] · bf16 · blocked/radix`. But at
  runtime `_select_topk_indices` calls `apply_forced_all_dense()` **after** the production selector
  (`deepseek_v2.py:2631`), which **overwrites** the dense scored top-k (rows `seq_len<=top_k`) with the
  logical sweep `[0..seq_len-1]` (`absorbed_latent.py:apply_forced_all_dense`). So the final dense
  selected set is **not** the production top-k.
- **Fix:** branch on `forced_all_dense_control` **before** the generic production case.
  `ds_forced_all` now renders: path `forced-all dense diagnostic (production scoring then dense
  override)`; selector_width `full live dense rows (seq_len<=top_k)`; score_reduce `not used for the
  final dense selected set`; topk `forced [0..seq_len-1] after production scoring`; scoring `production
  pre-override only`; scorer raw-dot pre-override. The table prefix is now **3-way** (`prod` /
  `forced-all` / `ref`).
- **Fail-closed assertion:** any `forced_all_dense_control=true` arm's `ds_selector_behavior.topk` must
  contain `forced` and must not be plain `blocked/radix` — **verified it fires** when re-broken.
- **Confirmed unchanged:** `production_ds` (`prod · [5120] · bf16 · blocked/radix`) and `ds_reduce_fp32`
  (`prod · [5120] · fp32 · blocked/radix`) still render production top-k; reference arms still render
  `ref · full · none · exact torch.topk`. Only `ds_forced_all` changed.

## Files Changed (committed `d11e752b8`)
- `build_ledger.py` (ds_selector_behavior_for forced-all branch + 3-way table prefix + forced-all topk
  assertion), `evidence/evidence_table.md`, `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json`.

## Validation
- `build_ledger.py` → provenance consistent (blob `a0eeed5f4832`); the forced-all topk guard **asserts
  (exit 1)** when ds_forced_all's topk is set to `blocked/radix`; reference-arm + effective-key +
  cuda-graph + DS-config + AC-6 corroboration assertions all still hold.
- Table check: `production_ds`/`ds_reduce_fp32` = `prod · [5120] · …`; `ds_forced_all` =
  `forced-all · full live dense rows · not used · forced [0..seq_len-1] …`; `ref_*` = `ref · full · none
  · exact torch.topk`.
- Full suite — `test_reference_selectors` (5/5), `verify_ac2_3`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `ac6_bisection_matrix` — all exit 0.
- CPU-only (GPUs idle). No `.pt`/`.humanize` committed. No selection/adapter fix.

## Remaining Items (for AC-8 COMPLETE — all GPU/instrumentation)
- **AC-2.1** forced-all **physical-slot** assertions (`forced_all_assertions.json`: equality to
  `req_to_token[req_pool, 0:seq_len]`, no dup/`-1`/unwritten/out-of-range, adapter errors 0) + **AC-4**
  length-cap garbage counters — guarded `logical_to_physical`→`transform_index_page_table_decode` adapter
  instrumentation + a GPU run (shared physical-slot boundary).
- **AC-3.1** captured-row materialized fp32 `K_label` selected-index equality — resident-latent capture +
  offline materialize/compare at top-2048.
- **AC-2.4** recall-oracle@2048 — NIAH-only instrument; GPU run, labeled corroboration.
- **AC-4** remaining serial cells (DSA-radix serial, production DS sparse serial); selected-vs-total gaps.
- **AC-8** final root-cause writeup — after the above.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260621-effective-config-not-launch-overrides
- Notes: Added the R12 second corollary — the dispatch key alone isn't enough; a RUNTIME OVERRIDE applied
  AFTER the selector (here `forced_all_dense_control` → `apply_forced_all_dense()` overwriting the dense
  selected set) also changes the effective behavior. The behavior view must enumerate everything that
  mutates the final selected set (dispatch impl AND post-dispatch overrides), branch on the override flag
  before the generic case, and guard that an override arm can't render the un-overridden top-k as used.

## Goal Tracker Update Request

### Requested Changes:
- Close **R11-review blocking: `ds_selector_behavior` ignores the forced-all override** — fixed via the
  forced-all branch + table 3-way prefix + the forced-all topk guard (verified fires).
- Mark **AC-4 (task9)** advanced: the selector-behavior surface is now correct + guarded for all DS arm
  classes (reference / production / forced-all). Remaining AC-4 gaps are GPU/instrumentation (garbage
  counters, serial cells, selected-vs-total).
- Plan Evolution Round-12 row added.

### Justification:
This was Codex's single new CPU blocker — the forced-all downstream-isolation control was mislabeled as
ordinary production top-k, which would mislead the AC-2.1/AC-4/AC-8 comparison. The behavior surface now
reflects the runtime override and is guarded. The AC-4 selector-behavior surface is complete; all
remaining close-out items (AC-2.1 physical-slot / 2.4 / 3.1 / 4-garbage / serial / 8) require GPU capture
or adapter instrumentation and are the next sequence toward AC-8 COMPLETE.
