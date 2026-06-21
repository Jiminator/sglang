# Round 15 Summary

Mainline: **AC-4 length-cap garbage counters on the production SCORED DS arm** — the complement to the
R14 forced-all control. Diagnostic/guarded instrumentation only (no selection/adapter fix; production
byte-identical when off).

## Work Completed
Reused the **verified R14 `forced_all_assert` capture** but WITHOUT `forced_all_dense_control`, so it dumps
the **REAL production scored top-k** (post-adapter physical slots + `_ds_slot_written` bits + KV capacity,
per `(rank, req, layer, step)`) instead of the forced sweep.

1. **`serve.sh ds_garbage`** — production `ds` config (`scorer_norm=off`, `head_agg=max`, top_k 2048) +
   `"forced_all_assert": true`, NO forced-all override, eager (`--disable-cuda-graph`). One TP=8 server;
   captured a small dense (5-shot) AND sparse (24-shot) run into `.sglang_ds_garbage`, torn down to 0 MiB.
2. **`ac4_garbage_counters.py`** — garbage-only reducer for the SCORED selection: NO sweep / `req_to_token`
   equality (the selection is scored, not `[0..seq_len-1]`); auto-splits **dense (seq_len≤top_k)** vs
   **sparse (seq_len>top_k)**; per regime counts duplicate / live-lane `-1` / out-of-range (vs the true KV
   capacity) / adapter-error, and unwritten split into **current-slot (H3 marker, if selected)** vs
   **non-current (real garbage)**. Fail-closed on zero rows / missing fields / any real garbage (verified
   exit 2 on an empty dir).
3. **`ac2_1` hardened** — `h3_finding` prose is now conditional on a new `h3_marker_on_all_rows` field, so a
   future rerun that does NOT have `current_unwritten == dense_rows` cannot inherit the "every row" claim
   (Codex R14 reuse note; cheap, done this round).
4. **Ledger + findings wired** — `build_ledger.py` attaches `garbage_counters_artifact =
   evidence/ac4_garbage_counters.json` to the production_ds arm; footer + `NOT_INSTRUMENTED` updated (only
   the reference arms remain); `findings.md` records the AC-4 scored-arm result.

## Result (CLEAN — H3 from the selection side)
On **41808 dense + 37440 sparse** captured scored rows, the production selection has **zero real garbage**
in BOTH regimes: 0 duplicate, 0 live-lane `-1`, 0 out-of-range, 0 adapter-error, **0 non-current
unwritten**. And **`current_slot_unwritten = 0`** — the production scored selection does **not** include
the current decode slot (it is masked/excluded by the `_slot_written` invalidation).

This is the **complement** to AC-2.1: the forced-all control *forces* the current slot in and finds it
marked unwritten on 61776/61776 rows; the production scored path simply *excludes* it (never selects it).
Both views pin the dense regression to the **current-slot invalidation (H3)** — the adapter + selected-index
path itself is clean (no duplicate/stale/out-of-range slots) in the real production selection, dense and
sparse alike.

## Files Changed (committed `e0f28d547`)
- `development/loop13/serve.sh` (+`ds_garbage` mode), `development/loop13/ac4_garbage_counters.py` (new),
  `development/loop13/evidence/ac4_garbage_counters.json` (new), `development/loop13/ac2_1_forced_all_assertions.py`
  (conditional H3 prose), `development/loop13/build_ledger.py` (artifact wiring + footer/NOT_INSTRUMENTED),
  `development/loop13/evidence/findings.md`, `evidence/forced_all_assertions.json` (regenerated: +`h3_marker_on_all_rows`),
  `evidence/evidence_table.md`, `evidence/meta/*` (regenerated), `.gitignore` (+`.sglang_ds_garbage/`).

## Validation
- CPU reducers/tests that consume committed evidence — `ac4_garbage_counters` (CLEAN), `ac2_1_forced_all_assertions`
  (61776/61776 PASS, `h3_marker_on_all_rows=true`), `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `test_reference_selectors` (5/5) — **all exit 0**.
- `verify_ac2_3.py` is a GPU-capture re-derivation (reads the ephemeral `.sglang_ds_scorecap` sparse dir,
  which is gitignored and not on disk this turn); its committed artifact `ac2_3_radix_width_equivalence.json`
  (4992/4992 pruning rows) is the canonical evidence and was left intact — NOT re-run from a stale dir.
- `ac4_garbage_counters.py` fail-closes (exit 2) on an empty dir; real (non-current) garbage is a number (0),
  not prose.
- `build_ledger.py` → provenance consistent; production_ds arm carries `garbage_counters_artifact`.
- One TP=8 server at a time, torn down to 0 MiB. No `.pt`/`.humanize` committed. No selection/adapter **fix**.

## Remaining Items (for AC-8 COMPLETE)
- **AC-3.1** captured-row materialized fp32 `K_label` selected-index equality (extend latent-VALUE capture).
- **AC-2.4** recall-oracle@2048 (NIAH-only).
- **AC-4** garbage counters on the REFERENCE arms (`ref_faithful`/`ref_cosine`); remaining serial cells
  (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial); selected-vs-total gaps.
- **AC-8** final root-cause writeup.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-forced-include-vs-scored-exclude-complementary-h3
- Notes: One instrumentation hook gives TWO complementary views of the same downstream bug depending on
  whether you force-include the suspect slot or read the real selection. Forced-all (force the current slot
  IN) shows it `_ds_slot_written`-False on every row; the production scored path simply EXCLUDES it
  (`current_slot_unwritten=0`). Same H3 cause (the `_slot_written[layer,out_cache_loc]=False`
  invalidation), seen from both sides — and the scored-exclude view also proves the real production
  selection has zero adapter/stale garbage in BOTH regimes (dense AND sparse), which the forced-all dense
  control alone could not show. Reuse a verified capture before building a new one; just drop the override.

## Goal Tracker Update Request

### Requested Changes:
- Mark **AC-4 production_ds scored garbage counters DONE** (task9 advanced) — `ac4_garbage_counters.json`:
  41808 dense + 37440 sparse scored rows, real garbage 0 both regimes, `current_slot_unwritten=0` (H3 from
  the selection side); wired as `garbage_counters_artifact` on the production_ds ledger arm.
- Narrow the standing AC-4 garbage-counter blocker to the **REFERENCE arms only** (production_ds + the
  forced-all control are now both instrumented).
- Note **AC-2.1 reducer hardened** (conditional `h3_finding` + `h3_marker_on_all_rows`).
- Plan Evolution Round-15 row added; Plan Version → 16.

### Justification:
The R14-review left AC-4 scored-arm garbage counters as an explicit open item. This round measured them on
the real production workload with the already-verified R14 instrumentation (no new production code — the
only change is a config-borne default-off capture and offline reducers). The result strengthens the H3
verdict from both sides: forced-include shows the current slot unwritten on every row, scored-exclude shows
it is never selected, and the scored path has zero adapter/stale garbage across dense AND sparse. Remaining
close-out items (AC-3.1, AC-2.4, AC-4 reference-arm garbage / serial / selected-vs-total, AC-8) are the next
sequence toward COMPLETE.
