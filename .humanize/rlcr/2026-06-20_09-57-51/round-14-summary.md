# Round 14 Summary

Mainline: **repair AC-2.1** — Codex correctly caught that my R13 forced-all artifact over-claimed and
cut three corners. All three fixed; the repaired result is sharper (it measures H3 directly on the
validity bitmap). Guarded instrumentation only (no fix; production byte-identical when off).

## Work Completed — the three R13 defects, all fixed
1. **`_ds_slot_written` now MEASURED** (was: I wrongly claimed "unwritten is subsumed by
   physical==req_to_token equality"). Equality proves the adapter GATHER, not slot VALIDITY. The capture
   hook now resolves the `_ds_slot_written` bitmap at the seam exactly as the production/reference selector
   does (`_get_attn_backend()` → `TboAttnBackend.primary` → `getattr(_, "_ds_slot_written", None)`,
   **fail-closed** if the flag is on but the bitmap is absent) and dumps the validity bit for every live
   physical slot.
2. **Per-step** (was: overwrote each `(rank, req, layer)`). The capture module now stamps a monotonic
   decode-step counter; filenames include `_step{N}`; the reducer keys by `(rank, req, layer, step)` →
   **61776 rows across 20+ decode steps**, no overwrite.
3. **Correct out-of-range bound** (was: `req_to_token.shape[1]` = 202756 max-context). Now checked against
   the **true KV-slot capacity** `_ds_slot_written.shape[1]` = **504704** — a different dimension.

## Result (PASS — and a direct H3 measurement)
On **61776/61776** dense rows: forced sweep `[0..seq_len-1]` 61776/61776; physical ==
`req_to_token[req, 0:seq_len]` 61776/61776; **0** duplicate, **0** live-lane `-1`, **0** out-of-range,
**0** adapter errors, **0 NON-current unwritten**. And **H3 observed directly on the bitmap**: on every
dense row exactly ONE live slot is `_ds_slot_written`-False, and it is exactly the **current decode slot**
(logical `seq_len-1`) — the production `_slot_written[layer, out_cache_loc] = False` invalidation. So the
`logical_to_physical`→`transform_index_page_table_decode` adapter + selected-index path is a provable
clean no-op (exact gather, every non-current slot valid), and the dense regression localizes to the
**current-slot invalidation (H3)** — now measured, not inferred. (Forcing all tokens recovers dense to
~0.950, so the current slot's KV is valid at attention time; the bit is merely stale.) The reducer reports
the current-slot-unwritten (H3 marker, expected) separately from non-current unwritten (real garbage = 0).

## Files Changed (committed `08caeda27`)
- `python/.../double_sparsity/forced_all_assert_capture.py` (rewritten: slot_written bits + per-step
  counter + kv_capacity), `python/.../models/deepseek_v2.py` (hook resolves the bitmap, fail-closed),
  `development/loop13/ac2_1_forced_all_assertions.py` (rewritten reducer: per-step key, unwritten via
  bits, range vs kv_capacity, current-vs-non-current split), `build_ledger.py` (DS_DEFAULTS
  `forced_all_assert: false` + footer/comment reconcile), `evidence/findings.md`,
  `evidence/forced_all_assertions.json`, `evidence/evidence_table.md`, `evidence/meta/*`.

## Validation
- Full suite — `test_reference_selectors` (5/5), `verify_ac2_3`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `ac6_bisection_matrix`,
  `ac2_1_forced_all_assertions` — **all exit 0**.
- Off-path: `forced_all_assert` defaults `False`; reference tests pass; `py_compile` clean → production
  byte-identical when off. New reducer **fail-closes (exit 2)** on the old field-missing captures and on an
  empty dir.
- `build_ledger.py` → provenance consistent; `effective_ds_config` now includes `forced_all_assert`.
- One TP=8 server at a time, torn down to 0 MiB. No `.pt`/`.humanize` committed. No selection/adapter **fix**.

## Remaining Items (for AC-8 COMPLETE)
- **AC-3.1** captured-row materialized fp32 `K_label` selected-index equality (extend latent capture + analyzer).
- **AC-2.4** recall-oracle@2048 (NIAH-only).
- **AC-4** garbage counters on the SCORED arms (enable the now-repaired capture on production_ds/ref_*);
  remaining serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial);
  selected-vs-total gaps.
- **AC-8** final root-cause writeup.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260621-forced-all-downstream-isolation-control
- Notes: Corrected the R13 over-claim (unwritten is NOT subsumed by physical==req_to_token; `_ds_slot_written`
  is a separate validity bitmap — capture it, and separate the expected current-slot marker from real
  garbage) and added the two other R13 corner-cuts Codex caught: capture must be PER-STEP (a decode-step id,
  not overwrite per (rank,req,layer)); a PHYSICAL out-of-range check must use the actual capacity tensor
  (`_ds_slot_written.shape[1]`), not a different dimension (`req_to_token.shape[1]` = max_context).

## Goal Tracker Update Request

### Requested Changes:
- Mark **AC-2.1 (task2) DONE** — repaired: `_ds_slot_written` measured + per-step + true KV range; 61776/61776
  dense rows, 0 real garbage, current-slot-unwritten = the H3 marker observed on the bitmap.
- Close **R13-review blocking: DS_DEFAULTS missing `forced_all_assert`** — added; `effective_ds_config` now
  fully resolved.
- Note **AC-4 garbage counters** for the forced-all control are now real (incl. unwritten, per-step);
  enabling on the scored arms is the remaining AC-4 garbage work.
- Plan Evolution Round-14 row added.

### Justification:
Codex was right on all three points; capturing `_ds_slot_written` turned the control into a direct H3
measurement on the validity bitmap (the current decode slot is unwritten on 61776/61776 rows) rather than
an inference, which strengthens the verdict. The per-step + true-range fixes make the AC-4 garbage counters
trustworthy. Remaining close-out items (AC-3.1, AC-2.4, AC-4 scored-arm garbage/serial/selected-vs-total,
AC-8) are the next sequence toward COMPLETE.
