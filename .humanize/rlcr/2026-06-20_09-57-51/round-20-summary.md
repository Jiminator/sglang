# Round 20 Summary

Mainline: **AC-3.1 CAPTURED decode-row materialized fp32 `K_label` selected-index equality** — Codex named
the committed `ac3_1_materialized_k.json` a SYNTHETIC CPU proof; the plan wants the equality on CAPTURED
rows. Diagnostic/guarded instrumentation only; no selection/adapter fix.

## Approach (reuse the proven math on real data)
The identity is already implemented + proven by the passing unit test
`test_materialized_raw_equals_absorbed_raw`: the served `reference_rawdot_select` →
`absorbed_latent_score_logical_fp8` (absorbed raw-dot) and `absorbed_latent_cosine_logical_fp8(normalize=
False)` (the raw dot on the MATERIALIZED per-head `K_label` signature) take the IDENTICAL arg set and select
the same top-k. So the captured-row proof = dump those inputs at a served decode row and re-run BOTH
functions offline on the SAME captured inputs — zero new math, just real data.

## Work Completed
1. **Config flag** — `materialized_k_capture: bool = False` wired in all 4 config places.
2. **Capture module** (`materialized_k_capture.py`, new) — guarded, default-off, eager-only. Dumps a
   SELF-CONTAINED minimal **bs=1 reconstruction** per (rank,layer,regime,step), regime-aware-capped: the
   per-request query + the **GATHERED live** fp8 latent/scales/`_ds_slot_written` (only the request's live
   slots, NOT the whole KV pool — so the reducer needs no pool) + the per-layer `w_sel`/channel mask.
3. **Hook** in `_reference_selector_topk` (deepseek_v2.py), inside the `not is_current_stream_capturing()`
   guard, before `reference_rawdot_select`, copy-only — production byte-identical when off (the 5 reference
   unit tests still pass).
4. **`serve.sh ref_faithful_matk`** — ref_faithful config + `materialized_k_capture`, eager. One TP=8
   server, small dense+sparse capture (192 rows = 96 dense + 96 sparse), torn down to 0 MiB.
5. **CPU reducer** (`ac3_1_materialized_k_equality.py`, new) — rebuilds each captured row
   (`req_to_token=[[0..seq_len-1]]` over the captured live latent so the functions gather exactly the
   captured slots) and runs `absorbed_latent_score_logical_fp8` (raw) vs
   `absorbed_latent_cosine_logical_fp8(normalize=False)` (materialized), `select_topk_sequence_order(@2048)`,
   asserts per-row selected-index SET equality. Fail-closed: requires BOTH regimes, writes the canonical
   artifact ONLY via atomic `.tmp`→`os.replace` when every row matches.
6. **Ledger** — `build_ledger.validate_materialized_k_artifact()` independently asserts both-regimes /
   all-rows-equal / source basename / index_topk before recording `run_meta.materialized_k_captured_row_
   equality`. The synthetic proof is marked SUPERSEDED. `findings.md` records the captured-row result.

## Result (`evidence/ac3_1_materialized_k_selected_index_equality.json`)
On **96 dense + 96 sparse REAL captured decode rows**, the absorbed raw-dot reference and the materialized
fp32 `K_label` score select the **IDENTICAL top-2048 indices** — 96/96 in both regimes, max abs score diff
**2e-9 dense / 7e-9 sparse** (fp32 round-off). So the served raw-dot ceiling **is** the materialized fp32
`K_label` ceiling on real data (the captured-row form of the exact-algebra identity).

## Verification (the guards fire)
- Reducer: an empty / single-regime capture dir → exit 2, canonical artifact UNTOUCHED (both-regimes +
  atomic write).
- Ledger: an injected `all_selected_index_equal=false` / partial (`eq<rows`) / missing-regime / wrong-source
  artifact each makes `build_ledger.py` ABORT; restored → provenance consistent.

## Files Changed (committed `e67f1b5f3`)
- `python/.../double_sparsity/config.py` (flag), `python/.../double_sparsity/materialized_k_capture.py`
  (new), `python/.../models/deepseek_v2.py` (guarded hook), `development/loop13/serve.sh`
  (ref_faithful_matk), `development/loop13/ac3_1_materialized_k_equality.py` (new),
  `development/loop13/build_ledger.py` (validate + wiring), `evidence/ac3_1_materialized_k_selected_index_equality.json`
  (new), `evidence/ac3_1_materialized_k.json` (superseded note), `evidence/findings.md`,
  `evidence/evidence_table.md` + `evidence/meta/*` (regenerated), `.gitignore`.

## Validation
- CPU suite, explicit args: `ac3_1_materialized_k_equality` (EQUAL), `ac4_garbage_counters`,
  `ac2_1_forced_all_assertions`, `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `verify_ac2_3 .sglang_ds_scorecap_sparse`
  (committed AC-2.3 artifact unchanged), `test_reference_selectors` (5/5 — production byte-identical when
  the capture flag is off) — **all exit 0**.
- `py_compile` clean; `build_ledger.py` → provenance consistent. No `.pt`/`.humanize` raw artifacts
  committed. One TP=8 server, eager, torn down to 0 MiB. No selection/adapter **fix**.

## Remaining Items (for AC-8 COMPLETE)
- **AC-4** serial cells (production DS sparse serial + dsa_noradix serial graph-mode; ref_faithful/ref_cosine
  serial eager-slow) + selected-vs-total verification (values are populated; make them capture-backed).
- **AC-8** final root-cause writeup (after AC-4).

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-captured-row-proof-via-minimal-reconstruction-of-the-served-fn
- Notes: To turn a SYNTHETIC algebra/unit-test proof into the CAPTURED-row evidence a reviewer demands,
  don't reimplement the math offline — capture the EXACT inputs of the served function and re-call the SAME
  function on real data. Two keys: (1) capture a SELF-CONTAINED MINIMAL reconstruction, not the whole hot
  state — the served scorer gathers `latent[req_to_token[req,:seq]]` from the full KV pool, so capturing the
  pool is intractable; instead capture the GATHERED live slots ([seq_len,…]) + set `req_to_token=[[0..seq_len
  -1]]` offline so the same function gathers exactly the captured slots (verified: scores matched to fp32
  round-off ~1e-9, top-k bit-identical). Read the function's gather/mask indexing FIRST so the minimal
  reconstruction is faithful (here `written[physical_slots]` and the per-block scale layout). (2) the
  capture is a guarded config-borne default-off flag (all 4 config places) hooked inside the existing
  `not is_current_stream_capturing()` guard, copy-only — prove byte-identical-when-off by re-running the
  existing unit tests. The reducer + ledger follow the now-standard fail-closed contract (both regimes,
  atomic write only when every row passes, independent ledger gate; verified on negatives). Builds on
  [[forced-all-downstream-isolation-control]] and [[niah-recall-oracle-fail-closed-span-self-verify]].

## Goal Tracker Update Request

### Requested Changes (already applied to the mutable section):
- Plan Version → 23 (Round 20); added a 19-review row + the Round-20 evolution row.
- task7 → done (R20 captured-row): `ac3_1_materialized_k_selected_index_equality.json`, 96/96 dense + 96/96
  sparse identical top-2048, fail-closed reducer + ledger gate; synthetic proof superseded.

### Justification:
Codex named AC-3.1 a remaining close-out item and the synthetic proof insufficient. The captured-row
artifact proves the absorbed raw-dot reference ceiling IS the materialized fp32 `K_label` ceiling on REAL
served decode rows in BOTH regimes, by reusing the proven served functions on captured inputs (zero math
risk) under the standard fail-closed producer + independent ledger gate. Remaining close-out (AC-4
serial/selected-vs-total, then AC-8) is the active sequence toward COMPLETE.
