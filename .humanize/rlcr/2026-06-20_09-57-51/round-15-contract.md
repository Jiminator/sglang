# Round 15 Contract

Round 14 was ADVANCED and Codex VERIFIED AC-2.1 (added to Completed). Remaining work is the
original-plan GPU/instrumentation close-out. This round takes the next decisive, low-risk item that
reuses the now-verified R14 instrumentation: **AC-4 length-cap garbage counters on the SCORED arms**
(the named AC-4 per-arm "invalid/unwritten/duplicate/out-of-range slot" rate), measured on the actual
production scored top-k — distinct from the forced-all sweep AC-2.1 already covered.

## Mainline Objective (exactly one)
**Produce AC-4 length-cap garbage counters for the production scored DS arm (dense + sparse):** enable
the verified `forced_all_assert` capture WITHOUT `forced_all_dense_control` (so it captures the real
scored selection's physical slots + `_ds_slot_written` bits), run eager, and reduce to a fail-closed
per-regime garbage-counter artifact wired into the ledger.

## Target ACs
- **AC-4** (primary): per-arm length-cap garbage-rate (duplicate / live-`-1` / unwritten / out-of-range
  physical slots + adapter error_count) for production_ds, dense AND sparse.

## Blocking Side Issues (the capture/reducer for the scored arm — it IS the mainline)
- The R14 capture/reducer is forced-all-specific (asserts physical==`req_to_token` sweep). The scored
  selection is NOT the sweep, so a separate garbage-only reducer is needed (no sweep / req_to_token
  equality; only the garbage counters on the selected physical slots, with current-slot-unwritten split
  from non-current). The hook already fires for scored selection when `forced_all_assert` is on (verified:
  it is gated on the flag, not on `forced_all_dense_control`).

## Queued Side Issues (documented, OUT OF SCOPE this round)
- **Harden the AC-2.1 reducer** (Codex R14 reuse note): make the `h3_finding` prose conditional / fail if
  a future rerun does not have `current_unwritten == dense_rows`. Small; do it this round if cheap.
- AC-3.1 captured-row materialized fp32 `K_label` selected-index equality (needs latent-VALUE capture).
- AC-2.4 recall-oracle@2048 (NIAH-only; existing oracle sink + loop7 NIAH drivers).
- AC-4 garbage counters on the REFERENCE arms; remaining serial cells (DSA-radix serial, production DS
  sparse serial, ref_faithful/ref_cosine serial); selected-vs-total gaps.
- AC-8 final writeup.
- Plan-term comment cleanup; serve.sh help text.

## Approach
- Add `serve.sh ds_garbage` = the production `ds` config + `"forced_all_assert": true` (NO
  `forced_all_dense_control`), EAGER. Drive a small dense (5-shot) AND a small sparse (24-shot) capture
  into separate dirs; teardown.
- `ac4_garbage_counters.py`: reduce the scored-selection captures → `evidence/ac4_garbage_counters.json`,
  per regime (dense/sparse): rows, duplicate / live-`-1` / out-of-range (vs `kv_capacity`) / adapter-error
  counts, and unwritten split into current-slot (H3, if the current slot is selected) vs non-current
  (real garbage). Fail-closed on zero rows / missing `slot_written_bits`/`kv_capacity`/`decode_step`.
- Wire the per-arm garbage counters into the production_ds ledger arm (a `garbage_counters_artifact`
  reference + the dense/sparse summary).

## Concrete Success Criteria
1. `serve.sh ds_garbage` exists (production ds + forced_all_assert, eager, no forced-all override). One
   eager run captures the SCORED production selection (dense + sparse), one TP=8 server, torn down to 0 MiB.
2. `evidence/ac4_garbage_counters.json` records, per regime, the AC-4 garbage counters (dup / live-`-1` /
   out-of-range-vs-kv-capacity / adapter-error + unwritten current-vs-non-current) on real scored rows;
   fail-closed (verified exit 2 on empty / missing fields). Real (non-current) garbage is reported with a
   number, not prose.
3. `build_ledger.py` references the artifact from the production_ds arm; `findings.md` records the AC-4
   scored-arm garbage result. The AC-2.1 reducer hardening (conditional H3 prose) is done if cheap.
4. Tests pass; provenance consistent. Commit; round-15-summary with BitLesson Delta + Goal Tracker Update
   Request. No selection/adapter FIX (guarded instrumentation only). No exit by lying / editing loop state
   / cancel-rlcr-loop.
