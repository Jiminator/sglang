# Round 14 Contract

Round 13 was ADVANCED, but Codex correctly found my AC-2.1 was INCOMPLETE and over-claimed. Three real
defects in the forced-all assertion (all P1, all correct):
1. The capture never records `_ds_slot_written`; I wrongly claimed "unwritten is subsumed by
   physical==req_to_token equality" — but `_ds_slot_written` is a SEPARATE backend validity bitmap;
   equality proves the GATHER, not that the KV slot is marked written. AC-2.1's "no unwritten" + AC-4's
   unwritten garbage counter are unmeasured.
2. The capture overwrites per `(rank, req, layer)` across decode steps → it's NOT per-step; a one-step
   invalid event is overwritten. AC-4 wants per-step garbage rates.
3. The out-of-range check uses `req_to_token.shape[1]` (max_context_len = 202756) instead of the KV-slot
   capacity (`_ds_slot_written.shape[1]` = `token_to_kv_pool.size + page_size`) — wrong bound.

## Mainline Objective (exactly one)
**Repair AC-2.1 to the plan's bar:** capture `_ds_slot_written` per live physical slot, key/record by
decode step (no overwrite), check physical out-of-range against the true KV-slot capacity, re-run, and
regenerate `forced_all_assertions.json` so the unwritten + per-step + true-range assertions are real.

## Target ACs
- **AC-2.1** (primary): physical==`req_to_token[0:seq_len]` AND no unwritten (`_ds_slot_written` True) AND
  no dup/`-1`/out-of-range (vs KV-slot capacity) AND adapter `error_count==0`, per layer AND step.
- **AC-4** (secondary): the same counters (incl. unwritten, per-step) as the forced-all garbage-rate.

## Blocking Side Issues (truly block AC-2.1 / AC-4 integrity)
- `build_ledger.py` `DS_DEFAULTS` is stale after the new `DoubleSparsityConfig.forced_all_assert` field —
  `effective_ds_config` omits it while claiming to be fully resolved. Add `forced_all_assert: false`.
- `findings.md` / `evidence_table.md` overclaim R13 as AC-2.1/AC-4-garbage completion. Until the repaired
  reducer passes, the result is adapter-GATHER evidence only; reconcile both to the repaired artifact.

## Queued Side Issues (documented, OUT OF SCOPE — subsequent rounds)
- AC-3.1 captured-row materialized fp32 `K_label` selected-index equality (extend latent capture + analyzer).
- AC-2.4 recall-oracle@2048 (NIAH-only).
- AC-4 garbage counters on the SCORED arms (enable the repaired capture on production_ds/ref_*); serial
  cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial); selected-vs-total.
- AC-8 final writeup.
- serve.sh help/error text omits newer modes; plan-term comment cleanup.

## Approach
- Resolve the bitmap at the hook the SAME way the reference selector / production path do:
  `_get_attn_backend()` → `.primary` if `_TboAttnBackend` → `getattr(_, "_ds_slot_written", None)` (both
  already used in this file). Fail closed in the diagnostic if `forced_all_assert=true` and it's absent.
  It is `[num_ds_layers, kv_slots]`; index `[layer_id]` (same layer_id the selector uses) and gather the
  bits at the live physical slots. Capacity for out-of-range = `slot_written.shape[1]`.
- Per-step identity: a capture-module monotonic counter keyed by `(rank, req, layer)` (no decode-step id
  at the seam); filename + record include `step`. Reducer keys by `(rank, req, layer, step)`.

## Concrete Success Criteria
1. `maybe_dump_forced_all_assert()` accepts `slot_written` and records, per live physical slot, the
   `_ds_slot_written[layer_id, slot]` bit + `kv_capacity = slot_written.shape[1]` + a `step` id; the
   filename includes the step (no overwrite). The hook fails closed when the flag is on and the bitmap is
   absent. Off-path byte-identical (flag default false; reference tests pass; py_compile clean).
2. `ac2_1_forced_all_assertions.py` requires `slot_written_bits`, `step`, `kv_capacity`; keys by
   `(rank,req,layer,step)`; counts unwritten live slots from the bits; checks out-of-range vs
   `kv_capacity`; fails closed if any dense forced-all record lacks step/bits/capacity. PASS only if all
   rows equal + sweep + 0 dup/`-1`/out-of-range/unwritten/error.
3. One `ds_forced_all_assert` eager GPU run regenerates `evidence/forced_all_assertions.json` with the new
   per-step + unwritten + true-range fields. One TP=8 server, torn down to 0 MiB.
4. `build_ledger.py` `DS_DEFAULTS` includes `forced_all_assert: false`; regenerated. `findings.md` /
   `evidence_table.md` reconciled to the repaired artifact (AC-2.1 incl. unwritten + per-step). Tests
   pass; provenance consistent. Commit; round-14-summary with BitLesson Delta + Goal Tracker Update
   Request. No selection/adapter FIX (guarded instrumentation only). No exit by lying / editing loop state
   / cancel-rlcr-loop.
