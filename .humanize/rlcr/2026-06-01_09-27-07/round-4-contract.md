# Round 4 Contract

## Mainline Objective
Make the M0 recall oracle **fail-closed and binding** (AC-1/AC-2): config-borne activation so it records on the TP worker processes (the root cause of the silently-missing 64K records), strict harness-span validation, explicit failure artifacts instead of silent skips/swallows, expected-record-count assertions in the sweep, and a re-run of 4K/16K/64K with no missing lengths — so task7's budget-vs-scorer attribution becomes binding.

## Target ACs (1–2)
- **AC-1** (primary): oracle fail-closed (no silent guessing / no silent failure) + no missing-length oracle records.
- **AC-2**: binding budget-vs-scorer evidence (re-adjudicated from complete oracle records).

## Blocking Side Issues In Scope (the objective itself)
- **Oracle hook fail-open** (flagged every review): `_maybe_record_recall_oracle` returns silently with no active trial, FILTERS out-of-range needle positions instead of rejecting them, and swallows all exceptions. Plus the activation is env-based, which does NOT reach the TP workers (the cause of the missing 64K records).

## Queued Side Issues Out Of Scope (justified)
- **Graph-safe Triton scorer port + full AC-3 measurement matrix** (task #13): heavy kernel + GPU measurement; sequenced after the oracle is binding (the oracle is the M0 evidence the rest rests on).
- **Tier-2.A / AC-4** (task13–17): next workstream after AC-3 measurement.
- **M4 consolidation / AC-6 perf + final decision record** (task19–20): end milestone.
- **Plan-marker code/comment cleanup**: pre-merge; queued.

## Round Success Criteria
- DS config gains `recall_oracle: bool` (default `false`); the oracle hook is **active via config** (reaches TP workers), default-off is a cheap no-op + byte-identical selection.
- Hook is **fail-closed when active**: a needle position out of `[0, max_tokens)` ⇒ explicit `span_out_of_range` failure record (NOT filtered); an exception ⇒ explicit `exception` failure record (NOT swallowed); no active trial ⇒ explicit `no_active_trial` failure record. All keyed by `(request_id, trial_id, layer_id, decode_step)`.
- Fixed default trial/sink paths (env-overridable) so the harness (writer) and server (reader/writer) agree without env propagation.
- The oracle sweep clears the sink, then **asserts every measured (length, trial) produced records** and reports per-length success/failure counts; it fails loudly on any missing length (catches the old silent-64K case).
- **Re-run 4K/16K/64K oracle on 8×H200**: records present for ALL lengths including 64K; produce the binding budget-vs-scorer artifact (no inferred 64K verdict).
- Unit tests for the fail-closed hook (no-trial / out-of-range / exception ⇒ failure record) and the config-borne activation. All DS unit tests pass.
