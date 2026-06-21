# Round 13 Contract

Round 12 was ADVANCED and closed the last CPU evidence-package item. The remaining work is the
original-plan GPU/instrumentation close-out (Codex Mainline Gap #1). Codex's ordered plan puts the
forced-all physical-slot assertions FIRST — and AC-2.1 is the plan's lower-bound, load-bearing control:
it is the assertion that the dense regression is DOWNSTREAM of selection (H3), which the whole verdict
rests on. This round delivers exactly that.

## Mainline Objective (exactly one)
**Produce AC-2.1: the forced-all dense physical-slot assertions** — guarded diagnostic instrumentation
at the `logical_to_physical` adapter seam (default-off, eager-only, byte-identical when off), one
`ds_forced_all` dense GPU run with it enabled, and a fail-closed offline reducer →
`evidence/forced_all_assertions.json`.

## Target ACs
- **AC-2.1** (primary): physical slots == `req_to_token[req_pool, 0:seq_len]`; no duplicate / live-lane
  `-1` / unwritten / out-of-range slots; adapter `error_count == 0` — per layer/step, on the forced-all
  dense control.
- **AC-4** (secondary): reuse the same counters as the per-arm length-cap garbage-rate columns.

## Blocking Side Issues (the instrumentation — it IS the mainline)
- No `forced_all_assertions.json` artifact exists; the adapter physical-slot equality / garbage counters
  are not instrumented. Add a guarded diagnostic at the existing seam (mirror the already-present
  `_publish_ds_request_summary` guard: skip during `torch.cuda.is_current_stream_capturing()`, gate on a
  new default-off config flag, eager-only) that dumps, per (rank, req, layer): `ds_out` (physical slots),
  `req_to_token[req_pool, 0:seq_len]`, the `_ds_slot_written` bits for those slots, `error_count`,
  `seq_len`. NO mutation of the selected set; host-side copy only; production byte-identical when the flag
  is off.

## Queued Side Issues (documented, OUT OF SCOPE — subsequent rounds)
- AC-3.1 captured-row materialized fp32 `K_label` selected-index equality (extend `latent_capture` to
  store bounded latent/scales/query, then offline analyzer).
- AC-2.4 recall-oracle@2048 (NIAH-only; existing `recall_oracle` flag + `.sglang_ds_oracle/trial.json`).
- AC-4 remaining serial cells (DSA-radix serial, production DS sparse serial, ref_faithful serial,
  ref_cosine serial) and selected-vs-total gaps.
- AC-8 final writeup (after the above).
- Plan-term comment cleanup; reference-mode fail-closed.

## Concrete Success Criteria
1. A new guarded diagnostic (config flag + a `forced_all_assert` capture module, mirroring
   `score_capture`/`selection_capture`) dumps the adapter physical-slot data when on; production is
   byte-identical when off (flag default false; skipped under CUDA-graph capture; eager-only).
   `test_reference_selectors.py`-style smoke or a CPU unit check confirms the off-path is unchanged.
2. One `ds_forced_all` dense GPU run (eager, `--disable-cuda-graph`) with the flag enabled produces the
   capture dumps; one TP=8 server at a time; teardown to 0 MiB after.
3. `development/loop13/ac2_1_forced_all_assertions.py` reduces the captures to
   `evidence/forced_all_assertions.json` with per-layer/step totals: rows checked, physical==
   `req_to_token[0:seq_len]` equality count, duplicate count, live-lane `-1` count, unwritten-slot count,
   out-of-range count, adapter `error_count`. **Fail-closed**: nonzero exit if zero dense forced-all rows
   are observed or any required counter is absent; report PASS only if all rows equal and all error
   counts are 0.
4. The artifact is wired into the ledger (AC-2.1 status + AC-4 garbage-rate reuse). `findings.md` records
   the AC-2.1 result (downstream assertions PASS ⇒ dense forced-all selection is a proven no-op, so the
   dense recovery is via slot-validity, consistent with the H3 verdict). Tests pass; provenance
   consistent. Commit; round-13-summary with BitLesson Delta + Goal Tracker Update Request. No
   selection/adapter FIX (instrumentation only, guarded). No exit by lying / editing loop state /
   cancel-rlcr-loop.
