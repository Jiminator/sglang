# Round 3 Contract

## Mainline Objective
**Close the two AC-0 / AC-1 gaps Codex flagged in the round-2 "done" claims** — i.e.
make the DS-on smoke boot fully evidence-complete:
1. **AC-0 hardware radix capture (task2)** — fix the blocking publish bug: production
   `ForwardBatch` has no `req_to_token_pool`, so `_ds_radix_publish_extend_snapshot`
   returns before writing `double_sparsity_radix_capture` (the same defect class as the
   selector's req_to_token bug). Add a shared ForwardContext resolver (unwrap
   `TboAttnBackend.primary`), use it in the publish, add a production-shaped regression
   (no `req_to_token_pool` on the batch; backend supplies `req_to_token`), then rerun the
   hardware probe until `/generate` returns non-empty `double_sparsity_radix_capture`
   with `per_token_slot_sha` populated, `per_layer_written_all_true=True`, no error key.
   Also capture the negatives (capture-disabled → no key; decode-only → no publish).
2. **Strict AC-1 (task5)** — reboot DS and save the LIVE `/server_info` (or
   `/get_server_info`) JSON with `curl --fail`, asserting DS enabled, TP=8,
   `kv_cache_dtype=fp8_e4m3`, `page_size=64`, expected radix setting, and the cluster
   model path; save `/generate` text; and capture the missing/invalid-mask validator
   rejection artifact (AC-1 negative test).

## Target ACs (≤ 2)
- **AC-0** (hardware capture probe) — primary.
- **AC-1** (strict live server-info + invalid-mask rejection) — primary.

## Blocking issues in scope
- **#B (now confirmed by Codex): `_ds_radix_publish_extend_snapshot` cannot resolve
  `req_to_token` on production `ForwardBatch`** (no `req_to_token_pool` field) → capture
  never publishes. Fix via the ForwardContext resolver. This blocks AC-0.

## Queued / explicitly out of scope this round
- TIER-1 smoke benchmarks + comparator (task7/task8, AC-8/9) and paired quality smoke
  (task9, AC-Q): the NEXT round, gated on a fully-evidenced boot (AC-0/AC-1). Begin only
  if AC-0+AC-1 close with ample time.
- TIER-2: AC-10 radix flip (task11), AC-1b (task12), AC-11 (task13), AC-12 (task14),
  evidence bundle (task15).
- Stale `calibrate.py` operator recipe docstring (queued cleanup; provenance recorded).

## Round success criteria
1. `_ds_radix_publish_extend_snapshot` resolves `req_to_token` via ForwardContext when
   `forward_batch.req_to_token_pool` is absent; a production-shaped unit regression
   (no `req_to_token_pool`; ForwardContext backend supplies `req_to_token`;
   `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`) asserts `double_sparsity_radix_capture` publishes.
   Full DS unit suite green.
2. Hardware AC-0: `/generate` with capture enabled returns non-empty
   `double_sparsity_radix_capture` (`per_token_slot_sha` populated,
   `per_layer_written_all_true=True`, no error key); negatives captured. Artifacts saved.
3. Hardware AC-1: live `/server_info` JSON saved (required fields asserted) + `/generate`
   text + invalid/missing-mask validator-rejection artifact. Artifacts saved.
4. Goal tracker updated (task2→done, task5→done); `round-3-summary.md` written with a
   BitLesson Delta. No immutable-section changes.
