# Round 3 Summary

## Work Completed
Closed the two gaps Codex flagged in the round-2 "done" claims, making the DS-on smoke
boot fully evidence-complete on 8x H200.

- **AC-0 — hardware radix capture (task2), #B fixed.** The producer
  `_ds_radix_publish_extend_snapshot` returned before writing
  `double_sparsity_radix_capture` because production `ForwardBatch` carries no
  `req_to_token_pool` (same defect class as the selector's req_to_token bug).
  - Added `_resolve_req_to_token_for_capture(forward_batch, backend)` resolving in order:
    `forward_batch.req_to_token_pool.req_to_token` → `getattr(backend, "req_to_token", None)`
    → the active ForwardContext attention backend (unwrapping `TboAttnBackend.primary`).
  - Removed the broad `except Exception` wrapping the *required* context lookup — it was
    swallowing the missing-field error and silently dropping the publish. Best-effort
    catching now sits only around the snapshot read.
  - Made the capture SHA dtype-agnostic (`t.view(torch.uint8).numpy()`), fixing the
    `.numpy()` crash on fp16/bf16/fp8 tensors; kept the CUDA-graph-capture safety guard.
- **AC-1 — strict live server-info (task5).** Calling `/get_server_info` crashed the
  whole server every time after the DS bind. `get_internal_state` shipped
  `dict(vars(server_args))`, which now includes the DS bind's private CUDA tensors/pools
  (`_ds_channel_selection`, `_double_sparsity_token_label_table`, `_ds_token_to_kv_pool`,
  `_ds_req_to_token_pool`, `_double_sparsity_channel_mask`); pickling a torch tensor over
  the ZMQ pyobj IPC hit `torch.load`→`_legacy_load`→`UntypedStorage.dtype` AttributeError
  and killed the TokenizerManager recv loop. Fixed by excluding `_`-prefixed runtime
  attrs from the internal-state response (config fields are all public).
- Added a production-shaped CPU regression
  (`test_publishes_when_forward_batch_lacks_req_to_token_pool`): no `req_to_token_pool` on
  the batch, backend supplies `req_to_token`, `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` → asserts
  the capture publishes.

## Files Changed
- `python/sglang/srt/layers/attention/dsa_backend.py` — `_resolve_req_to_token_for_capture`;
  publish uses it; removed swallowing try/except around the context lookup.
- `python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py` —
  dtype-agnostic `_tensor_bytes_sha` (uint8 view).
- `python/sglang/srt/managers/scheduler.py` — `get_internal_state` excludes `_`-prefixed
  runtime attrs.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — production-shaped
  publish regression.
- `runs/20260528_dsv32_mvp/` — `ac0_capture_positive.json`, `ac1_server_info.json`,
  `ac1_generate.json`, `ac1_invalid_mask_rejection.md`.
- Commits: `6f95a9711` (AC-0 resolver + dtype-safe SHA), `bc534da7c` (get_internal_state
  fix + AC-0/AC-1 evidence), `76eef9c80` (AC-1 invalid-mask negative test).

## Validation
- DS unit suite green: `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` → **254 passed**.
- **AC-0 hardware (8x H200):** capture-enabled `/generate` returns non-empty
  `double_sparsity_radix_capture` — `prompt_len=11`, `per_token_slot_sha` count 11,
  `per_layer_written_all_true=True`, no `error` key. Capture-disabled negative: no key.
  (`runs/20260528_dsv32_mvp/ac0_capture_positive.json`)
- **AC-1 hardware (8x H200):** live `/get_server_info` returns and the server **stays
  alive** — `model_path=/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`, `tp_size=8`,
  `kv_cache_dtype=fp8_e4m3`, `page_size=64`, `enable_double_sparsity=True`,
  `disable_radix_cache=True`, `attention_backend=dsa`; `/generate` coherent
  (" Paris. The capital of the United States is Washington, D"). Invalid `CHANNEL_MASK_PATH`
  boot rejected fail-closed: `check_server_args` → `validate_double_sparsity` →
  `load_channel_mask` raises `DoubleSparsityChannelMaskMissing` before model load (no
  silent dense fallback). (`ac1_server_info.json`, `ac1_generate.json`,
  `ac1_invalid_mask_rejection.md`)
- GPUs freed after the probes (servers shut down).

## Remaining Items
- TIER-1 next round (gated on this fully-evidenced boot): AC-8/AC-9 smoke benchmarks
  (task7) + comparator (task8), AC-Q paired quality smoke (task9).
- TIER-2: AC-10 no-env-override radix flip + both fixtures (task11), AC-1b chunked-prefill
  probe (task12), AC-11 sweep (task13), AC-12 full quality (task14), evidence bundle (task15).
- Queued cleanup (non-blocking): stale `calibrate.py` operator recipe docstring.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260528-ds-private-server-args-attrs-crash-ipc
- Notes: Added `BL-20260528-ds-private-server-args-attrs-crash-ipc` — DS stashes private
  (`_`-prefixed) CUDA tensors on `server_args`; any handler that serializes
  `vars(server_args)` over the ZMQ pyobj IPC pickles a tensor and crashes the server on
  this torch build, so the introspection path must filter `_`-prefixed runtime attrs.
  I also extended the existing `BL-20260527-ds-metadata-via-forward-context` in
  `.humanize/bitlesson.md` (scope now covers the radix-capture producer and the
  `req_to_token` field; added the loop5-R3 `_resolve_req_to_token_for_capture` application
  and the rule that the required ForwardContext lookup must NOT sit inside a broad
  best-effort `except`). Both reinforce the same two-source-of-truth boundary: production
  `ForwardBatch` lacks DS fields (resolve via ForwardContext), and DS bind-time data lives
  on `server_args._*` (must be filtered before any IPC/pickle hop).
