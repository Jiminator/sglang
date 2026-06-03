# Round 8 Summary

## Work Completed
Fixed the #J false-pass (AC-Q now cleanly passes) and implemented + verified **AC-10**
end-to-end on 8x H200 — DS boots radix-on via a no-env-override config-bound artifact. TIER-1
is clean; TIER-2 has begun.

- **#J (blocking) — first-8 false-pass fixed.** Replaced Round-7's string-prefix fallback in
  `first_n_tokens_match` with **alphanumeric-subtoken overlap**: each first-n token splits into
  alnum runs (`100°C`→{`100`,`C`}; `53,59,61`→{`53`,`59`,`61`}); overlap requires a shared alnum
  subtoken; the exact whitespace check still handles pure punctuation (`.`). Now `100`/`100°C`
  match (shared `100`), but `10`/`100` and `Paris.`/`London.` do NOT. +4 false-pass regressions.
  Recomputed the concise AC-Q gate offline from the saved deterministic outputs → `all_pass=true`
  under the corrected gate (prefix 0.95, rouge 0.944, niah 5/5, first8_div 0; the 100/100°C match
  now comes from the shared `100` subtoken, not a broad prefix).
- **AC-10 (mainline) — no-env-override radix flip (DEC-5).**
  - `write_radix_fixture_state(...)` writes a JSON state recording BOTH M3-B fixtures passed +
    a config fingerprint (model/tp/page/kv-dtype/channel-mask-SHA).
  - `--double-sparsity-radix-fixture-artifact` + `apply_radix_fixture_artifact` (called in
    `check_server_args` BEFORE `validate_double_sparsity`) verify the state matches THIS boot and
    record the flip — fail-closed on mismatch/missing/partial. 6 CPU regressions.
  - `serve_double_sparsity.sh`: `RADIX_FIXTURE_ARTIFACT` → pass artifact + drop
    `--disable-radix-cache`; the env override now only enables radix-on for dev/fixture runs.
  - **Hardware:** FP8 scale-stability fixture PASSED; label-capture fixture PASSED (cold==warm
    label SHAs, `cached_tokens>0`) after fixing its `/flush_cache` JSON-parse bug; wrote the state
    file; booted DS **radix-on authorized purely by the artifact (no `SGLANG_DS_RADIX_OVERRIDE`)** —
    validator logged the flip with the artifact SHA, `/get_server_info` shows
    `disable_radix_cache=false`, DS on, TP=8, fp8, page 64; `/generate`→"Paris".

## Files Changed
- `test/manual/_dsv32_quality_smoke_lib.py` — alnum-subtoken `first_n_tokens_match` (#J).
- `test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py` — +4 #J regressions.
- `python/sglang/srt/layers/attention/double_sparsity/validator.py` — `write_radix_fixture_state`,
  `apply_radix_fixture_artifact`, `radix_fixture_config_fingerprint`, `RADIX_FIXTURE_STATE_SCHEMA`.
- `python/sglang/srt/server_args.py` — `double_sparsity_radix_fixture_artifact` field + CLI +
  `check_server_args` wiring.
- `development/serve_double_sparsity.sh` — `RADIX_FIXTURE_ARTIFACT` knob; env override = dev-only.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — +6 AC-10 regressions.
- `test/manual/test_dsv32_radix_label_capture_fixture.py` — `_flush_cache` plain-text fix.
- `runs/20260528_dsv32_mvp/` — `ac10_fp8_scale_stability.json`, `ac10_label_capture.json`,
  `ds_radix_fixture_state.json`, `ac10_radixon_server_info.json`, recomputed
  `dsv32_quality_smoke_concise.json`.
- Commits: `d47dcbadb` (#J), `fa4473694` (AC-10 mechanism), `67422e698` (AC-10 hardware). Pushed.

## Validation
- `pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py -q` → **278 passed**
  (254 DS unit incl. 6 new AC-10 + 18 sequential incl. 4 new #J + first-8 tests).
- Hardware AC-10: both fixtures pass; radix-on boot via artifact (no env), `disable_radix_cache=false`,
  `/generate` coherent. AC-Q recomputed → `all_pass=true` under the precise gate.

## Remaining Items
- **TIER-2:** task12 AC-1b (chunked-prefill probe), task13 AC-11 (3-trial radix-on sweep; gated on
  #F), task14 AC-12 (NIAH 4K/16K/64K + MMLU 5-shot), task15 evidence bundle.
- **#F (queued):** DS KV-pool/effective-concurrency at mem 0.6 — resolve/account for before AC-11
  TTFT (now relevant since AC-11 runs radix-on next).
- Round-7 artifact-completeness note (graph-mode primes JSON / server-info in meta JSONs) — fold
  into the task15 bundle. Stale `calibrate.py` operator recipe docstring — queued cleanup.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-ds-radix-flip-config-bound-artifact
- Notes: Added the AC-10 no-env-override radix-flip pattern — a config-bound fixtures-passed state
  file (`write_radix_fixture_state` / `apply_radix_fixture_artifact`) verified against the boot's
  fingerprint before validation, with the env override demoted to dev/fixture-only; includes the
  chicken-and-egg resolution (run fixtures under the override → write state → final boot uses the
  artifact) and the `/flush_cache`-returns-plain-text fixture gotcha. The #J fix is recorded as a
  refinement of the AC-Q gate (alnum-subtoken overlap) in the goal tracker rather than a separate
  lesson.
