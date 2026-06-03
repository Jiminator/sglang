# Round 9 Summary

## Work Completed
Fixed the #K stale launcher-contract tests (+ plan-term cleanup) and ran/recorded the **AC-1b
chunked-prefill probe** — it passes, so the default chunked prefill is kept for the AC-11 sweep.

- **#K (blocking) — Option-B launcher tests updated to the evolved contract.**
  `test/registered/unit/development/test_option_b_scripts.py` encoded the pre-Round-4/Round-8
  contract and failed 2 tests. Rewrote them:
  - `test_dsa_server_radix_on_by_default_with_smoke_knob` — DSA default radix-on; `--disable-radix-cache`
    only inside the `DISABLE_RADIX_CACHE=1` guard.
  - `test_ds_server_radix_off_by_default` — `RADIX_ARGS=(--disable-radix-cache)` default.
  - `test_ds_server_artifact_driven_radix_on` — radix-on via `--double-sparsity-radix-fixture-artifact`;
    removed the obsolete fixed-marker assertion.
  Also reworded the Round-4/8 `AC-`/`DEC-`/`TIER` plan markers in production comments/help
  (serve_double_sparsity.sh, serve_native_nsa.sh, validator.py, server_args.py) to
  behavior-based language (plan Code Style); pre-plan markers left as-is.
- **AC-1b (mainline) — chunked-prefill probe PASSES at the radix-on operating point.** Booted
  DS radix-on via the fixtures-passed artifact (no env override) with `chunked_prefill_size=8192`.
  A 10565-token uncached prompt (radix flushed + unique prefix) prefilled in **genuine multiple
  chunks** (server log: `#new-token 8192` then `2432` for one seq) and was served without crash;
  the in-context needle "OSPREY-3141" was **recalled exactly**. `/get_server_info` confirms
  `chunked_prefill_size=8192`, `disable_radix_cache=false`, DS on, TP=8, page 64.
  Verdict: keep the default chunked prefill on BOTH DS and DSA (no disable needed; sidecars
  match for AC-11). A ~37k-token variant served (multi-chunk, no crash) but mangled the needle
  word — a DS sparse-decode (top_k=2048) long-context recall limit, not a chunked-prefill bug.

## Files Changed
- `test/registered/unit/development/test_option_b_scripts.py` — evolved launcher-contract tests.
- `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh` — plan-term reword.
- `python/sglang/srt/layers/attention/double_sparsity/validator.py`,
  `python/sglang/srt/server_args.py` — plan-term reword in the Round-8 additions.
- `runs/20260528_dsv32_mvp/` — `ac1b_probe.json` (verdict=PASS), `ac1b_server_info.json`.
- Commits: `e7951a59d` (#K + cleanup), `461119b46` (AC-1b probe). Pushed.

## Validation
- `pytest test/registered/unit/development/test_option_b_scripts.py
  test/registered/unit/layers/attention/test_double_sparsity_unit.py
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py -q` → **301 passed**
  (option-B-scripts 23/23 + DS unit + sequential).
- AC-1b hardware: multi-chunk prefill (8192+2432) served radix-on without crash; needle
  recalled exactly; `chunked_prefill_size=8192` confirmed.

## Remaining Items
- **TIER-2:** task13 AC-11 (3-trial radix-on DSA+DS sweep at conc 16/32/64, 120s/600s; gated on
  #F), task14 AC-12 (NIAH 4K/16K/64K + MMLU 5-shot), task15 evidence bundle.
- **#F (queued, now front-of-line):** DS KV-pool/effective-concurrency at mem 0.6 — must be
  resolved or explicitly accounted for before the AC-11 TTFT comparison (next round's gate).
- AC-10 label-capture artifact provenance note (server_args null / stale commit SHA) — fold
  into task15. Stale `calibrate.py` operator recipe docstring — queued cleanup.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-ds-longcontext-needle-recall-vs-topk
- Notes: Added the lesson that DS long-context needle recall is bounded by `top_k` (a needle in
  a ~37k context isn't selected from top-2048 → garbled recall) and is a sparsity tradeoff, NOT
  a chunked-prefill/serving bug — recorded now because the upcoming AC-12 NIAH 64K gate will hit
  exactly this and must distinguish a sparse-recall limit from a regression. The #K launcher-test
  update is tracked as a tracker/plan-evolution entry rather than a separate lesson.
