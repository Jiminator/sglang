# R1 finding — DS error-abort crashed the server (now fixed) + selector reuse-edge

## What happened
mb_v2 (R1, first launch) booted DS@0.8 radix-on, then the conc-64 tax probe burst tripped a DS
**selector_runtime_error** at layer 0 for many rows at once:

    double_sparsity error cls=selector_runtime_error layer_id=0 selector_id=layer0-row0
    message=The size of tensor a (36) must match the size of tensor b (6790) at non-singleton dimension 0

The per-request error-abort path (`_maybe_abort_on_ds_error`, batch_result_processor.py) then called
`req.check_finished()`. Upstream PR #25725 ("Fix the misnamed request finish-check method...") RENAMED
that method to `update_finish_state()` and updated all in-tree callers, but the DS feature-branch abort
path drifted and kept the old name → `AttributeError` inside the scheduler event loop → **SIGQUIT killed
the whole TP=8 server**. Result: the DS sweep produced zero jsonls (rc=1); the c30 tax probe + sweep hit
a dead server. (DSA side was unaffected — native_nsa never runs the DS path.)

## Two layers
1. **Crash (FIXED, commit 1a29be00d):** `check_finished()` → `update_finish_state()` (code + docstring).
   `set_finish_with_abort` sets `to_finish`; `update_finish_state` materialises `finished_reason = to_finish`.
   The DS error-containment seam (`try_run_ds_step`, error_containment.py) is BY DESIGN: catch the typed
   selector exception, mark THAT request failed, keep siblings serving. The crash defeated that design.
2. **Selector reuse-edge (characterized, contained — not root-caused this round):** "36 vs 6790 at dim 0"
   is the absorbed-latent scoring on the **radix prefix-reuse + chunked-prefill** path: 36 = the extend
   (new query tokens), 6790 = the cached-context latent dim. This is DEC-12's authorized reuse scenario.
   Round-0's steady-state sweep ran CLEAN at conc-64 — **zero** selector_runtime_error lines in its serve
   log; its end-of-run crash was teardown (`SIGTERM received. Draining requests...` -> scheduler exit code
   -15), NOT the bug. The edge fires en masse only under the R1 tax probe's 100%-identical-prefix burst
   (gsp-range-ratio 1.0, 1 group) at conc-64: 77500 errors in ~10s (28750 in one second). The bug scales
   with FULL-prefix reuse; the production ~55% reuse barely touches it. Root-causing the paged-scoring
   shape mismatch is a separate kernel project; the containment design explicitly degrades these per-request.

## Why this escaped earlier detection (evidence-backed)
1. **No representative workload triggered it.** Round-0's verdict sweep at production ~55% reuse logged
   ZERO selector errors. The `36 vs 6790` mismatch needs a short extend over a long FULLY-shared cached
   prefix at concurrency. R1's fixed-conc 100%-identical-prefix tax probe was the first run to create that
   (Round-0's AC-4 fell back to sweep-derived TPOT — `bench_one_batch` was blocked — so no fixed-conc
   serving probe had ever existed).
2. **All three failure modes live on the error-abort path** (`_maybe_abort_on_ds_error` -> check_finished
   -> KV release), which only executes WHEN a row is sanitized. Selector errors never fired under
   representative load, so the path was never exercised: dead code on the happy path.
3. **Semantic merge conflict.** The rename check_finished -> update_finish_state (#25725) landed in main
   2026-05-19; the DS branch merged main 2026-05-25 (65618a8d3). The rename updated every caller that
   existed in main, but the DS abort path is a NEW call site on the feature branch — textually clean merge,
   semantically broken. Git cannot flag a renamed method called by unmerged code.
4. **Containment swallows it by design + no aggregate trace pre-B1.** `try_run_ds_step` catches the selector
   error, fails the one request, keeps serving. The verdict is computed over successful requests. Before
   B1 (this round) bench_serving emitted no DS-error/dense-fallback aggregate, so a contained error left a
   trace ONLY in the serve log. (This is exactly why mb_v2/ds_only now print `selector_runtime_errors=N`.)
5. **Unit test stops one layer short.** `test_double_sparsity_unit.py` tests selector sanitization by
   MagicMock-ing `retrieve_topk` and asserting the error is RECORDED; it never drives a real Req through
   the scheduler's `_maybe_abort_on_ds_error` to the `req.check_finished()` call (the sibling code notes
   "per-request abort plumbing through the scheduler boundary remains queued").

## In-scope deliverable (R1)
- Crash-fix landed so the server survives → the verdict sweep can complete.
- Re-run (HEAD 1a29be00d) measures the per-phase selector-error count (mb_v2 emits
  `DS selector_runtime_errors=N server_crashed=N`) so the verdict reports the abort rate HONESTLY.
- If the SWEEP (not the tax burst) shows material selector errors, the radix-on verdict is reported as
  compromised; if ~0 (Round-0 evidence), the verdict stands with the reuse-edge documented as a known
  DS radix-on containment case.

## Raw evidence
`results_v2/crash_evidence_r1/serve_ds080_CRASHED.log` (the selector errors + SIGQUIT),
`results_v2/crash_evidence_r1/mb_v2_firstrun.log`, `.../log_ds_c64.txt`.
