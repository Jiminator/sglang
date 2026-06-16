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
   Round-0's steady-state sweep ran clean at conc-64 (one straggler at teardown); the edge fires en masse
   only under the tax probe's cold concurrent shared-prefix burst. Root-causing the paged-scoring shape
   mismatch is a separate kernel project; the containment design explicitly degrades these per-request.

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
