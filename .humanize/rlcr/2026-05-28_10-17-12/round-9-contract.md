# Round 9 Contract

## Mainline Objective
**Run and record the AC-1b chunked-prefill probe (task12) at the final radix-on operating
point**, so the AC-11 sweep can later collect artifacts at a settled config. Boot DS radix-on
(via the AC-10 artifact), drive a prompt longer than `chunked_prefill_size` so chunked prefill
actually engages, and record whether DS serves correctly. If the probe passes, keep the
default chunked-prefill; if it fails, disable chunked prefill on BOTH DS and DSA
(`chunked_prefill_size=-1`) and file a follow-up. Save the probe artifact under
`runs/20260528_dsv32_mvp/`.

A small **blocking prerequisite (#K)** must be fixed first: the registered Option-B launcher
tests encode the old launcher contract and now fail.

## Target ACs (≤ 2)
- **AC-1b** — chunked-prefill probe run and recorded; if disabled, both sides show
  `chunked_prefill_size=-1`; a mismatched-setting AC-11 set is invalid.

## Blocking issues in scope
- **#K — stale registered Option-B launcher tests fail.**
  `test/registered/unit/development/test_option_b_scripts.py` forbids any `--disable-radix-cache`
  path in `serve_native_nsa.sh` (Round 4 added the `DISABLE_RADIX_CACHE=1` smoke knob,
  default radix-on) and requires the obsolete fixed-line AC-10 marker before the first
  `--disable-radix-cache` in `serve_double_sparsity.sh` (Round 8 replaced it with the
  `RADIX_ARGS`/`RADIX_FIXTURE_ARTIFACT` contract). Update the test to assert the EVOLVED
  contract: (1) DSA default radix-on + explicit `DISABLE_RADIX_CACHE=1` smoke path; (2) DS
  default radix-off, artifact-driven radix-on via `--double-sparsity-radix-fixture-artifact`
  (no fixed marker); (3) remove/rewrite the obsolete marker assertion. Rerun the launcher
  test + DS/sequential suites green.

## Queued / in-scope cleanup (fold into the #K commit; do NOT expand the round)
- **Plan-term hygiene (plan §Code Style).** Round 8 introduced `AC-`/`DEC-`/`Tier`/`Phase`
  markers in production-facing code/comments (`serve_double_sparsity.sh`, `validator.py`,
  `server_args.py` help/comments). Reword to behavior-based language. This is plan-required
  style cleanup on code I just added; bundle it with the #K launcher-contract edits since
  they touch the same files. Keep existing pre-plan in-code markers (e.g. AC-10-FIXTURE
  history) only where they already existed.

## Queued / explicitly out of scope this round
- **#F** — DS KV-pool/effective-concurrency at mem 0.6; resolve/account for before AC-11
  (task13), NOT this round (AC-1b is a single-prompt probe, not a concurrency sweep).
- **task13 AC-11**, **task14 AC-12**, **task15 evidence bundle** — subsequent rounds, in plan
  order, after AC-1b.
- AC-10 label-capture artifact provenance note (server_args null / stale commit SHA) — fold
  into task15.
- Stale `calibrate.py` operator recipe docstring.

## Round success criteria
1. `test/registered/unit/development/test_option_b_scripts.py` updated to the evolved
   launcher contract and passes; plan-term markers removed from the Round-8 production
   additions; full DS unit + sequential + option-B-scripts suites green.
2. AC-1b probe executed on 8x H200: DS booted radix-on (artifact) with chunked prefill at the
   default `chunked_prefill_size`; a prompt longer than that size is served; the probe records
   whether chunked prefill works (coherent output, no crash) or fails. Artifact saved with the
   recorded `chunked_prefill_size` and verdict.
3. If the probe fails, the recorded decision disables chunked prefill on BOTH DS and DSA
   (`chunked_prefill_size=-1`) and a follow-up is noted; if it passes, the default is kept and
   recorded. Either way the operating point is pinned for the future AC-11 sweep.
4. Commit + push each step. Goal tracker updated (task12/AC-1b status, #K resolved);
   `round-9-summary.md` with a BitLesson Delta. No immutable-section changes.

## Known risks / notes
- A >8192-token prompt at DS mem 0.6 (small KV pool) may stress memory; watch for OOM and
  record it as a probe outcome (it would motivate disabling chunked prefill / a follow-up),
  not a silent failure.
- Operational: don't kill the pre-existing port-30000 router; free port; verify `nvidia-smi`
  clear before boots; standalone `pkill`/`commit`/`push`.
