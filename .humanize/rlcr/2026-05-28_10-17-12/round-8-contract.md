# Round 8 Contract

## Mainline Objective
**Implement AC-10 (task11): flip DS to radix-cache-ON via a no-env-override mechanism, with
both radix fixtures passing on hardware.** Add a ServerArgs/launcher or state-file/artifact-
path contract that sets `_double_sparsity_radix_fixture_passed` before
`validate_double_sparsity` runs in `check_server_args()` (DEC-5, NO env override); run both
radix fixtures (label-capture + FP8 scale-stability) on 8x H200; remove `--disable-radix-cache`
from the DS launch; boot DS radix-on and confirm it serves.

A small **blocking prerequisite (#J)** must be fixed first so TIER-1's AC-Q is cleanly
accepted before TIER-2 work is meaningful.

## Target ACs (≤ 2)
- **AC-Q** (close cleanly via the #J fix — re-pass all four gates under a precise first-8
  overlap check).
- **AC-10** (radix flip + both fixtures + radix-on boot, no env override).

## Blocking issues in scope
- **#J — first-8 overlap false-pass hole.** Round-7's prefix fallback makes
  `first_n_tokens_match('10','100')` return True, which would pass a wrong short answer.
  Replace the string-prefix fallback with **alphanumeric-subtoken normalization**: split each
  first-n whitespace token into alnum runs (e.g. `100°C` → {`100`,`C`}; `53,59,61` →
  {`53`,`59`,`61`}) and require a shared alnum subtoken; keep the exact whitespace-token check
  for pure-punctuation answers (e.g. `.` vs `.`). Result: `100` vs `100°C` → match (shared
  `100`); `10` vs `100` → NO match; `Paris.` vs `London.` → NO match. Add regressions for the
  false-pass cases, rerun the CPU suite, and **recompute** the concise AC-Q gate from the
  saved deterministic outputs (`dsv32_quality_smoke_concise.json`) to show `all_pass=true`
  under the corrected gate (no hardware reboot needed — outputs are fixed).

## Queued / explicitly out of scope this round
- **#F** — DS KV-pool/effective-concurrency at mem 0.6; resolve/account for before AC-11
  (task13), NOT this round.
- **task12 AC-1b** (chunked-prefill probe), **task13 AC-11** (sweep), **task14 AC-12**,
  **task15 bundle** — subsequent rounds, in plan order, after AC-10.
- Round-7 artifact-completeness note (missing graph-mode primes JSON / server-info in the meta
  JSONs) — fold into the task15 evidence bundle.
- Stale `calibrate.py` operator recipe docstring.

## Round success criteria
1. `first_n_tokens_match` uses precise alnum-subtoken overlap: `100`/`100°C` pass, `10`/`100`
   and `Paris.`/`London.` do NOT; `.`/`.` still matches. New false-pass regressions pass; full
   DS + sequential suites green. Concise AC-Q recomputed → `all_pass=true` under the corrected
   gate (artifact updated).
2. AC-10 flip mechanism implemented with NO env override: a ServerArgs/launcher field or
   state-file/artifact-path sets `_double_sparsity_radix_fixture_passed` before
   `validate_double_sparsity` in `check_server_args()`; a CPU regression proves the guard
   accepts radix-on only when the fixture-passed contract is satisfied (and still rejects a
   bare radix-on boot). `SGLANG_DS_RADIX_OVERRIDE` env path is NOT the mechanism.
3. Both radix fixtures pass on 8x H200: `test_dsv32_radix_label_capture_fixture.py` (with
   `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`) and `test_dsv32_fp8_scale_stability.py` (with
   `SGLANG_DS_FP8_SCALE_PROOF=1`). Artifacts saved under `runs/20260528_dsv32_mvp/`.
4. `--disable-radix-cache` removed from the final DS launch path; DS boots **radix-on**
   (`/get_server_info` shows `disable_radix_cache=false`, DS enabled, TP=8, fp8, page 64) and
   `/generate` is coherent. Artifact saved.
5. Commit + push each step. Goal tracker updated (task9→AC-Q clean, #J resolved, task11→AC-10
   status); `round-8-summary.md` with a BitLesson Delta. No immutable-section changes.

## Known risks / notes
- The M3-B radix label-capture fixture has a long defect history (loop4 rounds 35-38, AC-0).
  AC-0's producer fix (Round 3) made the capture publish; verify the fixture consumes it
  correctly. If a fixture surfaces a real defect, fixing it is in-scope for AC-10.
- Radix-on may change KV/memory behavior; watch for OOM at boot (DS mem 0.6) and adjust only
  if needed, documented.
- Operational: don't kill the pre-existing port-30000 router; free port; verify `nvidia-smi`
  clear before boots; standalone `pkill`/`commit`/`push`.
