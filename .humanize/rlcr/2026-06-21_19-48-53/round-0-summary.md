# Skip-impl code-review summary — loop13 Double-Sparsity diagnosis

This is a `--skip-impl` review of an already-complete body of work: the 22-round Loop-13 **diagnosis loop**
(commits `fc6ac20a7 … 762330437`, base `loop13-base` = `180f6dd6d`). The prior RLCR loop's implementation
phase already returned **COMPLETE** (8/8 ACs); this session re-runs the code-review phase against a valid
ancestor base (the original `main` base was a disjoint single-commit history, so `git merge-base` errored —
fixed by reviewing against `loop13-base`). No new code was written this round; the tree is clean.

## What Was Implemented
The diff under review (`loop13-base..HEAD`, 58 files, +13132/-10) is the Loop-13 diagnosis harness, evidence
package, and verdict — **diagnostic instrumentation only, no selection/adapter fix landed**.
- **`development/loop13/`** — guarded serve modes (`serve.sh`) + `run_gsm8k.sh`; the per-AC reducers
  (`ac2_1_*`, `ac2_2_*`, `verify_ac2_3.py`, `ac4_garbage_counters.py`, `niah_recall_oracle.py`,
  `ac3_1_materialized_k_equality.py`, `ac4_selected_vs_total_probe.py`, `ac6_*`, `ac8_selfcheck.py`);
  `build_ledger.py` (provenance-consistent ledger with per-artifact fail-closed gates); the deliverable
  `ROOT_CAUSE.md` + committed `evidence/`.
- **Production code** (`python/sglang/srt/layers/attention/double_sparsity/*` + `models/deepseek_v2.py`) —
  config-borne, **default-off** diagnostic additions only: the reference selectors (`selector_impl ∈
  {reference_rawdot, reference_cosine}`, `reference_include_current`) and capture flags (`forced_all_assert`,
  `recall_oracle`, `materialized_k_capture`, `score_reduce_dtype`). Every capture hook is inside the existing
  `not torch.cuda.is_current_stream_capturing()` guard, host-side copy only, mutates nothing in the selected
  set, eager-only → the production decode path is **byte-identical when the flags are off** (the 5
  reference-selector unit tests pass; flags default `False` in all 4 config places).

## Files Changed
58 files vs `loop13-base`; net new diagnostic code + evidence under `development/loop13/`, plus the guarded
default-off DS instrumentation in `python/sglang/srt/layers/attention/double_sparsity/` and
`python/sglang/srt/models/deepseek_v2.py`. No production behavior change when the diagnostic flags are off.

## Validation
- `python3 development/loop13/ac8_selfcheck.py` → "AC-8 PACKAGE COMPLETE".
- `python3 development/loop13/build_ledger.py` → provenance consistent.
- Full CPU reducer suite + the 5 reference-selector unit tests pass; AC-2.3 artifact unchanged.
- Raw capture dirs (`.sglang_ds_*`) and GSM8K `.out` logs are gitignored (derived scores committed in the
  per-arm JSONs/table).

## Verdict delivered (`development/loop13/ROOT_CAUSE.md`)
Two regressions, not an algorithm/mask failure: dense 0.620 = **H3** (current decode slot excluded from its
own selection — `_slot_written` not restored), measured on the `_ds_slot_written` bitmap (61776/61776);
sparse 0.000 = the raw-dot `scorer_norm="off"` lock (Loop-11 dropped the cosine scorer), interacting with H3
(AC-6 2×2 — sparse recovery to ≈0.94 needs both). GOOD gate; not H0/H2; AC-7 moot. Recommendation only.

## Remaining Items
None blocking. Queued non-blocking cleanup (out of scope for the diagnosis loop): plan-workflow terms
(`AC-*`/`H3`) in retained diagnostic comments; reference-selector CUDA-graph safety if these modes are ever
exposed outside `development/loop13`; the `ac4_garbage_counters.py --arm <non-prod>` default-CAPDIR ergonomics
(the ledger already rejects a wrong-source artifact).

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Code-review pass over already-committed diagnosis work; no new implementation, so no new lesson. The
one operational note (an RLCR code-review phase needs a base that is a real ancestor of HEAD — a disjoint base
branch makes `codex review`'s `git merge-base` error) is loop tooling, not a project lesson.
