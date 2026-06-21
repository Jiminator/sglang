# Round 6 Summary

Mainline: **close out AC-6.** Round 5 was ADVANCED; Codex blocked close-out on four items — all
addressed this round, entirely on CPU (no server launched; GPUs idle throughout).

## Work Completed
- **AC-6 corroboration for `ref_cosine_noinc`** (`ac6_corrob_ref_cosine_noinc.py` →
  `evidence/ac6_ref_cosine_noinc_corrob.json`). Replays the REAL `_select_topk_with_optional_current`
  — the ONLY code differing between `ref_cosine` and `ref_cosine_noinc` — on the 4992 captured sparse
  pruning rows, include_current True vs False. **4992/4992**: the include flag swaps **exactly** the
  current decode slot into the selected set (Jaccard **0.999024 = (2048−1)/(2048+1)**, symmetric
  difference **== 2** on every row), and the current slot is **−inf-masked** in every capture (the
  production `_slot_written` exclusion). Ties the selection-level difference to the measured
  0.940→0.625 dense / 0.940→0.313 sparse cost. Fail-closed on zero pruning rows / mechanism violation.
- **Complete per-leg AC-6 bisection matrix** (`ac6_bisection_matrix.py` →
  `evidence/ac6_bisection_matrix.json`) — no blanket "out of scope"; every leg classified:
  - **scorer** (raw-dot↔cosine) — measured (2×2 + materialized-raw selection equality)
  - **current-slot** (incl↔excl) — measured (`ref_cosine_noinc` + the corroboration above)
  - **radix top-k**, **selector width** — retired (AC-2.3, 4992/4992)
  - **head_agg** — not-a-differing-variable (`max` on both paths; cross-TP sum-of-max is AC-2.2)
  - **fp8-absorbed**, **bf16-reduce** — **blocked** with a specific code citation: these live only in the
    production absorbed-latent Triton kernel (`absorbed_latent_kernel.py`, called at
    `deepseek_v2.py:2588/2602`), which implements **only** `scorer_norm="off"`; `config.py:110`
    `_ALLOWED_SCORER_NORM=("off",)` + validation at `:170` hard-reject cosine — testing fp8/reduce
    under cosine needs a new production-path cosine kernel = a selection-path fix (forbidden). Bounded
    second-order (raw-dot exact-fp32 0.013 vs fp8+bf16 0.000 ⇒ ≤~1.3 pp).
- **Blocking evidence-integrity fixes:**
  - `ref_cosine_noinc` measured provenance corrected: `measured_git_sha` now `393966c02` (run HEAD,
    dirty) + a `measured_source` recording the `serve.sh` blob `e1c83e22` that defined the mode
    (committed `c7b66f04b`) — replayable. Was `fea920c06` (where the mode did not exist).
  - `cheap_controls.json`: the stale Round-3 join result (`81/546`, `…=false`) moved out of the
    authoritative `summary` into `superseded_round3_join_summary`; `summary` now carries the
    pruning-valid `4992/4992` AC-2.3 verdict — one machine-readable verdict, no contradiction.
  - `build_ledger.py`: **fails loud** if an AC-6 arm records GSM8K scores but has no corroboration
    artifact on disk (verified: asserts when the artifact is removed). Arm JSON now carries
    `ac6_leg` + `corroboration_artifact` + `measured_source`.
- ROOT_CAUSE.md / findings.md updated with the per-leg matrix + citations (replacing the blanket
  out-of-scope text).

## Files Changed (committed `8b55dfba3`)
- NEW: `development/loop13/ac6_corrob_ref_cosine_noinc.py`,
  `development/loop13/ac6_bisection_matrix.py`,
  `evidence/ac6_ref_cosine_noinc_corrob.json`, `evidence/ac6_bisection_matrix.json`.
- MODIFIED: `build_ledger.py` (measured_source/ac6_leg/corroboration guard), `ROOT_CAUSE.md`,
  `evidence/findings.md`, `evidence/cheap_controls.json`, `evidence/evidence_table.md`,
  `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json`.

## Validation
- `test_reference_selectors.py` → **all 5 pass**.
- `verify_ac2_3.py` (sparse) → 4992/4992, exit 0.
- `ac6_corrob_ref_cosine_noinc.py` → 4992/4992 single-swap, exit 0.
- `ac6_bisection_matrix.py` → 7 legs classified, exit 0.
- `build_ledger.py` → provenance consistent (blob `3757eb5363`); AC-6 guard **asserts (exit 1)** when
  the corroboration artifact is removed, passes when present.
- GPUs idle (0 MiB), no server launched this round. No `.pt`/`.humanize` committed. No fix landed.

## Remaining Items (for AC-8 close-out)
- The two per-leg blockers (fp8-absorbed, bf16-reduce) need review sign-off that the code citation is
  accepted as a valid non-fix-route block.
- AC-2.4 recall-oracle@2048 (the `recall_oracle` instrument is NIAH-only per DEC; GSM8K has no oracle —
  selected-index/current-slot corroboration is the accepted GOOD-branch alternative).
- AC-2.1 `forced_all_assertions.json`; AC-4 per-example sample IDs/order + length-cap garbage counters;
  AC-3.1 captured-row materialized-K; AC-2.2 head-agg `pre_reduce` semantics; AC-8 final writeup.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-corroborate-via-selection-replay, BL-20260621-per-leg-blocker-not-blanket
- Notes: (1) Corroborate a served single-variable bisection arm by replaying the EXACT selection
  function on already-captured score rows instead of a second GPU capture+join — with a fixed-size
  top-k, force-including one position is a SWAP (symdiff==2, Jaccard exactly (k−1)/(k+1)), assert the
  exact value. (2) A planned diagnostic deferred under "no fix" needs a PER-LEG blocker citing the exact
  code path (verified against source this round: config.py:110/170 + the raw-dot-only kernel), not a
  blanket "out of scope"; distinguish `blocked` from `not-a-differing-variable`.

## Goal Tracker Update Request

### Requested Changes:
- Close **Blocking: AC-6 arm lacks corroboration** — `ac6_ref_cosine_noinc_corrob.json` (4992/4992
  single-swap) + `ac6_bisection_matrix.json` + the build_ledger AC-6 guard.
- Close **Blocking: ref_cosine_noinc measured provenance inaccurate** — now records run HEAD
  `393966c02` + serve.sh blob `e1c83e22` (was `fea920c06`).
- Close **Blocking: cheap_controls.json stale AC-2.3 summary** — moved to
  `superseded_round3_join_summary`; `summary` carries 4992/4992.
- Mark **task11 (AC-6)**: corroborated + per-leg matrix complete; only review sign-off on the two
  fp8/bf16 per-leg blockers remains (they are genuinely blocked — config.py:110/170 + raw-dot-only
  kernel — not deferred).
- Plan Evolution Round-6 row added.

### Justification:
Every Round-5-review item now has a concrete generated artifact with code citations, validated and
fail-closed. The two remaining numeric legs are blocked by a real, source-verified two-level lock
(config validation + a raw-dot-only kernel); isolating them under cosine would require a new
production-path cosine kernel, which is a selection-path fix forbidden by the Ultimate Goal's "no fix"
constraint — so they are documented per-leg blockers awaiting sign-off, not silent deferrals.
