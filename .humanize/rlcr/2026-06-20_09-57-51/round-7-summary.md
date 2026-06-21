# Round 7 Summary

Mainline: **finish the AC-6 bisection matrix honestly.** Round 6 was ADVANCED; Codex flagged two correct
gaps — a runnable leg wrongly marked "blocked", and a sparse-only corroboration used for the dense cost.
Both fixed and measured on the actual workload.

## Work Completed
- **Score-reduce leg now MEASURED** (was wrongly "blocked"). `serve.sh ds_reduce_fp32` = the production
  `ds` config + `score_reduce_dtype="fp32"` (same graph mode; the **only** variable vs `production_ds`;
  config accepts `{fp32,bf16}` — a runnable route, not a fix). GSM8K: **dense 0.620 / sparse 0.000 —
  identical to production_ds** ⇒ the reduce dtype is **not** a culprit. Selection-level corroboration
  (`ac6_score_reduce_corrob.py` → `ac6_score_reduce_fp32_corrob.json`): reduce the SAME captured per-rank
  `pre_reduce_scores` (validated `sum(pre)==post` **702/702**) in bf16 vs fp32 → median selected-set
  **Jaccard 0.998** (127/702 identical); only bottom-of-top-k near-ties reshuffle.
- **Dense current-slot corroboration added** (Codex gap #2). `ac6_corrob_ref_cosine_noinc.py` restructured
  into regime sections with **distinct** invariants, on real captures:
  - **sparse** (seq_len>top_k) **4992/4992** — full top-k, include SWAPS the current slot in (symdiff==2,
    Jaccard (k−1)/(k+1)).
  - **dense** (seq_len≤top_k) **3744/3744** — room for all, exclude DROPS the current slot
    (`valid_length==seq_len-1`), include ADDS it (`valid_length==seq_len`, symdiff==1, no eviction).
  Real dense captures (seq_len ~790) came from a separate eager run.
- **fp8-absorbed leg re-verified as the only blocker** with a precise, source-checked citation: no
  production config flag toggles fp8-vs-fp32 absorbed scoring (the graph selector scores the fp8 resident
  latent in-register, `deepseek_v2.py:2602`→`absorbed_latent_kernel.py`); exact-fp32 absorbed exists only
  on the `reference_*` path, which bundles current-slot/TF32/radix/width/reduce (no single-variable
  isolation). Bounded second-order (≤~1.3 pp) now reduce is measured and radix/width retired.
- **Matrix + ledger + generated text reconciled:** `ac6_bisection_matrix.json` legs
  measured[2,3,7]/retired[4,5]/not-a-difference[1]/blocked[6=fp8 only]; `build_ledger.py` adds the
  `ds_reduce_fp32` arm (measured_source + ac6_leg + corroboration), and the AC-6 guard now protects BOTH
  AC-6 arms (verified: asserts when either corroboration artifact is missing); the generated
  `evidence_table.md` verdict text no longer says fp8/bf16-reduce/head_agg are out of scope
  (**0 occurrences**). `cheap_controls.json._status` pointer fixed to `superseded_round3_join_summary`.

## Files Changed (committed `8281361e7`)
- NEW: `development/loop13/ac6_score_reduce_corrob.py`, `evidence/ac6_score_reduce_fp32_corrob.json`.
- MODIFIED: `serve.sh` (ds_reduce_fp32 mode), `build_ledger.py` (arm + verdict text), `ac6_bisection_matrix.py`,
  `ac6_corrob_ref_cosine_noinc.py` (dual-regime), `ROOT_CAUSE.md`, `evidence/findings.md`,
  `evidence/cheap_controls.json`, `evidence/ac6_bisection_matrix.json`, `evidence/ac6_ref_cosine_noinc_corrob.json`,
  `evidence/evidence_table.md`, `evidence/meta/run_meta.json`, `evidence/meta/arms/*.json`, `.gitignore`.

## Validation
- `test_reference_selectors.py` → **all 5 pass**.
- `verify_ac2_3.py` (sparse) → 4992/4992, exit 0.
- `ac6_corrob_ref_cosine_noinc.py` → sparse 4992/4992 + dense 3744/3744, exit 0.
- `ac6_score_reduce_corrob.py` → 702 groups, sum==post 702/702, median Jaccard 0.998, exit 0.
- `ac6_bisection_matrix.py` → measured[2,3,7]/retired[4,5]/not-a-difference[1]/blocked[6], exit 0.
- `build_ledger.py` → provenance consistent (blob `1280fa0339`); AC-6 guard **asserts** when either
  corroboration artifact is removed.
- `evidence_table.md` → **0** "out of scope"/"Untested numeric" occurrences; `ds_reduce_fp32` row 0.620/0.000.
- Discipline: one TP=8 server at a time (eager capture run + graph measurement run, each torn down to
  0 MiB). No `.pt`/`.humanize` committed. No selection/adapter fix landed.

## Remaining Items (for AC-8 close-out)
- fp8-absorbed per-leg blocker awaits review sign-off (the only un-measured AC-6 leg; genuinely blocked).
- AC-2.4 recall-oracle@2048 (NIAH-only instrument; GSM8K has no oracle).
- AC-2.1 `forced_all_assertions.json`; AC-3.1 captured-row materialized-K; AC-2.2 head-agg semantics
  (R7 note: `sum(pre_reduce)==post` 702/702 confirms the reduce, but AC-2.2's SUM-vs-global-max question
  is separate); AC-4 sample IDs/order + garbage counters; AC-8 final writeup.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-corroboration-cardinality-dependent
- Notes: Added a lesson that selected-index corroboration invariants are cardinality-dependent — a
  fixed-size top-k SWAP (sparse, seq_len>top_k, symdiff==2) is a different mechanism than a pure ADD
  (dense, seq_len≤top_k, symdiff==1, valid_length seq_len-1→seq_len); corroborate each regime with its own
  invariant, do not reuse one for the other. Also UPDATED BL-20260621-per-leg-blocker-not-blanket with the
  R7 refinement: before declaring a leg `blocked`, verify no existing CONFIG ROUTE can test it on the same
  servable path (the R6 reduce leg was wrongly blocked because `score_reduce_dtype="fp32"` is a runnable
  toggle on the raw-dot path); run the route first, block only when none exists (e.g. fp8-absorbed).

## Goal Tracker Update Request

### Requested Changes:
- Close **R6 mainline gap: reduce leg wrongly blocked** — `ds_reduce_fp32` measured (0.620/0.000 =
  production; corrob median Jaccard 0.998); matrix leg 7 = measured.
- Close **R6 mainline gap: sparse-only current-slot corroboration** — dense regime added (3744/3744,
  symdiff==1, valid_length invariant), distinct from the sparse swap.
- Close **R6 blocking: contradictory generated evidence** — `evidence_table.md` regenerated (0 "out of
  scope"); `cheap_controls.json._status` pointer fixed.
- Mark **task11 (AC-6)**: all legs measured/retired/not-a-difference except fp8-absorbed, which carries an
  accepted per-leg blocker (no production config for fp32 absorbed scoring) awaiting review sign-off.
- Plan Evolution Round-7 row added.

### Justification:
Both R6-review mainline gaps are now measured/corroborated on the actual workload with the correct
per-regime invariants. The reduce result (fp32 reduce = production 0.620/0.000, selection Jaccard 0.998)
is decisive that the reduce dtype is innocent. The sole remaining AC-6 leg (fp8-absorbed) has no
production config route — exact-fp32 absorbed scoring exists only on the multi-variable reference path,
so isolating it would require a new production kernel (a fix forbidden by the Ultimate Goal) — making it a
genuine documented per-leg blocker, not a deferral.
