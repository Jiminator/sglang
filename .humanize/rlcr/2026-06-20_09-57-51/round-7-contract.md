# Round 7 Contract

Round 6 was ADVANCED. Codex accepts the provenance/cheap-controls/sparse-corroboration progress but
blocks AC-6 close-out on two correct findings: (1) the BF16 score-reduce leg was wrongly marked
"blocked" when the production raw-dot selector accepts `score_reduce_dtype="fp32"` (a runnable config
route, not a fix); (2) the current-slot corroboration is sparse-only (fixed-size swap, symdiff==2) but
the writeup uses it for the DENSE 0.940→0.625 cost too (a different cardinality case).

## Mainline Objective (exactly one)
**Finish the AC-6 bisection matrix honestly:** run the `score_reduce_dtype="fp32"` leg as a MEASURED
single-variable arm (it is runnable), add a DENSE-regime current-slot corroboration (distinct invariant
from the sparse swap), and re-verify the one remaining `blocked` leg (fp8-absorbed) with a precise,
source-checked citation — so every AC-6 leg is measured/retired/not-a-difference/blocked-with-proof and
the generated evidence text agrees with the matrix.

## Target ACs
- **AC-6** (primary): leg 7 (reduce) measured + corroborated; dense current-slot corroborated; leg 6
  (fp8) blocker re-verified; matrix + generated text consistent.
- **AC-4** (secondary): the new `ds_reduce_fp32` arm in the ledger with full metadata + corroboration.

## Blocking Side Issues (truly block AC-6 close-out)
- **`evidence_table.md` generated verdict text contradicts the new matrix** — it still says
  "Untested numeric legs (fp8/bf16-reduce/head_agg) need a production-path cosine kernel = out of scope"
  (`build_ledger.py:197`). Must be regenerated: head_agg is not-a-differing-variable; bf16-reduce is
  MEASURED; only fp8-absorbed is blocked (different reason).
- **`cheap_controls.json._status` stale sentence** still says "The 81/546 in `summary` here is the OLD…"
  but the old data now lives under `superseded_round3_join_summary`. Fix the pointer.

## Queued Side Issues (documented, OUT OF SCOPE this round)
- AC-2.4 recall-oracle (NIAH-only); AC-2.1 `forced_all_assertions.json`; AC-2.2 head-agg `pre_reduce`
  semantics (note: R7 confirms `sum(pre_reduce)≈post-reduce` 624/624, which helps but AC-2.2's
  SUM-vs-global-max question is separate); AC-3.1 captured-row materialized-K; AC-4 sample IDs/order +
  garbage counters; AC-8 final writeup; plan-term comment cleanup; reference-mode fail-closed.

## Approach
- **Reduce leg (measured).** Add `serve.sh ds_reduce_fp32` = production `ds` config + `"score_reduce_dtype":
  "fp32"` (config-only; no selection/adapter fix). Run dense + sparse GSM8K (eager, with score_capture
  on, so the same server yields real dense-regime captures for the current-slot corroboration). Expect
  ≈ production (0.620/0.000) since reduce dtype is near-selection-neutral.
- **Reduce corroboration (offline, no extra run).** From the existing `.sglang_ds_scorecap_sparse`
  per-rank `pre_reduce_scores` (validated: `sum(pre_reduce)≈post-reduce` on 624/624 rows), reduce the
  cross-TP SUM in bf16 vs fp32 and compare selected sets → median Jaccard 0.998 (49/624 identical).
  Persist `evidence/ac6_score_reduce_fp32_corrob.json` (selected-set Jaccard min/median, identical-row
  count, sum≈post validation, row counts). Fail-closed on zero rows.
- **Dense current-slot corroboration.** Replay `_select_topk_with_optional_current` on real DENSE
  captures (seq_len ≤ top_k, captured this round). DISTINCT invariant from sparse: exclude → current
  masked, `valid_length == seq_len-1`; include → current force-added, `valid_length == seq_len`;
  selected-set delta == {current} (symdiff == 1, NOT the sparse symdiff==2 swap — there is room, no
  eviction). Add a `dense` regime section to `ac6_ref_cosine_noinc_corrob.json` (and scope the existing
  result as `sparse`).
- **fp8-absorbed leg (re-verified block).** Confirmed this round: config exposes no fp8-vs-fp32 absorbed
  scoring flag (`config.py` has `scorer_norm`, `head_agg`, `score_reduce_dtype`, `selector_width`,
  `anchor`, `selector_impl` — none toggles absorbed precision); the production graph path scores the
  fp8 resident latent in-register, and exact-fp32 absorbed scoring exists ONLY on the `reference_*`
  path, which bundles other changes (no single-variable isolation). Stays `blocked` with that citation +
  a tighter second-order bound now that reduce is measured.

## Concrete Success Criteria
1. `serve.sh ds_reduce_fp32` exists (config-only diff vs `ds`); `ds_reduce_fp32_dense.out` +
   `ds_reduce_fp32_sparse.out` persisted with real GSM8K scores; one TP=8 server at a time, torn down.
2. `evidence/ac6_score_reduce_fp32_corrob.json`: bf16-vs-fp32 reduce selected-set overlap (Jaccard
   min/median + identical count) on the captured rows + the `sum(pre_reduce)≈post-reduce` validation;
   fail-closed on zero rows.
3. `ac6_ref_cosine_noinc_corrob.json` has a DENSE section using the seq_len≤top_k invariant (symdiff==1,
   valid_length seq_len-1→seq_len), distinct from the sparse swap; the sparse result is labeled sparse.
4. `ac6_bisection_matrix.json`: leg 7 (reduce) = **measured** (ds_reduce_fp32 scores + corrob); leg 6
   (fp8) = **blocked** with the re-verified citation; leg 1 (head_agg) = not-a-differing-variable. No leg
   left as a contradiction.
5. `evidence_table.md` generated verdict text no longer calls bf16-reduce/head_agg "out of scope"; it
   reflects the matrix. `cheap_controls.json._status` points at `superseded_round3_join_summary`.
6. `build_ledger.py` includes `ds_reduce_fp32` (AC-6 leg + corroboration_artifact + measured_source) and
   still fails closed on an uncorroborated AC-6 arm; provenance consistent. Tests pass. Commit;
   round-7-summary with BitLesson Delta + Goal Tracker Update Request. No fix landed; no exit by lying /
   editing loop state / cancel-rlcr-loop.
