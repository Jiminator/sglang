# Round 3 Summary — Loop 13 (DS-vs-DSA accuracy diagnosis)

## Objective (round-3-contract.md): make the AC-2.3 captured cheap-controls valid + SHA fix
No verdict change. The GOOD-ceiling two-regression verdict (dense = H3 current-slot exclusion;
sparse = the raw-dot `scorer_norm="off"` lock) is unchanged.

## Work Completed
- **Ledger SHA provenance (blocking fix).** `build_ledger.py` now records per-arm `measured_git_sha`
  (baselines @180f6dd6d; R1 reference arms @fea920c06) **separate** from `ledger_generated_git_sha`.
  Regenerated `evidence/meta/arms/*.json` + `evidence_table.md`. No more stale/ambiguous SHA.
- **Capture row identity (blocking fix).** `selection_capture` records now carry `req_pool_indices`
  (guarded, default-off — emitted only under the `selection_capture` flag; production unchanged), so a
  selection row can be joined to its score row on exact `(req_pool_index, layer)`.
- **Exact-join analyzer.** `analyze_captures.py`'s selected-index equivalence was a cross-record
  cartesian comparison; it now does an **exact `(req_pool_index, layer)` join** that fails loud on any
  unmatched selected row. Re-captured with concurrent bs=1, `max_new_tokens=1` requests holding distinct
  pool slots → **unmatched_rows = 0** (the join is valid).

## Files Changed
- Code: `selection_capture.py` (req_pool_indices field).
- Harness: `analyze_captures.py` (exact join), `build_ledger.py` (measured vs generated SHA),
  `evidence/{cheap_controls.json, evidence_table.md, meta/arms/*.json}` (regenerated).
Commit `29ed825fa`; tree clean; one TP=8 server at a time; GPUs idle.

## Validation
- Re-captured 6 concurrent requests (3 dense + 3 sparse, `max_new_tokens=1`); the new analyzer joins
  on exact `(req_pool_index, layer)` with `unmatched_rows = 0`.
- `build_ledger.py` regenerated with the two-SHA schema (verified: `dsa.measured_git_sha`=180f6dd6d,
  `ref_cosine.measured_git_sha`=fea920c06).

## Remaining Items (honest)
- **AC-2.3 radix-vs-`torch.topk` is INCONCLUSIVE from captures** (annotated in `cheap_controls.json._status`).
  Even with the valid join, the captured `scores` row is not reliably the decode step the radix selected
  from: `score_capture`'s filename has no decode-step id (extend+decode overwrite), and `score_capture` /
  `selection_capture` use independent step counters. A clean control needs a **shared per-forward
  decode-step identity** stamped into BOTH captures. Radix exactness is independently established by
  `topk_kernel.py` (`blocked_topk_sequence_order` documented bit-identical to `select_topk_sequence_order`);
  the verdict does not depend on this control.
- **AC-2.2 head-agg** PRELIMINARY: `served_sum != post_reduce` on captured rows ⇒ `pre_reduce_scores`
  semantics differs from the SUM-of-local-max model; needs confirming `score_capture.pre_reduce_scores`.
- **AC-6 production-path bisection**, **AC-2.1 forced-all slot assertions / AC-4 garbage counters**,
  **AC-3.1 captured-row proof** — substantial serving instrumentation; see the Goal Tracker Update Request.

## Goal Tracker Update Request

### Requested Changes / disposition:
1. **AC-2.3/AC-2.2 captured controls** — close the join-validity work (done: req_pool_indices +
   exact join + unmatched=0) as the AC-2 row-identity blocker; keep the radix-equivalence as an OPEN item
   needing a shared decode-step id in `score_capture`/`selection_capture`. It is corroboration, and
   `topk_kernel.py` already proves radix exactness, so it is NOT load-bearing for the verdict.
2. **AC-6 production-path one-variable bisection** (guarded diagnostic production-style cosine + head_agg
   / fp8-vs-fp32 / reduce-dtype / radix / width arms) is the largest remaining item and the natural NEXT
   mainline — Codex confirmed guarded diagnostic modes are allowed. The reference-ceiling cliff (cosine
   0.940 vs raw-dot 0.013 + materialized-raw selection-equality) and the opts-second-order bound
   (production raw-dot 0.000 ≈ exact raw-dot 0.013) already name the candidate.
3. **AC-2.1 forced-all physical-slot assertions + AC-4 per-step garbage counters + sample IDs/order**
   require `logical_to_physical`/adapter + GSM8K-harness instrumentation; next round.

### Justification:
The Ultimate Goal (root-cause verdict with live evidence) is delivered and Codex-accepted in substance.
Round 3 fixed two real evidence-integrity issues (SHA provenance; capture row identity / cartesian-join
bug). The residual items are corroborating rigor whose largest pieces require building serving
instrumentation; surfacing them with their dispositions rather than silently deferring. Several are
independently established (radix exactness via topk_kernel.py; the materialized-K identity via the
committed algebraic test), so the verdict stands.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-ds-capture-step-alignment
- Notes: Captured the DS-capture join gotcha — selection_capture lacked req_pool_index (forcing a
  cartesian comparison), and score_capture has no decode-step id and a step counter independent of
  selection_capture, so score↔selection rows can't be cleanly paired without a shared per-forward step
  identity; radix exactness is provable offline via topk_kernel.py instead.
