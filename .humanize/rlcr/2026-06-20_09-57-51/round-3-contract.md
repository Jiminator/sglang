# Round 3 Contract

## Mainline Objective
Make the **AC-2.3 / AC-2.2 captured cheap-controls VALID** (Codex Round-2 mainline gap #1 / required-plan
step 1): give the capture records stable row identity so score rows join to selection rows on exact
`(req_pool_index, layer, decode_step)`, fix `analyze_captures.py` to do exact joins (fail on any
unmatched selected row), and regenerate `cheap_controls.json` with a VALID radix-vs-`torch.topk`
selected-index equivalence + selector-width `[5120]`-vs-`[]` equivalence (and a TP head-agg result if
the `pre_reduce_scores` semantics is confirmed). Replace the PRELIMINARY/cartesian artifact.

## Target ACs
- **AC-2.3** (radix top-k == `torch.topk`; width `[5120]` vs `[]` identical — retire/confirm the
  contradicted suspects on real captured rows) + **AC-2.2** (TP head-agg micro-test, if `pre_reduce_scores`
  is confirmed; else recorded as needing the pre-reduce semantics).

## Blocking Side Issues In Scope (truly block the mainline)
- `selection_capture` records lack `req_pool_indices`, so score↔selection rows cannot be exact-joined.
  Add `req_pool_indices` to the record (guarded diagnostic instrumentation; default-off capture flag,
  no production behavior change).
- `score_capture` has no decode-step in its key (filename `rank_req_layer` overwrites across steps).
  Drive captures with `max_new_tokens=1` single bs=1 requests so each (req, layer) is exactly one
  decode step — clean alignment without changing the score-capture filename schema.
- **Ledger SHA blocking fix (Codex blocking #1):** `build_ledger.py` stamps generation HEAD onto every
  arm. Store `measured_git_sha` (per arm, the SHA the run happened at) separately from
  `ledger_generated_git_sha`; regenerate.

## Queued Side Issues (documented, OUT OF SCOPE this round)
- **AC-6 production-path one-variable bisection** via a guarded diagnostic production-style cosine mode
  (Codex unblocked this: guarded diagnostic modes are allowed). This is the largest remaining item and
  the explicit NEXT-round mainline; the reference-ceiling cliff (cosine 0.940 vs raw-dot 0.013 + the
  materialized-raw selection-equality proof) + opts-second-order bound already name the candidate.
- **AC-2.1 forced-all physical-slot assertions + AC-4 per-step garbage counters** (adapter
  instrumentation) — next round.
- **AC-3.1 on captured rows** (offline materialized-`K_label` row-by-row vs absorbed raw-dot) — next
  round (the algebraic identity is already proven; the captured-row version uses this round's aligned
  captures).
- Fail-closed ledger with sample IDs/order + all serial cells — next round (needs GSM8K harness
  sample-ID persistence + serial reruns).

## Round Success Criteria
1. `selection_capture` record carries `req_pool_indices`; a default-off flag so production is unchanged.
2. Bounded captures (a few bs=1 dense + sparse requests, `max_new_tokens=1`) produce score+selection
   dumps that `analyze_captures.py` joins on exact `(req_pool_index, layer, decode_step)`, failing if any
   selected row has no matching score row.
3. `cheap_controls.json` regenerated with: radix-vs-`torch.topk` selected-index equality per joined row
   (identical / Jaccard), and width `[5120]`-vs-`[]` equivalence (two capture configs), with a clear
   pass/fail and the PRELIMINARY caveat removed for the now-valid results.
4. `build_ledger.py` records `measured_git_sha` per arm separate from `ledger_generated_git_sha`; ledger
   + table regenerated; no stale/ambiguous SHA.
5. goal-tracker mutable section updated; round-3-summary written with BitLesson Delta; committed;
   one TP=8 server at a time.
