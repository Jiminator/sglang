# Round 4 Contract

## Mainline Objective
Deliver a **VALID AC-2.3 captured cheap-control** (the Round-3 mainline that stalled): stamp a shared
per-decode-forward `decode_step_id` into BOTH `score_capture` and `selection_capture`, key score-capture
files/records by `(req_pool_index, layer_id, decode_step_id)` so rows never overwrite, make
`analyze_captures.py` join on that exact triple and **fail loud** (nonzero exit) on any
unmatched/duplicate/zero-row condition, and produce radix-vs-`torch.topk` AND selector-width `[5120]`-vs-`[]`
selected-index equivalence on exact shared keys. Fix the two real bugs Codex found (fail-open analyzer;
stale ledger generated-SHA).

## Target ACs
- **AC-2.3** (radix top-k == `torch.topk`; width `[5120]` vs `[]` identical — retire/promote the suspects on
  exactly-keyed captured rows) — the stalled Round-3 mainline.

## Blocking Side Issues In Scope (truly block the mainline)
- `score_capture` filename/record has no `decode_step_id`; `selection_capture` uses an independent step
  counter — so score↔selection rows can't be paired at the same decode step. Add a shared
  `forward_batch._ds_capture_step` (stamped once per decode forward, default-off path) read by both.
  Guarded diagnostic instrumentation; production behavior unchanged when capture flags are off.
- `analyze_captures.py` is fail-OPEN (rc=0 with zero score groups / on unmatched rows). Make it exit
  nonzero on: any selected row lacking `req_pool_indices`, any unmatched score row, any duplicate score
  key, or zero equivalence rows. Matched-subset output is debug-only, not AC evidence.
- Ledger generated-SHA is stale (`ac479aeb3` in committed artifacts; HEAD `29ed825fa`) because
  `build_ledger.py` reads HEAD from a dirty pre-commit worktree. Record the generator-file blob hash +
  an explicit "generated from working tree at HEAD <sha> (+uncommitted)" marker so the source is
  unambiguous; fix `run_meta.json` consistently; use full SHAs.

## Queued Side Issues (documented, OUT OF SCOPE this round)
- **AC-6 production-path one-variable bisection** (guarded diagnostic production-style cosine + per-variable
  arms) — the largest substantive gap; explicit NEXT mainline. Reference-ceiling cliff already names the
  candidate.
- **AC-2.1 forced-all physical-slot assertions + AC-4 garbage counters + sample IDs/order + AC-3.1
  captured-row proof** — adapter/harness instrumentation; subsequent rounds (the AC-3.1 captured-row proof
  reuses this round's shared-step capture keys).
- **AC-2.2 head-agg semantics** — needs `score_capture` to dump explicitly-named pre-reduce / post-reduce /
  post-mask rows; do alongside AC-2.3 if cheap, else next round.
- Plan-term comment cleanup in remaining files; reference-mode CUDA-graph fail-closed.

## Round Success Criteria
1. A shared `decode_step_id` (per decode forward) is stamped on `forward_batch` and written into BOTH
   capture records; `score_capture` files are keyed by `(req_pool_index, layer_id, decode_step_id)` with no
   overwrite. Default-off (capture flags); production byte-identical when off.
2. `analyze_captures.py` joins on exact `(req_pool_index, layer_id, decode_step_id)` and EXITS NONZERO on
   unmatched/duplicate/zero rows. Verified: it fails (rc!=0) on an empty capture dir.
3. `cheap_controls.json` regenerated with a VALID radix-vs-`torch.topk` selected-index equivalence (identical
   per row, with counts) AND a selector-width `[5120]`-vs-`[]` equivalence (two capture configs), with no
   PRELIMINARY/INCONCLUSIVE caveat on the AC-2.3 result (or, if a real discrepancy is found, it is reported
   as a promoted suspect with numbers — not a measurement artifact).
4. `build_ledger.py` records an unambiguous generator source (file blob hash + worktree-at-HEAD marker);
   `run_meta.json` consistent; full SHAs.
5. goal-tracker mutable section updated; round-4-summary written with BitLesson Delta; committed; one TP=8
   server at a time.
