# Round 3 Contract

## Mainline Objective
**Close the AC-3 non-learned selector-variant correctness** so task10 and task11 can be marked complete: (1) fix the anchor over-budget force-include bug, and (2) replace the TP determinism test with a real TP=8, full-matrix, production-logical-path test.

## Target ACs (1–2)
- **AC-3** (primary): variant correctness + TP=8 cross-rank selected-index equality across the full `scorer_norm × head_agg × anchor_mode` matrix.

## Blocking Side Issues In Scope (must fix this round)
- **Anchor force-include mishandles `anchor_budget > top_k`** (Codex Round-2 blocking). `_force_include_anchor` clamps the budget only to `seq_len`, not to the selected count, so e.g. `top_k=3, seq_len=8, recency, budget=5` forces `[3,4,5]` instead of the most-recent `[5,6,7]`. Fix: compute `effective_budget = min(anchor_budget, len(selected_real))` (then `_anchor_positions` clamps to `seq_len`) and generate anchors from the effective budget.

## Queued Side Issues Out Of Scope (justified)
- **Graph-safe Triton scorer/head/anchor port + full AC-3 measurement matrix** (task #13): heavy kernel + GPU work (MMLU, dense-DS, N≥50, DSA, eager-vs-graph perf). The Round-2 startup guard makes non-default variants *safe* (reject under graph); the production port + binding matrix is the next round. AC-3 *measurement* closure depends on it.
- **Oracle fail-closed + 64K re-run** (task #12, AC-1/AC-2): M0 diagnostic hardening + GPU re-run; does not block AC-3 selector correctness. Next round (this is the most-deferred item; it gets its own focused round after AC-3 correctness closes).
- **Tier-2.A / AC-4** (task13–17), **M4 consolidation / AC-6 perf + final decision record** (task19–20): separate milestones; sequenced after the selector + oracle are binding.
- **Plan-marker code/comment cleanup**: pre-merge; queued.

## Round Success Criteria
- `_force_include_anchor` clamps the effective anchor budget to the selected count before generating anchor positions; for `top_k=3, seq_len=8`: recency(budget≥3) ⇒ `{5,6,7}`, global ⇒ `{0,1,2}`, strided ⇒ 3 evenly-spaced over `[0,8)`. Regression tests for recency + strided with `anchor_budget > top_k` and `anchor_budget ≥ seq_len`.
- TP determinism test uses **`WORLD=8`** gloo ranks and the **production logical path** (`retrieve_topk_via_labels` with `req_pool_indices`/`req_to_token`/`seq_lens` + config threading + anchor application), covering the **full matrix** `scorer_norm={off,cosine,hybrid} × head_agg={max,mean} × anchor_mode={off,recency,global,strided}` (24 combos), asserting identical per-rank `selected_indices`/`valid_lengths`. `DoubleSparsityTPMisconfigured`/`DoubleSparsityRebindError` fail-fast preserved.
- All DS unit tests pass. The round makes no AC-3 *measurement* closure claim (Triton port + matrix pending).
