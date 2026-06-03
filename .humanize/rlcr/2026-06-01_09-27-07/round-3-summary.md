# Round 3 Summary — Loop 7

## Mainline objective (round-3-contract.md)
Close the AC-3 non-learned selector-variant correctness so task10/task11 can be marked complete: fix the anchor over-budget bug and replace the TP test with a real TP=8, full-matrix, production-logical-path test.

## Outcome: ACHIEVED

## Work Completed
1. **Anchor over-budget bug fixed (Blocking).** `_force_include_anchor` clamped the anchor budget only to `seq_len`, not to the selected count — so with `top_k=3, seq_len=8` a recency `budget=5` forced `[3,4,5]` instead of the most-recent `[5,6,7]`. Now it computes `effective_budget = min(anchor_budget, len(selected))` BEFORE generating anchor positions, so recency/global/strided produce the correct effective set. Verified: `recency b5 → [5,6,7]`, `strided b5 → [0,4,7]`, `recency b10 → [5,6,7]`, `global b5 → [0,1,2]`. Regression tests added for recency/strided `budget > top_k` and `budget ≥ seq_len`.
2. **TP determinism upgraded to the required TP=8 full-matrix, production-path test.** Replaced the partial TP=2 / 5-combo / direct-`compute_token_scores` test with a real **8-rank gloo** test that drives the **production logical path** (`retrieve_topk_via_labels` with `req_pool_indices`/`req_to_token`/`seq_lens` + config threading: head-sharded per-rank scoring → SUM all-reduce → deterministic top-K → anchor force-include). It covers the **full 24-combo matrix** `scorer_norm{off,cosine,hybrid} × head_agg{max,mean} × anchor_mode{off,recency,global,strided}` and asserts identical per-rank `selected_indices`/`valid_lengths`. Fail-fast (`DoubleSparsityTPMisconfigured`/`DoubleSparsityRebindError`) preserved.

## Files Changed
`selection_kernel.py` (anchor effective-budget clamp), `test_scorer_variants.py` (over-budget regressions), `test_ds_scorer_tp_determinism.py` (rewritten TP=8 logical-path full-matrix). Commit `fc8871372`.

## Validation
- **323 DS unit tests pass** + the **TP=8 24-combo gloo matrix** test (21 s, all ranks identical for every combo).
- Anchor reproduction verified against the Round-2 review counterexamples.

## Remaining Items (queued, justified)
- **Graph-safe Triton scorer/head/anchor port + full AC-3 measurement matrix** (task #13): heavy kernel + GPU work (MMLU, dense-DS, N≥50, DSA same-node, eager-vs-graph perf, per-variant attribution). The variants are *correct* and *safe* (Round-2 startup guard); the production port + binding matrix is the next round. AC-3 *measurement* closure depends on it.
- **Oracle fail-closed + 64K re-run** (task #12, AC-1/AC-2): the most-deferred item; gets its own focused round next (config-borne activation so it records on TP workers + fail-closed semantics + re-run).
- **Tier-2.A / AC-4** (task13–17), **M4 consolidation / AC-6 perf + final decision record** (task19–20): sequenced after the selector + oracle are binding.
- **Plan-marker code/comment cleanup**: pre-merge; queued.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: no new reusable engineering pitfall (the anchor bug and TP-test-coverage gap are round-specific defects, captured in the commit + this summary).

## Goal Tracker Update Request

### Requested Changes
- **Resolve Blocking Side Issue "anchor force-include mishandles `anchor_budget > top_k`"**: fixed (`effective_budget = min(anchor_budget, len(selected))` before anchor generation; reproduction verified; regression tests added).
- **task10 (anchor-budget ablation)** → **implemented + unit-tested**: full `anchor_mode {off,recency,global,strided}` with the over-budget correctness fix; per-variant *measurement* still part of the AC-3 matrix (task #13).
- **task11 (TP determinism)** → **implemented + tested**: TP=8 gloo test through the production logical path over the full `scorer_norm × head_agg × anchor_mode` matrix (24 combos), cross-rank-identical. (CPU gloo TP=8 equality; an 8×H200 hardware TP=8 artifact can be added during the task #13 measurement run.)
- **Keep Active**: task #13 (graph-safe Triton port + full AC-3 measurement matrix) and task #12 (oracle fail-closed + 64K) as the next rounds' mainline; AC-4 (task13–17) and M4 (task19–20) sequenced after.

### Justification
Round 3 closed the two AC-3 selector-variant correctness gaps the Round-2 review rejected (the anchor over-budget bug and the partial TP test), so the variants are now correct and have TP=8 full-matrix cross-rank determinism evidence. The remaining work is the heavy graph-safe Triton port + binding measurement matrix and the oracle fail-closed hardening, sequenced as the next rounds' mainline to converge the loop on binding closure.
