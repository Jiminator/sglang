# Round 3 Review Result

Mainline Progress Verdict: ADVANCED

Claude advanced the Round-3 contract. The anchor over-budget bug is fixed, and the TP determinism test now covers the requested TP=8 full scorer/head/anchor matrix through the logical `retrieve_topk_via_labels` path. I did not find a new high-signal bug in the three files changed by `fc8871372`.

However, the original Loop 7 plan is still not complete. Claude's queued items are plan-derived acceptance work, not optional cleanup. Do not stop the loop and do not treat AC-3 or the full Loop 7 goal as closed.

## Accepted Round-3 Fixes

1. **AC-3/task10: anchor over-budget semantics are now correct.**

   Evidence: `_force_include_anchor()` now builds `real` before anchor generation and clamps `effective_budget = min(anchor_budget, len(real))` before calling `_anchor_positions()` (`python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:789-797`). The regression tests cover the Round-2 counterexamples: recency `budget > top_k` forces `[5,6,7]`, strided `budget > top_k` forces `[0,4,7]`, and recency `budget >= seq_len` remains bounded by the selected count (`test/registered/unit/layers/attention/test_scorer_variants.py:124-139`).

2. **AC-3/task11: the TP test now matches the requested shape.**

   Evidence: `WORLD = 8` (`test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py:25`), and each worker iterates the full `scorer_norm={off,cosine,hybrid} × head_agg={max,mean} × anchor_mode={off,recency,global,strided}` matrix through logical `retrieve_topk_via_labels` with `req_pool_indices`, `req_to_token`, `seq_lens`, `hybrid_threshold`, and anchor parameters (`:75-96`). The test asserts every rank reports identical `selected_indices`/`valid_lengths` for every combo (`:102-120`), and the fail-fast guard test is still present (`:123-135`).

## Mainline Gaps

1. **AC-1/AC-2 remain incomplete: oracle fail-closed behavior, oracle-off graph allocation evidence, and binding 64K oracle evidence are still missing.**

   Evidence: `_maybe_record_recall_oracle()` still returns silently when no active trial exists (`selection_kernel.py:978-981`), filters the harness-provided needle span instead of rejecting invalid positions (`:987-991`), and swallows all payload/sink exceptions (`:1010-1012`). The tracker still has task3/task4/task6/task7 active for these reasons.

   Required implementation plan: make oracle-enabled mode strict. Validate that an active trial exists and that every harness-provided needle position is in range; remove span filtering; emit explicit failure artifacts keyed by request/trial/layer/decode-step before aborting a trial; add expected-record-count assertions to the oracle sweep; run the CUDA allocation detector under graph replay for oracle-off; then re-run 4K/16K/64K oracle sweeps before treating task7 attribution as binding.

2. **AC-3/task8/task12 are still not production-ready: the selector variants remain eager/safety-gated and lack the full non-regression measurement matrix.**

   Evidence: `validate_double_sparsity()` rejects any non-default scorer variant while CUDA graph is enabled (`validator.py:96-115`), and `DeepseekV2AttentionMLA.forward` forces non-default variants to the eager selector because the graph-safe Triton scorer only supports the raw/max/no-anchor path (`deepseek_v2.py:2235-2248`). This is a correct safety stopgap, but it is not the graph-safe production selector required by AC-3. The NIAH/MMLU/dense/within-budget/DSA same-node/perf matrix is still absent.

   Required implementation plan: port the selected non-learned variants into `retrieve_topk_graph_safe` and the Triton logical scorer. Add graph-safe parameters for scorer normalization, head aggregation, hybrid threshold, anchor mode, and anchor budget; implement anchor force-include without capture-time allocation or host sync; add eager-vs-graph equality and zero-allocation replay tests; remove the eager-only route only for combinations the graph-safe path supports. Then run task12: DS baseline, DSA same-node reference, each independent variant, 1K/1.5K/4K/16K/64K NIAH with N>=50 for binding 16K, MMLU at mem0.7, dense-DS, within-budget parity, and eager-vs-graph perf deltas.

3. **AC-4/task13-task17 are still unimplemented despite the 4K oracle gate.**

   Evidence: repository search still finds no implementation of `enable_lifted_budget_decode` or `lifted_budget_top_k` outside plan/review text, and there is no compact-domain remap path, lifted sparse decode path, correctness suite, or landing/disposition record.

   Required implementation plan: execute task13-task17 as the next AC-4 workstream. First produce the Codex/analyze ABI design for `enable_lifted_budget_decode` and `lifted_budget_top_k`. Then add the config fields and validators so `top_k > index_topk` is rejected unless the explicit opt-in backend path is selected. Implement physical slot -> `page_table_1_flattened` -> compact index remap, mask or safe-replace `-1` before dequant, use fixed `lifted_budget_top_k` with padding, preserve the R23 tie-break, and add reference-attention, prefix-sharing, padding, duplicate-index, valid-length, graph-replay allocation, and TP equality tests. Finish with the task17 production-ready-or-disposition record.

4. **AC-6/task19-task20 remain missing.**

   Evidence: no Round-3 artifact records conc-1/16 TTFT, decode TPS/req, GPU memory, graph replay success, admission, or Tier-1 spine non-regression for the variants or a lifted-budget path. The final strategic-gate supersession decision record is still not written as the plan's closing artifact.

   Required implementation plan: after task12 and task17 are complete, run the conc-1/16 guardrail suite at the Loop-7 op-point, write the consolidated DS-vs-DSA recall/perf/non-regression report, and then write the final decision record that supersedes the strategic gate with corrected M0 evidence and the final Tier-2.A disposition.

## Blocking Side Issues

1. **Oracle hook fail-open remains blocking for AC-1/AC-2.**

   This is the same unresolved issue from prior reviews. It blocks binding attribution because missing or invalid oracle records can still disappear silently. Fix it before relying on the M0 oracle result for final AC-2/AC-4 decisions.

## Queued Side Issues

1. **Plan-specific workflow markers remain in code/tests.**

   Examples still exist in `test_scorer_variants.py:1-3`, `test_ds_scorer_tp_determinism.py:1-12`, and `selection_kernel.py:345-356`. This should be cleaned before final merge, but it does not block the next oracle/measurement round.

## Goal Alignment Summary

ACs: 5/6 addressed | Forgotten items: 0 | Unjustified deferrals: 5

AC-1, AC-2, AC-3, AC-5, and AC-6 have some progress; only AC-5 is effectively complete. AC-4 remains unimplemented. The unjustified deferral buckets are oracle fail-closed/64K evidence, graph-safe scorer port plus full AC-3 measurement, Tier-2.A task13-task17, M4 perf/consolidation task19, and task20 final decision record.

## Tracker Update

I updated the mutable section of `goal-tracker.md` only:

- accepted the Round-3 anchor over-budget fix and removed the resolved AC-3 blocking side issue;
- accepted task10 as completed/verified for implementation and unit-tested anchor-budget ablation;
- accepted task11 as completed/verified for TP=8 full-matrix logical-path determinism;
- kept task8/task12 active because graph-safe scorer support and the binding non-regression matrix are still missing;
- kept task3/task4/task6/task7, task13-task17, task19, and task20 active/pending.

## Verification Run During Review

Targeted edited tests:

`python3 -m pytest test/registered/unit/layers/attention/test_scorer_variants.py test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py -q`

Result: 17 passed, 5 warnings.

Broader DS unit files:

`python3 -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py test/registered/unit/layers/attention/test_selection_recall_oracle.py test/registered/unit/layers/attention/test_oracle_sink_and_force.py test/registered/unit/layers/attention/test_scorer_norm.py test/registered/unit/layers/attention/test_scorer_variants.py test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py -q`

Result: 339 passed, 24 warnings, 9 subtests passed.

Not complete: original-plan acceptance work remains active.
