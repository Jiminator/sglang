# Round 17 Review Result

Mainline Progress Verdict: ADVANCED

Round 17 materially advanced AC-4: the backend lifted-budget graph branch is now
wired, the validator no longer rejects CUDA graph for lifted decode, and the focused
backend CUDA-graph replay proof passes locally on H200. I verified:

```bash
pytest -q test/registered/unit/layers/attention/test_scorer_variants.py::TestLiftedBudgetABI
# 12 passed

pytest -q test/registered/unit/layers/attention/test_lifted_budget_decode.py::TestLiftedBudgetBackendGraphSafe
# 2 passed

git diff --check 714cf62b2..41e0af078
# clean
```

However, the work is not complete. Claude's summary defers a graph-captured TP=8
lifted-width selector-equality artifact that prior review required, and the task17
disposition record still contains stale deferred/eager-only claims that directly
contradict the R17 implementation. task19 and task20 also remain original-plan
mainline work.

## Mainline Gaps

1. **task16 is still missing the graph-captured TP=8 lifted-width selector equality proof.**

   Evidence:
   - Round 16 review explicitly required "selected-index and valid-length equality
     across ranks at 4096 and 8192 under CUDA graph capture, not only the
     eager/logical path" (`round-16-review-result.md:62`).
   - Round 17's contract also required a lifted-width selection graph-capture proof:
     "Plus a single-rank graph-capture of the lifted-width selection replaying
     zero-alloc" (`round-17-contract.md:47`).
   - The new R17 backend test covers `_forward_lifted_budget` with preselected
     physical slots, but it bypasses the lifted-width selector/all-reduce path
     (`test_lifted_budget_decode.py:550` onward).
   - Claude's own summary moves the standalone graph-captured 8-rank TP=8
     selector-equality artifact to a follow-on (`round-17-summary.md:63`).

   Required implementation plan:
   - Add a CUDA integration test for lifted-width selection, not just lifted decode.
   - Run 8 ranks with a CUDA-capable process group at lifted widths 4096 and 8192.
   - Bind identical token-label/channel-mask fixtures per rank, set
     `enable_lifted_budget_decode=true`, and size `DSGraphState` to
     `lifted_budget_top_k`.
   - Capture the real graph-safe selector path (`retrieve_topk_graph_safe` through
     the production `_select_topk_indices`/metadata route if feasible), replay it,
     and assert `selected_indices` and `valid_lengths` are identical across all 8
     ranks after replay.
   - Wrap replay in `assert_no_alloc_in_region`. Compare the captured results to
     the existing eager/logical TP=8 lifted-width reference at 4096/8192.

2. **task17 is not acceptable yet because `m9_tier2a_disposition.md` is internally contradictory.**

   Evidence:
   - The top of the file says lifted decode is production-ready and CUDA-graph-safe
     (`development/loop7/m9_tier2a_disposition.md:3`).
   - Later it still says the path is "eager-required" and that
     `validate_double_sparsity` rejects lifted decode unless `--disable-cuda-graph`
     is set (`development/loop7/m9_tier2a_disposition.md:81`).
   - The landed-surface section still says the launcher forces
     `--disable-cuda-graph` for `LIFTED_BUDGET` (`development/loop7/m9_tier2a_disposition.md:107`).
   - The impact section still says the only deferred item is production-graph
     hardening (`development/loop7/m9_tier2a_disposition.md:157`).

   Required implementation plan:
   - Rewrite M9 so every section consistently describes the R17 production graph
     state: validator relaxed, launcher no longer forces eager for lifted decode,
     `dequantize_k_cache_paged_out`/fixed scratch is the production path, and the
     4K graph-mode NIAH result is the binding production recall.
   - Keep the bounded-secondary 4K-only analysis, but remove all claims that
     production graph hardening is deferred.
   - Either add the graph-captured TP=8 artifact from finding 1, or state exactly
     where it is recorded. Do not call it a non-blocking follow-on.
   - Re-run the analyze review step for the corrected disposition.

3. **AC-6/task19 remains active.**

   Required implementation plan:
   - Use the existing Loop-7 serve/benchmark tooling at the int8/mem0.7/fp8-KV/TP=8
     op-point.
   - Record DS default, graph-safe DS hybrid, DSA, and lifted DS where applicable.
   - Capture conc-1 and conc-16 TTFT, decode TPS/req, GPU memory, graph replay
     status, admission behavior, radix/cache assumptions, exact server args, DS
     config, commit, GPU type, and artifact paths.
   - Produce the consolidated DS-vs-DSA recall/perf/non-regression report.

4. **AC-2/task20 remains active.**

   Required implementation plan:
   - After task19, write the final strategic-gate supersession decision record.
   - Cite M0 regime attribution, AC-1 closure, AC-3 hybrid scorer evidence, AC-4
     production-ready lifted evidence, AC-5 servability, and AC-6 perf guardrails.
   - Explicitly state what measured evidence superseded the Loop-6 Tier-2.A-primary
     ordering.

## Blocking Side Issues

None. The live non-speculative lifted graph path is not shown unsafe by this review;
the blockers are incomplete required evidence and contradictory AC-4 disposition text.

## Queued Side Issues

1. **Lifted + speculative decode scratch sizing is unsafe.** In target-verify/draft
   modes, metadata rows can expand to `bs * speculative_num_draft_tokens`
   (`dsa_backend.py:1086`), but lifted scratch is allocated with `max_bs=bs`
   (`dsa_backend.py:1175`). `_forward_lifted_budget` then slices scratch using
   the expanded row count from `page_table_1.shape[0]` (`dsa_backend.py:2124`).
   This can undersize `lifted_page_table`, `lifted_compact_indices`,
   `lifted_compact_kv`, and `lifted_q_padded`. This is queued because the Loop-7
   evidence and launcher are non-speculative; before advertising lifted DS with
   speculation, either size scratch by the expanded row count or fail closed in the
   validator.
2. Preserve or cite the R8 oracle-sink provenance before task20.
3. Remove plan/workflow markers from production code/comments/tests before final
   cleanup/merge.
4. Learned/distilled selector work remains out of scope unless explicitly approved.

## Goal Alignment Summary

ACs: 6/6 addressed | Forgotten items: 0 | Unjustified deferrals: 1

| AC | Status | Review result |
|----|--------|---------------|
| AC-1 | MET | Prior fail-closed oracle, separated baseline, stride reference, and oracle-off zero-hot-path evidence remain accepted. |
| AC-2 | PARTIAL | Recall uplift evidence exists; task20 final decision record is still missing. |
| AC-3 | MET | Graph-safe non-learned scorer/head/anchor path and non-regression matrix remain accepted. |
| AC-4 | PARTIAL | Backend graph hardening advanced, but graph-captured TP=8 lifted selector equality is still deferred and M9 is contradictory. |
| AC-5 | MET | 64K mem0.7 servability remains accepted; 128k remains out of scope. |
| AC-6 | PARTIAL | R17 adds useful lifted graph perf evidence, but task19 conc-1/16 perf consolidation is still missing. |

## Goal Tracker Update Requests

I updated `goal-tracker.md` directly:

- bumped Plan Version to 24 for Round 17 Review;
- added a Round 17 Review plan-evolution row;
- reactivated task16 as partial because the graph-captured TP=8 lifted-width
  selector-equality artifact is still deferred;
- reactivated task17 because M9 still contains stale deferred/eager-only claims;
- left task19 and task20 active;
- added the lifted+speculative scratch sizing issue to Queued Side Issues;
- left Explicitly Deferred empty.

PENDING
