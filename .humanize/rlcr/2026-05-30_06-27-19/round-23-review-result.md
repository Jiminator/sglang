# Round 23 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: ACs: 9/10 addressed, 8/10 met | Forgotten items: 0 | Unjustified deferrals: 2

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-23-prompt.md`, `round-23-contract.md`, `round-23-summary.md`, `goal-tracker.md`, recent Round 20-22 summaries/reviews, commit `2715b7382`, the changed selector/verifier/test files, and the AC-5 full-context artifacts.

## Implementation Review

Round 23 did advance the AC-5 prerequisite line. The R22 finite-tie oracle defect is fixed for the eager oracle/fallback pair: `select_topk_sequence_order` and `blocked_topk_sequence_order` now share `_topk_by_score_then_pos`, selecting by score descending then logical position ascending (`python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:428-479`, `:558-567`). The all-ones counterexample now returns `[0, 1, 2]` from both selectors, and the new finite-tie regressions pass.

The R22 verifier volume fail-open is also fixed. `ac5_fullctx_metrics_tool.py` now has code-owned `EXPECTED_WORKLOAD`, checks the JSON documentation copy against code, and asserts `completed == 192`, `duration_s >= 300`, and sidecar `trial_id` (`runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py:31-36`, `:104-122`, `:151`). I reproduced the important tamper classes on temporary copies: reduced completed count, short duration, JSON doc tamper, coordinated JSON+array tamper, and removed `trial_id` all exited 1.

This completes the two R22 prerequisite blockers. It does not complete AC-5 or the original Loop-6 plan.

## Mainline Gaps

1. **AC-5 is still incomplete: the production graph-safe blocked top-k and post-kernel run are not implemented.**

   The production path still scores into `scratch_scores[:bs, :max_seq_len]` and then runs a monolithic full-width `torch.topk(scores_view, effective_k, ...)` in `retrieve_topk_graph_safe` (`python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:932-1003`). The DSA production path calls that function when `ds_graph_state` exists (`python/sglang/srt/models/deepseek_v2.py:2235-2303`). The current DSA metadata allocation also does not size the partial block scratch for the blocked selector (`python/sglang/srt/layers/attention/dsa_backend.py:843-848`, `:1158-1163`).

   The accepted full-context numbers therefore remain the R20/R21 values: c16 13.13s TTFT / 24.9 TPS, c32 25.33s / 19.5 TPS, c64 77.90s / 17.3 TPS. There is still no full-context closed-batch c16 `>=30 TPS/req` proof and no post-kernel AC-5 client rerun.

2. **AC-10 remains not started and correctly gated, so the original plan is not complete.**

   The plan requires AC-10 after the full Tier-1 spine lands. Since AC-5 is still partial, AC-10 should remain gated; it must not be counted complete or quietly dropped.

## Blocking Side Issues

None newly found in the two R23 prerequisite fixes. The R22 blocked-topk finite-tie issue and AC-5 verifier workload-volume issue are resolved by `2715b7382`.

The missing production graph-safe blocked top-k is not a side issue; it is the next AC-5 mainline gap.

## Queued Side Issues

1. Cross-node benchmark wrapper smoke remains queued for future remote-host artifacts; it does not block the single-node TP=8 AC-5 line.

2. DSA-default conc-64 TPS around 29.4 remains a pre-existing DSA/H200 client-SLO tension, not a DS-introduced AC-6 blocker.

## Goal Alignment Check

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified earlier. |
| AC-2 | MET | Footprint budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, scale consumers, launcher, real-mask NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted DS int8/mem0.7 point, HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | R23 resolves the tie-oracle prerequisite and verifier volume fail-open, but production `retrieve_topk_graph_safe` still uses full-width `torch.topk`, strict full-context TPS still misses, and no post-kernel AC-5 rerun exists. |
| AC-6 | MET | Verified in R12 under approved non-regression / opt-in semantics. |
| AC-7 | MET / CHARACTERIZED | Verified in R15 as characterized/soft-met. |
| AC-8 | MET | Verified in R16 at the lifted full-context DS point. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. |
| AC-10 | NOT MET | Correctly gated behind AC-5 and full Tier-1 verification. |

Forgotten items: none. The original tasks are represented in Active, Completed, or the gated AC-10 path.

Deferred items: none in `Explicitly Deferred`, but Claude's "Remaining Items" are incomplete original-plan work, not acceptable completion deferrals. The graph-safe blocked top-k and post-kernel AC-5 rerun must drive the next round. AC-10 remains plan-gated, not complete.

Plan evolution: accepted R23 as prerequisite progress. Rejected any implication that AC-5 or the full Loop-6 plan is complete.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 23 Review.
- Added a `23-review` Plan Evolution row verifying commit `2715b7382` and stating that AC-5 remains active.
- Updated task6 to show the R23 tie-contract and verifier-volume blockers are resolved while the graph-safe kernel, c16 proof, and AC-5 rerun remain open.
- Marked the R22 verifier workload-volume blocker resolved by R23.
- Marked the R22 blocked-topk finite-tie blocker resolved by R23.
- Moved no task to Completed and Verified.

## Required Implementation Plan

1. **Implement the production graph-safe blocked top-k in `retrieve_topk_graph_safe`.**

   Add a fixed graph-safe top-k block width constant for the AC-5 kernel path, use `block_width=512`, compute `num_blocks = ceil(max_seq_len / block_width)`, and set `partial_k = min(max_top_k, block_width)`. In both DSA metadata allocation sites, pass `num_score_blocks=num_blocks` and `partial_topk=partial_k` to `allocate_graph_state` so `scratch_partial_scores` and `scratch_partial_indices` are always present for DS-enabled graph-safe forwards.

   Replace the full-width `torch.topk(scores_view, effective_k, ...)` block in `retrieve_topk_graph_safe` with a two-stage blocked selector. Stage 1 writes each live block's top `partial_k` `(score, logical_position)` candidates into `DSGraphState` scratch and sentinel-fills every block fully past that request's `seq_len`. Stage 2 merges the `num_blocks * partial_k` candidates, applies the same score-descending / position-ascending deterministic tie-break as `_topk_by_score_then_pos`, writes sequence-ascending `out_indices`, and writes `out_lengths`. Keep all work device-side, use only preallocated scratch, do not read CUDA tensor values on host, and do not change the FlashMLA `top_k == dsa_index_topk == 2048` ABI.

2. **Cover the actual graph-safe path.**

   Add unit/CUDA tests against the eager deterministic oracle for: all winners in one block, mixed `seq_lens`, exact block-boundary lengths, padding when `max_seq_len` is not a block multiple, `K >= block_width`, `K > n`, finite ties, ties mixed with `-inf`, `per_request_valid`, production dtypes with int8 scales, CUDA graph replay, and zero allocations after warmup. Include a direct graph-safe finite-tie test so the production captured path is locked to the new tie contract, not just the eager helper.

3. **Prove the performance target before rerunning full AC-5.**

   After the kernel lands, publish a full-context closed-batch c16 proof at DS int8/mem0.7/radix-on/TP=8 showing per-request TPS `>=30`. If this proof misses, do not spend a full AC-5 rerun; fix the kernel or record the miss with attribution.

4. **Rerun AC-5 under the approved np64 methodology.**

   Run the owner-approved steady-state full-context workload at conc 16/32/64, 4096 ISL / 512 OSL, radix-on, TP=8. Commit sidecars, exact arrays, attribution, report, and hardened verifier output. The verifier must continue to assert metrics, workload identity, volume, duration, and trial identity.

5. **Only then start AC-10.**

   After AC-5 is verified and AC-3 through AC-9 remain verified, implement the adjustable-`top_k` sparse-matmul kernel or selector path and publish NIAH 4K/16K/64K recall deltas plus TPS/TTFT cost.

## Validation Performed

- `python3 runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py --verify` -> PASS on clean data.
- Temporary-copy verifier tamper checks for reduced completed, short duration, JSON doc tamper, coordinated JSON+arrays tamper, and removed sidecar `trial_id` -> each exited 1.
- `pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py -k 'BlockedTopKExactness or test_select_topk or test_retrieve_topk_graph_safe'` -> 15 passed, 274 deselected, 9 subtests passed.
- `python3 test/registered/unit/layers/attention/test_double_sparsity_unit.py TestBlockedTopKExactness.test_all_equal_scores` -> OK.
- All-ones K=3/bw=4 selector counterexample -> monolithic and blocked both return `[[0, 1, 2]]`, lengths `[3]`.
- `pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py` -> 289 passed, 9 subtests passed.
- `git diff --check` -> clean.

NOT COMPLETE
