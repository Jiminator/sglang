# Round 22 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: ACs: 9/10 addressed, 8/10 met | Forgotten items: 0 | Unjustified deferrals: 2

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-22-prompt.md`, `round-22-contract.md`, `round-22-summary.md`, `goal-tracker.md`, Round 19-21 summaries/reviews, commits `704be382f` and `8ab6c7db0`, the AC-5 full-context verifier/artifacts, and the changed selection tests/code.

## Implementation Review

R22 did advance the AC-5 line. The R21 workload-metadata fail-open is fixed for the specific sidecar tamper class Codex reported: clean `ac5_fullctx_metrics_tool.py --verify` exits 0, and temporary-copy mutations of `mode`, `num_prompts`, `isl_total_tokens`, `osl_tokens`, sidecar `concurrency`, `warmup_seconds`, and `server_args.max_total_num_tokens` each exit 1.

I accept the R22 owner decisions as plan evolution based on the round summary/tracker: AC-5 methodology is now np64 steady-state, and bounded-context rescope is declined in favor of the full-context blocked-topk kernel. That removes the prior methodology ambiguity. It does not complete AC-5.

The blocked-topk commit is useful foundation, but it is not the graph-safe kernel and it has an exactness hole that must be fixed before using it as the oracle for the Triton path.

## Mainline Gaps

1. **AC-5 is still incomplete: the actual graph-safe blocked top-k and post-kernel rerun are not implemented.**

   `blocked_topk_sequence_order` is only an eager torch helper. The production graph-safe path still does a monolithic `torch.topk(scores_view, effective_k, ...)` over the full captured score width in `retrieve_topk_graph_safe` (`python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:940-977`). `rg` shows the new helper is not called by production code.

   The current accepted full-context numbers therefore remain unchanged: c16 13.13s TTFT / 24.9 TPS, c32 25.33s / 19.5 TPS, c64 77.90s / 17.3 TPS. There is no new full-context closed-batch c16 `>=30 TPS/req` proof and no post-kernel AC-5 client rerun. AC-10 remains gated.

2. **The R22 "exact oracle" is not identical to monolithic top-k when finite scores tie.**

   The docstring claims identical output to `select_topk_sequence_order` (`selection_kernel.py:494-505`), and the new tests assert exact tensor equality, but the tests deliberately use distinct finite scores. A finite-tie counterexample fails:

   ```text
   scores = torch.ones(1, 8), K=3, block_width=4
   monolithic -> [[4, 5, 6]]
   blocked    -> [[4, 6, 7]]
   equal      -> False
   ```

   This matters because DS scores can tie after int8/scaled scoring, masked positions, or repeated score values. Before the Triton kernel is written against this contract, the selector must either define a deterministic shared tie-break or explicitly weaken the contract/tests to tie-equivalent sets. The current "identical oracle" claim is false.

## Blocking Side Issues

1. **The AC-5 verifier still does not prove workload volume/duration.**

   R22 validates the top-level sidecar fields, but `verify()` only checks `completed > 0` and array lengths equal `completed` (`ac5_fullctx_metrics_tool.py:101-110`). It does not assert the approved artifact's expected completed count, minimum measured duration, trial identity, or that expected workload constants are code-owned rather than read from `ac5_fullctx_arrays.json` (`ac5_fullctx_metrics_tool.py:45-47`, `:133`).

   I reproduced the gap by reducing each concurrency to one internally-consistent request and updating the stored headline/means; `--verify` still printed PASS. That is not enough for "artifact IS the claimed AC run." Add expected completed/duration/trial checks before calling this fully acceptance-grade.

2. **The R22 blocked-topk tie hole blocks the kernel foundation from being a safe oracle.**

   Fix the tie contract before implementing the Triton skip-kernel, otherwise the future kernel can match the new helper while still diverging from the existing monolithic path on real tied scores.

## Queued Side Issues

1. Cross-node benchmark wrapper smoke remains queued for future remote-host artifacts; it does not block the single-node TP=8 AC-5 line.

2. DSA-default conc-64 TPS around 29.4 remains a pre-existing DSA/H200 client-SLO tension, not a DS-introduced AC-6 blocker.

3. The new `TestBlockedTopKExactness` class was appended after an existing `if __name__ == "__main__": unittest.main()` block. Pytest discovery works, but direct `python test_double_sparsity_unit.py TestBlockedTopKExactness...` fails before the class is defined. This is non-blocking because the registered pytest path runs it, but the next test-file edit should move the main guard to the end.

## Goal Alignment Check

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified earlier. |
| AC-2 | MET | Footprint budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, scale consumers, launcher, real-mask NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted DS int8/mem0.7 point, HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | R22 fixes the R21 metadata fail-open and resolves methodology/kernel decisions, but strict full-context TPS still misses, the graph-safe kernel is not implemented, verifier workload-volume checks are incomplete, and no post-kernel AC-5 rerun exists. |
| AC-6 | MET | Verified in R12 under approved non-regression / opt-in semantics. |
| AC-7 | MET / CHARACTERIZED | Verified in R15 as characterized/soft-met. |
| AC-8 | MET | Verified in R16 at the lifted full-context DS point. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. |
| AC-10 | NOT MET | Correctly gated behind AC-5 and full Tier-1 verification. |

Forgotten items: none. The original tasks are represented in Active, Completed, or the gated AC-10 path.

Deferred items: none in `Explicitly Deferred`, but Claude's R22 "Remaining Items" are incomplete AC-5 work, not acceptable completion deferrals: graph-safe Triton blocked top-k and the post-kernel AC-5 rerun remain mandatory. AC-10 is still gated, not complete.

Plan evolution: accepted R22 owner decisions: np64 steady-state AC-5 methodology and full-context blocked-topk kernel path. Rejected any implication that AC-5 is complete.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 22 Review.
- Added a `22-review` Plan Evolution row accepting the R22 owner decisions and verifier metadata progress while rejecting AC-5 completion.
- Updated task6 to the owner-approved np64 methodology and the next required kernel/rerun path.
- Marked the R21 workload-metadata verifier blocker resolved by R22.
- Updated strict-SLO and bounded-context blocker rows to reflect the owner-chosen full-context kernel path.
- Added new Blocking Side Issues for verifier workload-volume anchoring and blocked-topk finite-tie exactness.
- Moved no task to Completed and Verified.

## Required Implementation Plan

1. **Fix the top-k exactness contract first.**

   Define one deterministic ordering for all DS top-k selectors: score descending, then logical position ascending for ties. Implement that shared contract in `select_topk_sequence_order`, `blocked_topk_sequence_order`, and the future graph-safe kernel. Add finite-tie regressions that currently fail, including all-equal scores, ties crossing block boundaries, ties at the K boundary, and ties mixed with `-inf` padding.

2. **Implement the graph-safe full-context blocked top-k in `retrieve_topk_graph_safe`.**

   Use `DSGraphState` preallocated partial-score/partial-index scratch. Size scratch by `num_blocks = ceil(max_seq_len / block_width)` and `partial_k = min(max_top_k, block_width)`. Stage 1 writes each live block's top candidates and sentinel-fills blocks past each request's `seq_len`; Stage 2 merges the partial candidates, applies the shared tie contract, writes sequence-ascending `out_indices`, and writes `out_lengths`. Do not allocate inside graph capture, do not read CUDA tensor values on host, and keep the FlashMLA `top_k == dsa_index_topk == 2048` ABI lock untouched.

3. **Cover the actual graph-safe path.**

   Add tests for all-winners-in-one-block, mixed `seq_lens`, exact block-boundary lengths, padding when `max_seq_len` is not a block multiple, `K >= block_width`, `K > n`, finite ties, `per_request_valid`, production dtypes/int8 scales, and CUDA graph replay/zero-allocation. Keep the eager helper as the oracle only after the tie contract is fixed.

4. **Tighten the AC-5 full-context verifier.**

   Move approved workload identity into code-owned constants and copy it into JSON only as documentation. Assert `completed` per concurrency for the committed np64 artifact, `duration_s >= measurement_window_seconds`, sidecar `trial_id`, and all current workload/operating-point fields. Add temporary-copy tamper tests for reduced `completed`, shortened `duration_s`, and coordinated `expected_workload`+sidecar mutation.

5. **Rerun AC-5 after the kernel.**

   First publish a full-context closed-batch c16 proof showing `>=30 TPS/req`. Then rerun the AC-5 client workload under the owner-approved np64 steady-state method at conc 16/32/64, 4096 ISL / 512 OSL, radix-on, TP=8, full context. Commit sidecars, exact arrays, attribution, report, and the hardened verifier output.

6. **Only then start AC-10.**

   After AC-5 is verified and AC-3 through AC-9 remain verified, implement the adjustable-`top_k` sparse-matmul kernel or selector path and publish NIAH 4K/16K/64K recall deltas plus TPS/TTFT cost.

## Validation Performed

- `python3 runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py --verify` -> PASS on clean data.
- Temporary-copy workload tamper tests for `mode`, `num_prompts`, `isl_total_tokens`, `osl_tokens`, `max_total_num_tokens`, sidecar `concurrency`, and `warmup_seconds` -> each exited 1.
- `pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py -k 'BlockedTopKExactness or test_retrieve_topk_graph_safe'` -> 7 passed, 278 deselected, 6 subtests passed.
- Finite-tie blocked-topk counterexample -> monolithic and blocked outputs differ.
- Temporary-copy one-request AC-5 artifact with internally consistent stored headlines -> verifier still exited 0, confirming the workload-volume gap.
- `python3 test/registered/unit/layers/attention/test_double_sparsity_unit.py TestBlockedTopKExactness.test_all_winners_in_one_block` -> fails because the class is below the first `unittest.main()` guard.
- `git diff --check 704be382f^..HEAD` -> clean.
- `git diff --check` after tracker/review updates -> clean.

NOT COMPLETE
