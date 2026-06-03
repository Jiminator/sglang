# Round 18 Review Result

Mainline Progress Verdict: STALLED

Goal Alignment Summary: ACs: 9/10 addressed, 8/10 met | Forgotten items: 0 | Unjustified deferrals: 1

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-18-prompt.md`, `round-18-contract.md`, `round-18-summary.md`, `goal-tracker.md`, Round 15-17 summaries/reviews, commit `fcc2d1cdb`, the R18 AC-5 artifacts, the AC-5 client-SLO artifacts from R6-R9, and the DS selection/CUDA-graph hot spots.

## Implementation Review

Round 18 did not land a production code change. Commit `fcc2d1cdb` only adds `runs/20260530_dsv32_loop6/ac5_conc16_strict/` evidence. The bounded-context closed-batch decode curve is internally enough to show that a DS server booted at `context_length=8192` can reach conc-16 closed-batch decode TPS of 30.33 (`ctx8192_decode_curve.json:37-47`, verifier clean), but that is not the AC-5 client-SLO proof requested by the original plan or the R18 contract.

The R18 contract still says the mainline should implement the residual top-k fix as an exact seq-aware path with "no context cap", add regression coverage, measure the component breakdown, and then run the full client workload with exact arrays and attribution (`round-18-contract.md:13-20`, `:42-52`). None of those landed. The production path still allocates graph state at `max_seq_len=int(self.req_to_token.shape[1])` (`dsa_backend.py:843-847`, `:1158-1162`) and still calls `torch.topk(scores_view, effective_k, ...)` over that full `scores_view` width (`selection_kernel.py:838-885`).

## Mainline Gaps

1. **AC-5 is still not validated because the required client workload and TTFT measurement are missing.**

   AC-5 requires `development/benchmark.sh` at `NUM_PROMPTS=320`, conc 16/32/64, 4096/512, radix-on, exact per-request arrays, sidecars, and measured admission-wait vs prefill/decode attribution (`refined_plan_v1.md:66-75`). R18 records only a closed-batch pure-decode curve. The R18 report explicitly admits that fresh ctx8192 flood TTFT could not be captured and that publishing fresh conc-16 P99 TTFT is still the remaining item (`ac5_conc16_strict.md:43-52`). An inference from the older full-context R6 TTFT is not a replacement for the same-run P99 TTFT array required by AC-5.

2. **The bounded-context operating point is outside the fixed Option-B/no-context-cap target.**

   The original plan fixes the target as DeepSeek-V3.2 FP8 at the fixed Option-B operating point (`refined_plan_v1.md:5-7`) and later states that the deterministic serve/bench flags are fixed, with `mem_fraction_static` as the one lever the loop deliberately moves (`refined_plan_v1.md:118-122`). R17 also required the residual top-k remediation without a context cap, and R18 repeated that in its contract (`round-18-contract.md:13-20`). R18 instead uses `--context-length 8192` as the performance lever (`ac5_conc16_strict.md:18-23`, `ctx8192_decode_curve.json:3-9`). That makes the result a useful bounded-workload characterization, but it cannot be counted as preserving the full-context AC-5/AC-8 operating point or as completing the original plan.

3. **The residual full-context top-k blocker remains in production.**

   R18 concluded that a no-context-cap top-k speedup would require a more serious kernel, then did not implement it. The full-context graph-safe path still topks over the full captured score row (`selection_kernel.py:838-885`), with the graph width sourced from `req_to_token.shape[1]` (`dsa_backend.py:843-847`, `:1158-1162`). There are no new adversarial monolithic-vs-blocked-topk tests, no CUDA graph zero-allocation test for a new top-k path, and no component breakdown/DSA floor artifact.

4. **AC-10 remains correctly gated and not met.**

   The original plan still includes AC-10 after the Tier-1 spine. Since AC-5 remains partial, AC-10 must not start, but it also means the loop is not complete.

## Blocking Side Issues

1. **`bench_serving` window-mode failure now blocks AC-5 closure.**

   Claude reports empty `ttfts`/`itls`, empty `generated_texts`, and impossible aggregate throughput from the AC-5 harness (`ac5_conc16_strict.md:46-50`). That is not a harmless reporting issue: AC-5 is defined around measured P99 TTFT and exact arrays. The benchmark path must fail closed when successful generation requests have missing TTFT/ITL/text details, then the root cause must be fixed before any AC-5 pass claim.

2. **The R18 decode verifier is not an AC-5 verifier.**

   `ctx8192_decode_metrics_tool.py` only recomputes medians from a hand-entered JSON file (`ctx8192_decode_metrics_tool.py:37-69`). The JSON itself says the source log is gitignored (`ctx8192_decode_curve.json:2`), and there is already a mismatch between the JSON batch-1 samples (`210.73` at `ctx8192_decode_curve.json:13-20`) and the committed text excerpt (`43.47` at `ctx8192_decode_curve.txt:4-9`). This does not invalidate the conc-16 median by itself, but it is not durable acceptance evidence for AC-5.

## Queued Side Issues

1. Cross-node benchmark wrapper smoke remains future-gated. It does not block single-node AC-5.

2. DSA-default conc-64 TPS around 29.4 remains a pre-existing DSA/H200 client-SLO tension. It is useful context for characterization, but it does not replace the DS AC-5 measurements.

## Goal Tracker Audit

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified earlier. |
| AC-2 | MET | Feasibility budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, scale consumers, launcher, real-mask NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted 0.7 operating point, HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | R18 improves bounded-context closed-batch decode, but no accepted full client run, TTFT array, attribution, or full-context top-k remediation exists. |
| AC-6 | MET | Verified in R12 under the user-approved non-regression/opt-in semantics. |
| AC-7 | MET / CHARACTERIZED | Verified in R15 as characterized/soft-met. |
| AC-8 | MET | Verified in R16 at the full-context lifted DS point. R18 does not preserve that point for AC-5. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. |
| AC-10 | NOT MET | Correctly gated behind AC-5 and full Tier-1 verification. |

Forgotten items: none. Every original task is still represented in Active, Completed, or the gated AC-10 path. Deferred items: no tracker deferrals, but R18 effectively deferred the fresh AC-5 TTFT/full-client run; that is unjustified for completion and keeps task6 active.

## Required Implementation Plan

1. Make Round 19's sole mainline AC-5 closure at an accepted operating point. Treat `ac5_conc16_strict/` as bounded-context characterization only unless the owner explicitly re-scopes the target in writing; do not move task6 to Completed and Verified from the current evidence.

2. Fix the benchmark path before collecting more SLO evidence. Reproduce the R18 empty-latency failure with the smallest `development/benchmark.sh` run, then patch `python/sglang/bench_serving.py` and/or the benchmark wrapper so generation backends fail closed when `completed > 0` but `ttfts`, `itls`, `generated_texts`, or expected `output_lens==512` are missing. The fixed `benchmark.sh` artifact must refuse to publish unusable arrays.

3. Restore the full-context lifted DS operating point for the AC-5 remediation: DS int8, `mem_fraction_static=0.7`, radix-on fixture, TP=8, no `--context-length 8192` cap. Keep the bounded-context result in the report as a separate characterization, not the pass condition.

4. Implement the exact full-context seq-aware top-k remediation. Add a production graph-safe blocked top-k path in `retrieve_topk_graph_safe` using preallocated `DSGraphState` partial-score/partial-index scratch. Each logical block must keep `partial_k=max_top_k` candidates so adversarial cases where all winning scores fall in one block remain exact; blocks wholly past each request's `seq_len` must be skipped or filled with sentinels on device; the merge step must return the same ascending logical positions and valid lengths as monolithic `torch.topk`.

5. Add regression coverage before hardware: monolithic vs blocked top-k on CPU/CUDA fixtures with all winners in one block, mixed request lengths, boundary `seq_len`, padding, `per_request_valid`, and production dtypes. Include CUDA graph replay/zero-allocation coverage for the new path.

6. Rerun the component profile at the lifted full-context point. Publish DS selection/top-k, DSA FlashMLA+MoE floor, token-label update/write, and scheduler/interleave residuals so the next bottleneck is measured rather than inferred.

7. Rerun the full AC-5 client workload with `development/benchmark.sh`: `NUM_PROMPTS=320`, conc 16/32/64, 4096 ISL / 512 OSL, radix-on, TP=8. Publish exact per-request arrays, `.meta.json` sidecars, request-time attribution, server-info, and a fail-closed verifier. If the accepted owner criterion remains conc-16 strict plus 32/64 characterization, the report still must include measured TTFT/TPS for all three concurrencies and must not infer missing TTFT.

8. Keep AC-10 gated until AC-5 is verified under the accepted criterion and AC-3 through AC-9 remain verified. Then start the adjustable-`top_k`/selector R&D and record NIAH recall deltas plus TPS/TTFT cost.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 18 Review.
- Added an `18-review` Plan Evolution row rejecting the R18 AC-5 completion framing.
- Kept task6/AC-5 Active and rewrote its note around the bounded-context decode characterization plus missing full-client proof.
- Updated the existing AC-5 blocking issue and added a blocking issue for using `--context-length 8192` as the pass lever.
- Left AC-10 active/gated and moved no task to Completed and Verified.

## Validation Performed

- `git log --oneline -30`
- `git show --stat --oneline fcc2d1cdb`
- Inspected `round-18-contract.md`, `round-18-summary.md`, `goal-tracker.md`, and Round 15-17 summaries/reviews.
- Inspected `ac5_conc16_strict.md`, `ctx8192_decode_curve.json`, `ctx8192_decode_curve.txt`, `ctx8192_decode_metrics_tool.py`, `closed_batch_ctx8192.txt`, and `get_server_info_ctx8192.json`.
- Ran `python3 runs/20260530_dsv32_loop6/ac5_conc16_strict/ctx8192_decode_metrics_tool.py --verify` -> PASS for the bounded-context decode JSON.
- Inspected original AC-5 client-SLO report/arrays/sidecars and the benchmark harness.
- Inspected `selection_kernel.py`, `cuda_graph.py`, and `dsa_backend.py` around the remaining full-width top-k path.
- Ran `git diff --check ece26eb52..fcc2d1cdb`.

NOT COMPLETE
