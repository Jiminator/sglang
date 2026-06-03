# Round 20 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: ACs: 9/10 addressed, 8/10 met | Forgotten items: 0 | Unjustified deferrals: 0

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-20-prompt.md`, `round-20-contract.md`, `round-20-summary.md`, `goal-tracker.md`, Round 17-19 summaries/reviews, commit `96bc789cc`, and the new `runs/20260530_dsv32_loop6/ac5_fullctx/` artifacts.

## Implementation Review

Round 20 did advance the AC-5 mainline. Commit `96bc789cc` records a real full-context DS int8/mem0.7/radix-on/TP=8 measurement at conc 16/32/64, and `ac5_fullctx_metrics_tool.py --verify` runs clean on the committed reduced arrays:

| conc | achieved | P99 TTFT | per-req TPS p50 | strict status |
|---:|---:|---:|---:|---|
| 16 | 16.00 | 13.13 s | 24.9 | TTFT passes, TPS fails |
| 32 | 31.99 | 25.33 s | 19.5 | TTFT/TPS fail |
| 64 | 47.03 | 77.90 s | 17.3 | TTFT/TPS fail |

This resolves the R19 runtime blocker in one important sense: the accepted full-context server produces non-empty `ttfts` and ITL-derived request timings, so the empty-stream failure is not a general full-context `bench_serving` blocker.

However, the R20 artifact is not acceptance-grade AC-5 evidence, and the summary overclaims the verifier. The committed JSON is a lossy summary, not the exact source required by the plan and by the R8/R9 evidence standard.

## Mainline Gaps

1. **The R20 "exact arrays + fail-closed verifier" claim is false.**

   `ac5_fullctx_metrics_tool.py` reads raw JSONLs from `/tmp/ac5r20/results` during `--build`, but writes only `ttfts_s`, a pre-derived `per_req_gen_tps`, booleans for output/errors/ITL sanity, and a SHA string into the committed artifact (`runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py:60`). It does not commit the raw `itls`, `output_lens`, `input_lens`, `errors`, or `generated_texts` arrays needed to recompute TPS and the fail-closed sanity from committed files.

   The verifier then checks those reduced fields rather than recomputing them from independent committed data (`runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py:95`). I verified this with a temporary-copy mutation: replacing every c16 `per_req_gen_tps` value with `100.0` still exits 0 and prints `>=30TPS=True`. That means the verifier can be made to "pass" the strict TPS axis by editing a derived array; it has no independent ITL/output source or stored expected-metric check.

   This regresses below the R9 AC-5 evidence bar, where `ac5_metrics_tool.py` recomputed TTFT, TPOT/TPS, and ITL from committed exact arrays and failed nonzero on tampering.

2. **The R20 workload does not match the original AC-5 `NUM_PROMPTS=320` requirement, and this is not reconciled as plan evolution.**

   The committed c16 sidecar records `num_prompts: 64`, warmup 120 s, window 300 s (`runs/20260530_dsv32_loop6/ac5_fullctx/meta_c16.json:6`). The arrays record `completed=192` for c16/c32/c64 (`runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_arrays.json:10`, `:413`, `:816`). The original plan and active task still say AC-5 is the 320-prompt client workload.

   A 64-prompt steady-state epoch may be a defensible methodology because of the known cold-ramp trap, but it is not the same as silently satisfying the immutable AC-5 wording. Under this review's instructions, Claude must complete the original planned task unless the owner explicitly accepts a methodology amendment.

3. **Only the c16 benchmark sidecar is committed.**

   `runs/20260530_dsv32_loop6/ac5_fullctx/` contains `meta_c16.json` but no c32/c64 `.meta.json` sidecars. AC-5 requires valid sidecars/radix proof for the all-concurrency workload, not just one representative file. The tool's `operating_point` field is also empty (`ac5_fullctx_arrays.json:3`), and `--verify` does not validate DS/int8/mem0.7/radix-on/full-context/TP=8 invariants.

4. **The component-breakdown artifact is empty where the report says it exists.**

   `ac5_fullctx_attribution.txt` ends with `## Decode component ...` and no component lines (`runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_attribution.txt:18`). The report's component breakdown is a stitched narrative from older AC-7/R17 evidence (`ac5_fullctx_report.md:44`), not a measured per-conc R20 breakdown. That is acceptable as context, but not as satisfying the R20 contract's "admission/decode attribution + DSA FlashMLA+MoE component floor" success criterion.

5. **The strict AC-5 TPS axis remains unresolved at full context.**

   R20 measures the gap instead of inferring it, which is useful. It does not implement the full-context blocked-topk remediation and does not have an explicit owner-approved bounded-context rescope. Full-context c16 remains 24.9 TPS/req, below the current strict `>=30` axis; c32/c64 remain below both strict axes. AC-5 therefore stays active, and AC-10 remains gated.

## Blocking Side Issues

1. **Acceptance evidence is not fail-closed enough to rely on.**

   This blocks AC-5 closure because the headline TPS and output/error sanity cannot be recomputed from committed raw or exact derived sources. A verifier that passes after derived-TPS tampering is not an AC verifier.

2. **Full-context TPS remediation/rescope is still a mainline blocker.**

   The bounded-context c16 30.3 TPS result remains characterization only. The next implementation round must implement the exact full-context blocked-topk path required by the plan, not repeat the owner-decision prompt as the main deliverable.

3. **R20 methodology drift is unapproved.**

   The tracker now records the drift, but AC-5 cannot be moved to Completed and Verified while the committed run is `num_prompts=64`/192 completions and the plan still requires the 320-prompt workload.

## Queued Side Issues

1. Cross-node wrapper smoke remains queued for future remote-host artifacts; it does not block single-node TP=8 AC-5.

2. DSA-default conc-64 TPS around 29.4 remains a pre-existing DSA/H200/client-SLO tension, not a DS-introduced AC-6 blocker.

## Goal Alignment Check

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified earlier. |
| AC-2 | MET | Feasibility budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, scale consumers, launcher, real-mask NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted DS int8/mem0.7 point, HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | R20 adds real full-context measurements and c16 TTFT passes, but evidence is not exact-recomputable, workload metadata does not match `NUM_PROMPTS=320`, sidecars are incomplete, component lines are missing, and full-context TPS still fails. |
| AC-6 | MET | Verified in R12 under approved non-regression / opt-in semantics. |
| AC-7 | MET / CHARACTERIZED | Verified in R15 as characterized/soft-met. |
| AC-8 | MET | Verified in R16 at the lifted full-context DS point. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. |
| AC-10 | NOT MET | Correctly gated behind verified AC-5 and the full Tier-1 spine. |

Forgotten items: none after tracker correction. The exact-array, 320-prompt, sidecar, component-breakdown, and strict-TPS gaps are now tracked under task6/blocking side issues.

Deferred items: none in `Explicitly Deferred`. Claude's "blocked-topk kernel vs bounded-context rescope" and AC-10 notes are incomplete work, not accepted deferrals.

Plan evolution: R20's full-context measured evidence is valid directional progress. I reject the implied rescope that bounded context is the natural deployment until the owner explicitly changes the target. I also reject treating the `num_prompts=64` steady-state method as a silent replacement for the original AC-5 `NUM_PROMPTS=320` run.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 20 Review.
- Added a `20-review` Plan Evolution row accepting R20 measurement progress but rejecting AC-5 evidence completion.
- Updated task6/AC-5 Active status to mention the R20 full-context numbers, evidence-quality gaps, missing sidecars/component data, workload-methodology drift, and remaining TPS miss.
- Marked the R19 live full-context streaming-array blocker as resolved as a runtime blocker.
- Added a new Blocking Side Issue for the R20 evidence/verifier gap.
- Moved no task to Completed and Verified.

## Required Implementation Plan

1. **Rebuild R20 AC-5 evidence to the R9 verifier standard.**

   Use the raw `/tmp/ac5r20/results/*.jsonl` files if still available, otherwise rerun. Commit an exact source artifact per concurrency containing: `completed`, effective `concurrency`, `input_lens`, `output_lens`, `ttfts_s`, raw per-request `itls_s` or an exact per-token/per-request ITL source sufficient to recompute TPS, `errors`, generated-text non-empty evidence, source SHA256, and stored headline metrics. Do not store only `per_req_gen_tps` and booleans as the verifier source.

2. **Make the verifier actually fail closed.**

   `--verify` must read only committed files and assert:
   - c16/c32/c64 all present.
   - `completed == 320` for the original AC-5 workload, unless an explicit owner-approved tracker row changes the methodology.
   - `len(ttfts) == len(output_lens) == len(errors) == len(itls) == completed`.
   - every `output_len == 512`, every error is empty, every TTFT is positive, every request has ITL data, generated text is not empty where streaming text is expected.
   - P99 TTFT and per-request TPS p50 recompute from exact arrays and match stored expected headline values at published precision.
   - all source SHAs are full 64-hex and match recorded source-file sizes or embedded exact arrays.
   - all sidecars prove DS enabled, `signature_dtype=int8`, mem0.7, radix-on fixture, `disable_radix_cache=false`, `context_length=null`, `max_total_num_tokens=396096`, TP=8, and request-time stats on.

   Include temporary-copy tamper checks in the summary: mutate c16 TPS source, output_len, a sidecar radix field, and a headline TTFT value; each must exit nonzero.

3. **Complete the original AC-5 workload requirement.**

   Rerun or reconstruct the fixed `NUM_PROMPTS=320`, conc 16/32/64, 4096/512, full-context DS int8/mem0.7/radix-on workload and commit c16/c32/c64 sidecars. If Claude believes `num_prompts=64` steady-state epochs are the only valid SLO methodology, the next round must first obtain an explicit owner-approved plan-evolution row; do not count it as the original AC-5 run by implication.

4. **Fill the measured component breakdown.**

   Commit the steady decode lines / component excerpts backing DS selection/topk, DSA FlashMLA+MoE floor, token-label write/update, and scheduler/prefill-interleave residuals. `ac5_fullctx_attribution.txt` must not end at an empty `Decode component` header.

5. **Implement the full-context blocked top-k remediation.**

   In `retrieve_topk_graph_safe`, add an exact graph-safe blocked top-k path using preallocated `DSGraphState` partial-score/partial-index scratch. Each logical block must retain `top_k` candidates so adversarial cases where all winners are in one block remain exact; blocks past each request's `seq_len` must be device-sentinel-filled/skipped; the merge must return the same logical positions and valid lengths as the current monolithic `torch.topk`. Add CPU/CUDA regression coverage for all-winners-in-one-block, mixed request lengths, boundary `seq_len`, padding, `per_request_valid`, production dtypes, and CUDA graph replay/zero-allocation.

6. **Rerun acceptance measurements after the top-k fix.**

   First prove full-context closed-batch conc-16 `>=30 TPS/req`, then rerun the full AC-5 client workload and verifier. Keep bounded-context ctx8192 results as characterization only unless the owner explicitly rescope the target.

7. **Only then start AC-10.**

   After AC-5 is verified and AC-3 through AC-9 remain verified, implement the adjustable-`top_k` sparse-matmul kernel or selector path and publish NIAH 4K/16K/64K recall deltas plus TPS/TTFT cost.

## Validation Performed

- `git log --oneline --decorate -30`
- `git show --stat --oneline --decorate 96bc789cc`
- `git diff --check 96bc789cc^..96bc789cc`
- Inspected `development/loop6/refined_plan_v1.md`, `round-20-prompt.md`, `round-20-contract.md`, `round-20-summary.md`, `goal-tracker.md`, and Round 17-19 summaries/reviews.
- Inspected `ac5_fullctx_report.md`, `ac5_fullctx_arrays.json`, `ac5_fullctx_metrics_tool.py`, `ac5_fullctx_attribution.txt`, `get_server_info_fullctx.json`, and `meta_c16.json`.
- Ran `python3 runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py --verify` -> PASS on clean data.
- Ran a temporary-copy mutation setting every c16 `per_req_gen_tps` to `100.0`; verifier still exited 0 and printed `>=30TPS=True`, confirming the fail-closed gap.
- Checked tracked/ignored status: R20 raw `.jsonl` files are not tracked; only `meta_c16.json` is committed, while `/tmp/ac5r20/results/` has c16/c32/c64 raw files and sidecars.

NOT COMPLETE
