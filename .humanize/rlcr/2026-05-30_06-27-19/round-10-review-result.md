# Round 10 Review Result

Mainline Progress Verdict: ADVANCED

ACs: 7/10 addressed, 5/10 met | Forgotten items: 0 | Unjustified deferrals: 1

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-10-prompt.md`, `round-10-contract.md`, `goal-tracker.md`, Round 7-9 summaries/reviews, commits `d6e884aa9`, `daad92923`, `2fd2c6937`, the AC-9 artifacts, the AC-6 product-proof artifacts, `development/CLIENT_SLOS.md`, and the serve/benchmark scripts.

## Implementation Review

AC-9 is verified. `test/manual/test_double_sparsity_v32.py` now carries real prompt-token counts through `_generate` / `_GenAttempt` / `_run_niah`, computes the hard gate premise from `ds_input_tokens`, fails closed if usage is missing, and keeps the old word-count proxy only as recorded comparison data. The live artifacts show `input_tokens=1128` and `1678`, both under `INDEX_TOPK=2048`, `usage_missing=false`, DS/DSA recall 100%, and unchanged DS-fair thresholds (`runs/20260530_dsv32_loop6/ac9_real_token_within_budget.md:23-43`).

I found two high-signal AC-6 gaps. The no-table/toggle evidence is useful, but it does not satisfy AC-6 or the Round 10 contract.

## Mainline Gaps

1. **AC-6 DSA-default SLO proof is missing; the fresh evidence actually fails the stated SLO.**

   AC-6 required a DSA-default boot that meets the SLO unchanged, and the Round 10 contract made that concrete as a representative client run with sub-22s P99 TTFT and >=30 TPS (`round-10-contract.md:6-8`, `:18-22`). The new `dsa_default_slo.txt` records the opposite: P99 TTFT `22.6 / 86.1 / 202.4` seconds and per-request TPS `16.9 / 14.1 / 14.1` at conc 16/32/64 (`runs/20260530_dsv32_loop6/ac6_product_proof/dsa_default_slo.txt:8-11`). The report then reclassifies those numbers as cold-ramp-only and defers the clean steady-state DSA sweep to AC-7 (`runs/20260530_dsv32_loop6/ac6_optin_dsa_default_product.md:52-61`).

   That deferral is rejected. AC-7 is the 3-trial DS+DSA lifted-point re-sweep; it does not replace AC-6's product/default positive test. AC-6 remains partial until a tracked DSA-default SLO artifact actually passes under the proper methodology, or an already-tracked identical-operating-point baseline is cited with enough metadata to prove it applies after the DS changes.

2. **AC-6's "same locked Option B / only DS enablement differs" claim is contradicted by the captured server info.**

   `ac6_optin_dsa_default_product.md` says the two servers use the same locked operating point and differ only by Double Sparsity enablement (`runs/20260530_dsv32_loop6/ac6_optin_dsa_default_product.md:3-7`, `:25-26`). But `get_server_info_keys.json` shows the DS opt-in node has `disable_radix_cache: true` and no radix fixture, while the DSA-default node has `disable_radix_cache: false` (`runs/20260530_dsv32_loop6/ac6_product_proof/get_server_info_keys.json:2-13`, `:15-26`). The Round 10 contract explicitly asked for DS int8 @ 0.7 radix-on (`round-10-contract.md:10-12`).

   This does not invalidate the narrow fact that the DS flag allocates the int8 table, but it invalidates the stronger product-proof claim that the AC-6 DS-vs-default evidence is at the locked/radix-on operating point.

3. **Original Loop-6 plan remains incomplete.**

   AC-7, AC-8, and gated AC-10 are still pending. AC-5 is still directional-only because the strict client SLO misses conc 32/64 TTFT and all-conc TPS. These are not new Round 10 regressions, but they prevent `COMPLETE`.

## Blocking Side Issues

1. **AC-6 product proof is incomplete.** The next round must finish AC-6 before moving on: prove DSA-default meets the SLO unchanged and fix the DS opt-in locked-point/radix evidence.

2. **Strict client SLO still blocks the ultimate goal.** The verified AC-5 artifact remains directional, not shippable: conc 32/64 TTFT exceed 22s and per-request TPS is below 30 at every concurrency.

## Queued Side Issues

None newly added.

## Goal Alignment Check

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified earlier. |
| AC-2 | MET | Feasibility budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, sidecar consumers, launcher, real-mask NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted 0.7 operating point, HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | Evidence and attribution verified; strict SLO still fails. |
| AC-6 | PARTIAL | DSA no-table and DS int8 table toggle proven; DSA-default SLO unchanged and DS radix-on locked-point proof not yet valid. |
| AC-7 | NOT MET | 3-trial DS+DSA lifted-point re-sweep pending. |
| AC-8 | NOT MET | Lifted ~70K-token servability probe pending. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified this review. |
| AC-10 | NOT MET | Correctly gated behind full Tier-1 completion. |

Forgotten items: none. Every original plan task is represented in Active, Completed, or the gated AC-10 path. Deferred items: one unjustified AC-6 deferral: "fresh all-trials steady-state DSA sweep is AC-7's job" is not acceptable for AC-6's DSA-default SLO positive test.

Plan evolution: The Round 10 attempt to satisfy AC-6 from an established Loop-5 steady-state baseline plus a failing WARMUP=0 smoke is not accepted as a plan change. Keep AC-6 active.

## Required Implementation Plan

Do this next, in order:

1. Repair AC-6 immediately. Kill stale `sglang::router` and worker processes. Boot DSA-default with `development/serve_native_nsa.sh` at `MEM_FRACTION_STATIC=0.85`, radix on, no DS flags, and capture `/get_server_info` plus boot excerpts proving `enable_double_sparsity=false`, `double_sparsity_config=null`, no `token_label_table`, and the full KV pool.
2. Run the DSA-default client SLO confirmation with the proper steady-state methodology, not `WARMUP=0`: use `development/benchmark_baseline.sh` with conc 16/32/64, `NUM_PROMPTS=320`, `TRIALS=1`, `WARMUP_SECONDS=120`, and `MEASUREMENT_WINDOW_S=600`. Record tracked JSONL/meta or an exact summary sufficient to recompute completed/errors/P99 TTFT/per-request TPS. The AC-6 report must show P99 TTFT `<22.0` and TPS `>=30` at every concurrency, or keep AC-6 failed.
3. Boot DS opt-in at the locked product point: `SIGNATURE_DTYPE=int8`, `MEM_FRACTION_STATIC=0.7`, and `RADIX_FIXTURE_ARTIFACT=runs/20260530_dsv32_loop6/ds_radix_fixture_state_int8.json`. Capture `/get_server_info` proving `enable_double_sparsity=true`, `signature_dtype=int8`, `disable_radix_cache=false`, and the fixture path; capture boot excerpts proving the int8 `token_label_table` on all 8 ranks.
4. Rewrite `ac6_optin_dsa_default_product.md` so it only claims what the artifacts prove. Do not say the servers differ only by DS enablement unless the captured fields support it. Keep the WARMUP=0 smoke as admission-only context or remove it from the SLO proof.
5. After AC-6 is verified, complete AC-7 exactly as planned: 3-trial DS+DSA lifted-point sweep, 120s warmup / 600s windows, radix-on both sides, refreshed `ac11_resweep.md` / `ac11_analysis.md`.
6. Then complete AC-8: run the lifted ~70K-token `/generate` probe and record either HTTP 200 with max-token capacity and no instability, or a characterized ceiling. Do not re-record the old 400 silently.
7. Start AC-10 only after AC-3 through AC-9 are verified; then implement the adjustable-`top_k` sparse-matmul path or learned selector and record NIAH 4K/16K/64K recall deltas plus TPS/TTFT cost.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md` mutable section:

- Plan version moved to Round 10 Review.
- Added an R10-review plan-evolution row.
- Moved AC-9/task10 to Completed and Verified.
- Kept AC-6/task7 Active as partial and added a blocking AC-6 product-proof issue.
- Left AC-7, AC-8, and gated AC-10 active/pending; no immutable section was changed.

## Validation Performed

- `git log --oneline -30`
- `git show --stat --oneline d6e884aa9 daad92923 2fd2c6937`
- Inspected `test/manual/test_double_sparsity_v32.py` and AC-9 artifacts.
- `python3 -m py_compile test/manual/test_double_sparsity_v32.py`
- Importlib helper check for `_summarize_prompt_tokens` fail-closed behavior.
- Inspected `ac6_optin_dsa_default_product.md`, `get_server_info_keys.json`, full AC-6 server-info captures, boot excerpts, and `dsa_default_slo.txt`.
- Read `development/CLIENT_SLOS.md`, `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh`, `development/benchmark.sh`, and `development/benchmark_baseline.sh`.
- `git diff --check 57f86b66f..HEAD`

NOT COMPLETE
