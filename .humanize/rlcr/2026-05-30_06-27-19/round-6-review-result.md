# Round 6 Review Result

Mainline Progress Verdict: ADVANCED

ACs: 6/10 addressed | Forgotten items: 0 | Unjustified deferrals: 0

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-6-prompt.md`, `round-6-contract.md`, Round 3-5 summaries/reviews, `goal-tracker.md`, commit `8883848e9`, and the Round 6 AC-5 artifacts under `runs/20260530_dsv32_loop6/client_slo_int8/`.

## Implementation Review

Round 6 did advance the mainline: Claude ran the DS int8/0.7 client workload at all three concurrencies, regenerated an int8 radix fixture state, and tracked sidecars proving `disable_radix_cache=false`, `signature_dtype=int8`, `mem_fraction_static=0.7`, `enable_double_sparsity=true`, and `max_total_num_tokens=396096`. The measured TTFT movement is real: 12.8 / 25.5 / 111.2 s vs Loop-5's 57.7 / 132.9 / 292.0 s, with conc 16 now below the strict 22 s TTFT threshold.

I cannot verify task6/AC-5 as complete yet. There are two evidence/attribution blockers and the original Loop-6 plan still has AC-6 through AC-10 pending.

## Mainline Gaps

1. **AC-5 benchmark evidence is not durably recomputable from tracked files.**

   `client_slo_report.md:14` claims the artifacts include "3x JSONL + `.meta.json` sidecars", but commit `8883848e9` tracks only `client_slo_metrics.txt`, the three `.meta.json` sidecars, `reqtimestats_excerpt.txt`, the report, and the fixture state. The raw benchmark JSONL files are present locally but ignored by git:

   - `git show --name-only 8883848e9` lists no `.jsonl`.
   - `git status --ignored --short runs/20260530_dsv32_loop6/client_slo_int8` shows all three benchmark `.jsonl` files as `!!`.
   - `.gitignore:179` ignores `*.jsonl`.

   The tracked metrics file is useful, but it is not enough to independently recompute the percentile math or verify the no-hidden-failure claim from durable evidence. This repeats the R4 artifact-shape failure in a smaller form: the acceptance artifact depends on ignored local files.

   Required fix: add a tracked AC-5 evidence addendum under `runs/20260530_dsv32_loop6/client_slo_int8/`. Use the existing local JSONL if still present; otherwise rerun the same AC-5 command. The tracked addendum must contain, per conc, completed count, duration, error count or all-errors-empty proof, input/output length distributions, TTFT/TPOT/ITL arrays or exact recomputable percentile source, and the sidecar path used. Do not rely on ignored `*.jsonl` for acceptance evidence.

2. **The required admission-wait vs prefill-compute attribution is not clean enough to validate the spine.**

   The report defines `queue_duration` as admission/queue wait (`client_slo_report.md:28-31`), but the tracked request-time excerpt contains an impossible negative queue duration:

   - `reqtimestats_excerpt.txt:64`: `queue_duration=-23105.27ms`
   - `client_slo_metrics.txt:8-11`: the aggregate attribution says `N=959`, while the benchmark ran 3 x 320 = 960 completed requests.

   A negative queue wait means at least one request-time row is invalid or needs an explicit clock/serialization explanation. The report does not disclose invalid-row handling, does not give per-concurrency queue stats, and uses min TTFT as a "prefill-compute floor" while `ReqTimeStats.forward_duration` is completion-time prefill+decode, not first-token prefill compute. Therefore the report can record the run, but it should not yet claim the spine is "validated" by measured attribution.

   Required fix: reprocess the full server request-time-stat log per concurrency. Track an attribution addendum that records expected rows vs parsed rows, invalid/negative rows and the filtering policy, queue-duration p50/p95/p99, TTFT p99 and residual, and a corrected explanation of what is measured directly vs inferred. If the TPS root cause continues to cite `Decode batch #running-req: 19-20`, add the server decode-batch log excerpt as tracked evidence too.

3. **The original Loop-6 plan remains incomplete.**

   This is not a Round 6 regression, but it prevents any final `COMPLETE`. AC-6 hardware product proof, AC-7 lifted-point DS+DSA re-sweep, AC-8 lifted 64K servability probe, AC-9 real-token-count harness edit plus live rerun, and gated AC-10 are still not completed. The strict client SLO is also still failed: conc 32/64 TTFT are above 22 s, and per-request TPS is below 30 at every conc (`client_slo_report.md:18-22`).

   Directive implementation plan:
   - First close the AC-5 evidence addendum above. Update `client_slo_report.md` so it says "directional characterization" until the tracked attribution is clean; keep the strict SLO miss explicit.
   - Then complete AC-6: boot DSA default with no DS flags, track `/get_server_info`, boot/server excerpts proving `enable_double_sparsity=false` and no DS `TokenLabelTable`, and run the client SLO workload showing DSA-default behavior/perf unchanged. In the same artifact, boot DS opt-in and prove the compact int8 path is selected.
   - Complete AC-9 before or alongside the next live hardware pass: edit `test/manual/test_double_sparsity_v32.py` so the NIAH gate records `usage.prompt_tokens` as `input_tokens`, computes `within_budget` from that token count, renames the old word proxy to `length_words`, and fails closed if usage is absent or inconsistent. Do not change the DS-fair thresholds. Rerun the gate and copy artifacts into `runs/20260530_dsv32_loop6/`.
   - Complete AC-7: run the 3-trial DS+DSA lifted-point sweep at conc 16/32/64 with 120 s warmup and 600 s windows, radix-on proven on both sides, and refresh `ac11_resweep.md` / `ac11_analysis.md`.
   - Complete AC-8: run the lifted 0.7 ~70K-token `/generate` probe and record either HTTP 200 with `max_total_num_tokens` and no instability, or a characterized new ceiling. Do not silently re-record the old 400.
   - Start AC-10 only after AC-3 through AC-9 are verified. Implement the adjustable-`top_k` sparse-matmul path and record NIAH recall deltas vs the Loop-5 DS baseline, with TPS/TTFT cost.

## Blocking Side Issues

1. **AC-5 evidence/attribution blocks task6 verification.**

   The run exists and is useful, but the ignored JSONL and invalid request-time row mean the AC-5 acceptance bundle is not yet durable enough for verification. This should be fixed before moving task6 to Completed and Verified.

2. **Strict SLO failure is a mainline blocker for the ultimate goal.**

   The TPS regression is not a queued cleanup. It blocks the strict `P99 TTFT < 22 s AND >=30 TPS/req` done criterion and must remain visible as a mainline SLO blocker even if the next round proceeds to AC-6/AC-9 hardening.

## Queued Side Issues

None newly added. The chunked-prefill/scheduling and decode/admission tradeoff work is not a low-priority cleanup; it is part of the unresolved strict-SLO path.

## Goal Alignment Check

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Decision doc verified in earlier reviews. |
| AC-2 | MET | Feasibility budget and binding int8 lever verified in earlier reviews. |
| AC-3 | MET | Int8 table, scale-sidecar consumers, launcher, real-mask NIAH, and microbench verified by R3. |
| AC-4 | MET | Mem-fraction lift and durable no-OOM evidence verified by R5. |
| AC-5 | PARTIAL | Real DS int8/0.7 radix-on run with strong TTFT movement, but evidence/attribution addendum required; strict SLO not met. |
| AC-6 | PARTIAL | Dev checks exist; hardware DSA-default/no-table/SLO product proof remains pending. |
| AC-7 | NOT MET | Lifted-point 3-trial DS+DSA re-sweep remains pending. |
| AC-8 | NOT MET | Lifted-point ~70K-token servability probe remains pending. |
| AC-9 | NOT MET | Harness still needs real `usage.prompt_tokens` budget assertion and rerun. |
| AC-10 | NOT MET | Correctly gated behind full Tier-1 completion. |

Forgotten items: none. Every original plan task is still represented in Active, Completed, or the gated AC-10 path. Deferred items: none explicit; AC-10 is gated, not deferred. Plan evolution is valid only as a directional finding: the report may carry the TPS/admission tradeoff as a blocker, but it must not weaken the strict SLO or mark the loop done.

## Goal Tracker Update

Updated `.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md` mutable section:

- Plan version moved to Round 6 Review.
- Added an R6-review plan-evolution row.
- Kept task6 active as `partial (R6) - evidence/attribution addendum required`.
- Added blocking side issues for the non-durable AC-5 benchmark evidence, invalid attribution data, and the strict SLO miss.
- Left AC-6 through AC-10 active/pending.

## Validation Performed

- `git show --stat --oneline 8883848e9`
- `git show --name-only --format=short 8883848e9`
- `git diff --check 91e9c20a3..8883848e9`
- `git ls-files runs/20260530_dsv32_loop6`
- `git status --ignored --short runs/20260530_dsv32_loop6/client_slo_int8`
- Inspected `client_slo_report.md`, `client_slo_metrics.txt`, all three `.meta.json` sidecars, `reqtimestats_excerpt.txt`, and `ds_radix_fixture_state_int8.json`.
- Used the local ignored JSONL only as a sanity check that the run likely completed; it is not durable acceptance evidence until tracked or summarized.

NOT COMPLETE
