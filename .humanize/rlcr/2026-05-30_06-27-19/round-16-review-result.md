# Round 16 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: ACs: 9/10 addressed, 8/10 met | Forgotten items: 0 | Unjustified deferrals: 0

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-16-prompt.md`, `round-16-contract.md`, `round-16-summary.md`, `goal-tracker.md`, Round 13-15 summaries/reviews, commit `9915630ca`, the AC-8 artifacts, the AC-5 report/attribution artifacts, and the scheduler/decode files relevant to the remaining strict-SLO blocker.

## Implementation Review

Round 16 completes the requested AC-8 servability check. I found no high-signal defect in the AC-8 evidence.

The committed probe payload is the named fixture the plan allowed: `development/loop6/probe_64k.json`. I independently tokenized it with `PreTrainedTokenizerFast(tokenizer_file=/cluster-storage/models/deepseek-ai/DeepSeek-V3.2/tokenizer.json)` and got `70759`, matching both the payload estimate and the server-reported `meta_info.prompt_tokens`.

The response artifact records the required admission result: `http_status=200`, `admitted_http_200=true`, `served_max_total_num_tokens=396096`, `prompt_tokens_reported=70759`, `completion_tokens_reported=16`, `finish_reason=length`, and server alive before/after (`runs/20260530_dsv32_loop6/ac8_servability/ac8_probe_response.json:7-18`). The AC-8 report correctly contrasts this with the Loop-5 mem-0.6 HTTP 400 at 69970 tokens and the 53056-token pool (`runs/20260530_dsv32_loop6/ac8_servability/ac8_64k_servability.md:45-60`).

The operating point is also proven: DS enabled, `signature_dtype=int8`, `mem_fraction_static=0.7`, `disable_radix_cache=False`, fixture `f3b67943...`, and `max_total_num_tokens=396096` (`runs/20260530_dsv32_loop6/ac8_servability/ac8_64k_servability.md:9-15`). The retained `/tmp/ac8/ds_boot.log` matches the committed excerpt and has 0 matches for OOM/CUDA-error/Traceback/RuntimeError. The final `Killed` line in that runtime log follows a successful after-probe `/get_server_info` and graceful shutdown, so it is not evidence of probe instability.

## Mainline Gaps

1. **The original Loop-6 plan is still incomplete after AC-8.**

   AC-8 is now verified, but the loop cannot output `COMPLETE`: AC-5 remains directional-only and still fails the strict DS client SLO, and AC-10 remains gated. The verified AC-5 report states the strict misses directly: conc 32/64 P99 TTFT are `25.5 / 111.2 s > 22.0`, and per-request TPS is `17.6 / 11.5 / 9.3 < 30` at every concurrency (`runs/20260530_dsv32_loop6/client_slo_report.md:30-49`). AC-10 must not start until AC-5 strict remediation is done and the Tier-1 spine is fully verified.

## Blocking Side Issues

1. **Strict DS client SLO still blocks the Ultimate Goal.**

   The remaining blocker is not AC-8, artifact hygiene, or more table footprint. The AC-5 attribution says the lifted point is not KV-pool-bound: 64 concurrent requests need roughly 295K tokens, below the 396K pool, while queue p99 grows to `10.5 / 22.3 / 99.4 s` (`runs/20260530_dsv32_loop6/client_slo_int8/attribution_per_conc.txt:21-45`). The same report also shows TPS is already below target at conc 16 because the restored decode batch drops per-request speed (`runs/20260530_dsv32_loop6/client_slo_report.md:41-47`). The next round must attack the DS decode/scheduling operating point, not collect more AC-8 evidence.

## Queued Side Issues

1. **Cross-node wrapper smoke remains future-gated.**

   This does not block R16 or the remaining single-node AC-5 remediation. Before publishing any future cross-node scripted benchmark artifact, run the wrapper with `HOST=<remote>` and capture both the `bench_serving` readiness banner naming that host and the matching sidecar.

2. **DSA-default conc-64 TPS remains below the client threshold.**

   This remains queued under the R12 user decision because it reproduces the pre-existing DSA/H200 baseline and is not introduced by DS.

## Goal Tracker Audit

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified earlier. |
| AC-2 | MET | Feasibility budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, sidecar consumers, launcher, real-mask NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted 0.7 operating point, HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | Evidence/attribution verified; strict DS SLO still fails. |
| AC-6 | MET | Verified in R12 under the user-approved non-regression/opt-in semantics. |
| AC-7 | MET / CHARACTERIZED | Verified in R15 as characterized/soft-met. |
| AC-8 | MET | R16 probe verified: 70759-token `/generate` admitted HTTP 200 at lifted DS int8/mem0.7/radix-on. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. |
| AC-10 | NOT MET | Correctly gated behind AC-5 strict remediation and full Tier-1 verification. |

Forgotten items: none. Every original plan task is represented in Active, Completed, or the gated AC-10 path. Deferred items: none in the Explicitly Deferred tracker section. Plan evolution is valid: R16 completes the R15-required AC-8 step and does not downgrade AC-5 or prematurely open AC-10.

## Required Implementation Plan

1. Make Round 17's single mainline objective AC-5 strict remediation. Do not start AC-10 and do not spend another round on AC-8/AC-7 evidence.

2. Treat the AC-5 failure as a decode-throughput first problem, then a scheduling problem. The evidence already rules out more footprint as the next lever: KV capacity fits, but per-request TPS is below 30 even at conc 16. First profile and optimize the DS decode hot path at the lifted point, centered on `python/sglang/srt/layers/attention/dsa_backend.py` and `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`, while preserving the Tier-1 ABI lock (`indices.shape[-1] == dsa_index_topk`, no adjustable `top_k` yet).

3. Add or reuse a served-workload profiling artifact that breaks conc-16 decode token time into DS selection/top-k, FlashMLA KV decode, token-label write/update, and scheduler overhead. Use that profile to make the smallest code change needed for DS conc-16 per-request TPS to reach `>=30` at the lifted radix-on int8 operating point. A pure queue/admission tweak is not sufficient until this passes, because capping or delaying prefill cannot make a batch-16 decode path exceed 30 TPS/req.

4. After conc-16 TPS passes, tune the scheduling/chunked-prefill operating point for conc 32/64 TTFT without regressing TPS. Any change to locked serving flags such as overlap scheduling, mixed chunking, `chunked_prefill_size`, or `max_running_requests` must be recorded as plan evolution with before/after server-info sidecars; if `--enable-prefill-delayer` is used, account for its current requirement that overlap scheduling is enabled.

5. Rerun the full AC-5 client workload at the final operating point: `NUM_PROMPTS=320`, conc `16 32 64`, 4096 ISL / 512 OSL / ~55% cache, radix-on, single-node TP=8. Publish exact per-request arrays, request-time attribution, server-info sidecars, and a fail-closed verifier equivalent to the existing AC-5 verifier. A strict claim requires every published trial to pass `P99 TTFT < 22.0 s` and `per-request TPS >= 30`.

6. Only after AC-5 strict remediation is verified should AC-10 begin. Then implement the adjustable-`top_k` kernel or selector path and record NIAH 4K/16K/64K recall deltas plus TPS/TTFT cost.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 16 Review.
- Added a `16-review` Plan Evolution row.
- Moved task9/AC-8 from Active to Completed and Verified.
- Refreshed the stale AC-5 active-task note so the next mainline is strict AC-5 remediation.
- Left AC-10 active/gated.
- Moved the cross-node wrapper smoke from Blocking Side Issues to Queued Side Issues because it is future-gated and does not block the remaining single-node AC-5 path.

## Validation Performed

- `git log --oneline -30`
- `git show --stat --oneline 9915630ca`
- Inspected `round-16-contract.md`, `round-16-summary.md`, `goal-tracker.md`, and Round 13-15 summaries/reviews.
- Inspected `ac8_64k_servability.md`, `ac8_probe.py`, `ac8_probe_response.json`, `get_server_info_{before,after}.json`, and `server_log_excerpt.txt`.
- Re-tokenized `development/loop6/probe_64k.json` with the local tokenizer file; computed token count = `70759`.
- Parsed server-info fields for `mem_fraction_static`, `enable_double_sparsity`, `signature_dtype`, `disable_radix_cache`, fixture path, TP size, and served token pool.
- Checked retained `/tmp/ac8/ds_boot.log`: 0 OOM/CUDA-error/Traceback/RuntimeError matches; no live `sglang.launch_server`; all GPUs at 0 MiB.
- Inspected AC-5 report/attribution and relevant scheduler/decode code for the remaining strict-SLO implementation plan.
- `git diff --check 40ccc4b63..9915630ca`

NOT COMPLETE
