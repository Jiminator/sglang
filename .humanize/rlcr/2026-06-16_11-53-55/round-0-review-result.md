# Round 0 Review Result

Mainline Progress Verdict: ADVANCED

Claude made real mainline progress on op-point re-establishment: the GLM mask recipe correction is justified by the recall evidence, the content-hash validator path passes the focused unit suite, and the radix/capacity evidence substantially advances AC-0/AC-6/AC-7. The completion claim is still false. M-B and AC-8 are not publishable yet.

Goal Alignment Summary:
`ACs: 10/10 addressed | Forgotten items: 5 | Unjustified deferrals: 2`

I updated the mutable section of `.humanize/rlcr/2026-06-16_11-53-55/goal-tracker.md` to keep task7/task8/task9/task10/task11 active and to record the rejected completion claim. I did not modify the immutable section.

## Mainline Gaps

1. **AC-9 is not met: the published cross-side comparison is comparator-refused and same-memory is deferred.**

Evidence: `development/loop11b/runs/20260616_mb/DS_absolute_verdict.md:44-52` says the official `benchmark_compare.py --ac11` refused the comparison, then reports ratios from `extract_verdict.py`; the same paragraph defers the SAME-MEMORY op-point. Re-running the comparator on the current raw artifacts returns `rc=2` with `DSA commit_sha='94313249...' disagrees with DS commit_sha='72cb2475...'`. AC-9 requires comparator acceptance for each matched op-point and both production-envelope and same-memory comparisons.

Required implementation plan:
- Re-run DS and DSA SLO artifacts from one clean HEAD, with the final `serve_native_nsa.sh` caps already present before either side is measured.
- Produce two separate artifact trees and comparator invocations: `production_envelope` (DS 0.8 / DSA 0.85) and `same_memory_080` (DS 0.8 / DSA 0.8).
- Commit the accepted `ac11_report.md` and `ac11_verdict.json` for both trees. Do not publish a manually extracted cross-side ratio table as the AC-9 verdict.

2. **AC-2/AC-3 publication is undermined by admission-capped conc-64 data.**

Evidence: `development/loop11b/runs/20260616_mb/DS_absolute_verdict.md:16` records DS achieved concurrency as `58.9 (admission-capped <64)`, and `:41` repeats that DS is admission-capped below 64. The plan says DS must not be admission-capped below nominal concurrency at conc <= 64, and AC-0.3 also asks for conc-64 running-req peak >=61. A documented SLO fail is acceptable only after the locked workload is actually exercised at the required concurrency.

Required implementation plan:
- Before the rerun, add an explicit per-trial admitted/running-request summary to the benchmark sidecar or run-order ledger.
- Re-run conc 64 with the locked `max_running_requests=64`/`cuda_graph_max_bs=64` op-point and verify the DS trial reaches nominal concurrency or records a running-request peak >=61 with an explanation that is accepted by the comparator/report.
- If DS still cannot reach the nominal workload, report that as an AC-2/AC-3 measurement failure, not as a complete SLO verdict.

3. **AC-4 was not measured as specified; a sweep TPOT ratio is not the loop-11 task8 tax guard.**

Evidence: `development/loop11b/queue.md:38` says task7 was "DERIVED FROM SWEEP"; `development/loop11b/results.md:60-61` claims AC-4 pass from per-request TPOT p50 ratios. The actual `tax_guard.sh` at `development/loop11b/runs/20260616_mb/tax_guard.sh:1-39` is a bench_one_batch driver, but Claude's summary says it was not usable, and no committed AC-4 output gives the required bs64 same-batch decode-window ratio or bs30 <=380k us graph-mode window.

Required implementation plan:
- Repair the tax harness instead of replacing it with sweep TPOT. Patch the bench path so DS validation/binding runs before model execution, or create a serving-backed fixed-batch decode-window probe that measures the same graph-mode one-batch decode window.
- Run DS and DSA at mem 0.8, radix state declared, graph mode, bs64 and bs30, with the sequence shape and warmup declared.
- Commit the raw log plus a small parsed summary containing bs64 DS window, bs64 DSA window, ratio, bs30 window, radix state, mem fractions, and graph-mode confirmation.

4. **AC-5/AC-9 SLO trial evidence is missing for prefix reuse and DS no-op refusal.**

Evidence: `development/loop11b/results.md:68-70` explicitly says per-request prefix reuse, aggregate throughput, and no-op `total_tokens` are not emitted and are treated as queued follow-ups. The committed `.meta.json` sidecars have only launch/workload metadata; they do not contain `cached_tokens`, `dense_fallback_total`, `selected_tokens_mean`, or `total_tokens_mean`. AC-5 says missing dense-fallback or sparse-selection proof blocks publication; AC-9 says GSP shape alone does not prove observed prefix reuse.

Required implementation plan:
- Extend `python/sglang/bench_serving.py` `RequestFuncOutput` and final JSONL/sidecar emission to carry response `meta_info` fields needed for `cached_tokens`/prefix reuse and DS selection counters.
- Add a committed per-trial summary with observed prefix-reuse distribution, `dense_fallback_total == 0`, and `selected_tokens_mean < total_tokens_mean` or a recorded equivalent formula.
- Make the SLO extraction/refusal path fail closed when these fields are absent for DS trials.

5. **AC-8 evidence discipline is incomplete: required raw artifacts are ignored, `queue.md` is stale, and no push occurred.**

Evidence: `.gitignore:62`, `.gitignore:179`, and `.gitignore:283` ignore logs, JSONLs, and `artifacts/`; `development/loop11b/results.md:80-86` says the `.jsonl`/`.log`/mask blobs are gitignored. `development/loop11b/queue.md:32-42` still marks task1 in progress, task2/task3 pending, task8 RUNNING, task9 pending, while results claim completion. `development/loop11b/runs/20260616_ma/capacity_ds_evidence.md:3-4` also still names the superseded `a4be98c4` mask in capacity evidence. Claude also explicitly did not push, despite AC-8 saying `git push` every round.

Required implementation plan:
- Force-add a bounded evidence package or produce committed lossless summaries with hashes that allow the verdict to be reproduced without local ignored files. Include raw or hashed references for bench JSONLs, per-boot logs, run-order ledger, per-trial reuse/no-op summaries, server_info, fixture, mask hashes, and exact commands.
- Rewrite `queue.md` so task statuses match the post-review state; remove the stale `a4be98c4` capacity claim or replace it with the final `35155ac4` evidence.
- Push the branch to an owner-approved remote/branch. If the public upstream cannot be used, get explicit owner direction and record the replacement remote/transfer path before claiming AC-8.

## Blocking Side Issues

- The current comparator code still observes no-op fields but the benchmark does not emit them for the SLO trials. This is blocking for AC-5 publication until the benchmark sidecar is extended and the report refuses missing fields.
- The current run order was block-scheduled and labeled unpaired. The plan permits labeling, but the rerun should use the enforceable alternating run order unless the owner explicitly accepts an unpaired hard verdict before publication.

## Queued Side Issues

- Production-facing text still leaks plan vocabulary. For example, `development/serve_double_sparsity.sh:20-24` and `:168-177` mention DEC/loop identifiers in the production launcher output, despite the plan's instruction to use domain language in serve-script/runbook/CLI-help edits. Clean this while finishing AC-UX, but do not let it displace the measurement fixes above.

## Goal Alignment

- AC-0: Advanced; op-point evidence exists, but stale capacity prose should be corrected.
- AC-2/AC-3: Attempted, but not publishable due comparator refusal, missing required evidence, admission-capped conc-64, and same-memory omission.
- AC-4: Not satisfied; needs the actual tax guard.
- AC-5: Recall side advanced; SLO-trial no-op refusal evidence missing.
- AC-6: Addressed by existing table-free DS path and no dense fallback probe evidence.
- AC-7: Addressed by DSA capacity probe evidence.
- AC-8: Not satisfied; ledger/evidence/push gaps remain.
- AC-9: Not satisfied; same-memory deferred, comparator refused, reuse/run-order evidence incomplete.
- AC-UX: Mostly advanced; production-facing plan terminology remains.

Do not stop. Complete the active mainline tasks in the order task7 -> task8 -> task9 -> task10 cleanup -> task11 close-out.
