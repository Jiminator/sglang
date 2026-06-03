# Round 0 Review Result

Mainline Progress Verdict: ADVANCED

Claude advanced the mainline by landing useful oracle/scorer code and real DS measurements, but the work is not complete against the original plan. The round must continue: several acceptance criteria are still missing required evidence, and some claimed conclusions are inferred from incomplete oracle data.

Plan source note: the review prompt names `development/loop7/refined_plan_v1.md`, but that file is not present in the repo or loop-start commit. I reviewed against `development/loop7/plan.md`, `.humanize/rlcr/2026-06-01_09-27-07/round-0-prompt.md`, and the updated goal tracker.

## Mainline Gaps

1. **AC-1/AC-2: The oracle fails open and already dropped the 64K evidence.**

   Evidence: `_maybe_record_recall_oracle()` silently returns when no active trial exists, filters out-of-range needle positions instead of rejecting them, and catches every exception without surfacing a failure artifact (`python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:853`, `:862`, `:885`). The artifact admits 64K records are absent and the 64K scorer-limited verdict is inferred (`development/loop7/m0_oracle_finding.md:9`, `:24-26`). This violates the plan’s no-silent-guessing/no-silent-oracle-failure requirement and prevents binding M0 closure for 64K.

   Required fix: make oracle-enabled mode fail closed for invalid/missing active trial state, invalid spans, and payload exceptions. Emit an explicit failure record keyed by `(request_id, trial_id, layer_id, decode_step)` before aborting the trial. Remove the span filtering that hides out-of-range spans. Re-run oracle sweeps for 4K/16K/64K with expected-record-count checks and no inferred 64K verdict.

2. **AC-2/AC-3: The claimed recall-uplift measurement lacks the required DSA, MMLU, dense, and TP evidence.**

   Evidence: the plan requires DS-vs-DSA same-node artifacts, MMLU re-anchor, within-budget parity, dense-DS non-regression, and TP=8 cross-rank selected-index equality (`round-0-prompt.md:235-241`). Claude’s own docs say DSA re-confirmation and MMLU are still pending (`development/loop7/m0_baseline.md:31`, `development/loop7/m0_decision.md:28`, `development/loop7/m1_cosine_finding.md:29`). The new TP test coverage is not parameterized for the new scorer flags and is only the existing TP=2 CPU harness.

   Required fix: run the full task12 measurement matrix before any AC-2/AC-3 closure claim: DSA same-node NIAH reference, DS baseline, chosen scorer/hybrid, NIAH at 1K/1.5K/4K/16K/64K, MMLU at the Loop-7 op-point, dense-DS/within-budget recall, and TP cross-rank equality for every scorer/anchor flag.

3. **AC-3: Uniform cosine is a useful research signal but not an acceptable landed selector.**

   Evidence: cosine improves 16K from 5% to 40%, but regresses 4K from 75% to 25% (`development/loop7/m1_cosine_finding.md:11-14`, `:24-29`). It is forced onto the eager fallback when `scorer_norm != off` (`python/sglang/srt/models/deepseek_v2.py:2235-2249`), while the graph-safe selector has no scorer-normalization parameter (`python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:957-985`). This does not meet the plan’s production-ready flag-gated non-regression bar.

   Required fix: implement a single length/budget-conditional hybrid scorer as the next landed candidate: raw scoring for <= 8K-token contexts, cosine scoring for longer contexts, with the exact same branch carried in the graph-safe Triton path. Keep `off` byte-identical. Then measure it; do not present uniform cosine as AC-3 closure.

4. **AC-3 tasks 9-10 are missing.**

   Evidence: the plan requires independently flag-gated head-aggregation and anchor-budget variants (`round-0-prompt.md:237-241`). The code only adds `scorer_norm=cosine`; there is no head-aggregation scorer variant and no recency/global/strided anchor-budget ablation.

   Required fix: add independent config fields for head aggregation and anchor-budget mode, implement both in the selector path, add CPU and TP multiprocess tests, and include them in the per-variant measurement table.

5. **AC-4: Tier-2.A is now required for the 4K regime but has not started.**

   Evidence: the M0 decision says the oracle-uplift gate is met at 4K (`development/loop7/m0_decision.md:9-11`). The plan makes the lifted-budget ABI and decode path conditional on that gate (`round-0-prompt.md:212-215`, `:242-246`). No `enable_lifted_budget_decode` / `lifted_budget_top_k` ABI, compact remap, sparse decode path, correctness tests, or landing disposition exists.

   Required fix: execute task13-task17 exactly as planned for the bounded 4K-class regime: design the ABI, add validators, implement physical-slot -> `page_table_1_flattened` -> compact-index remap, mask `-1` before dequant, add reference-attention/prefix-sharing/padding/duplicate/TP tests, and write the production-ready-or-disposition record.

6. **AC-6/M4: Consolidation and perf guardrails are absent.**

   Evidence: task19 requires DS-vs-DSA recall report, Tier-1 spine non-regression, and perf guardrails at conc-1/16 (`round-0-prompt.md:248`). No TTFT/decode TPS/GPU memory/graph-replay/admission consolidation artifact exists for the new scorer or any lifted-budget path.

   Required fix: after the selector and Tier-2.A disposition are complete, run the conc-1/16 guardrail suite and write the final consolidation report plus the task20 gate-supersession decision record.

## Blocking Side Issues

1. **Graph/capture helper silently ignores non-default scorer config.**

   `capture_decode_step()` chooses `retrieve_topk_graph_safe()` whenever scratch exists, with no check for `selector.config.scorer_norm` (`python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py:283-321`). That would run raw scoring under capture for a cosine/hybrid-configured selector. Fix this before graph-safe scorer validation: thread scorer config into graph-safe selection, or fail fast if a non-off scorer would enter a raw graph path.

2. **The DS launcher defaults do not match the Loop-7 op-point.**

   The runbook says the `serve_*.sh` scripts encapsulate the int8 / mem 0.7 operating point, but `development/serve_double_sparsity.sh` defaults to `SIGNATURE_DTYPE=fp16` and `MEM_FRACTION_STATIC=0.6` (`development/serve_double_sparsity.sh:47-54`). That can silently reproduce the old unservable/misleading regime. Align the defaults or add a fail-fast Loop-7 measurement mode.

## Queued Side Issues

1. Production code comments/tests contain plan-specific markers despite the implementation-note ban (`AC-*`, `task*`, `Loop-7`, `Tier-2`). This should be cleaned before merge, but it does not block the next measurement round.

2. The tracker and review prompt referenced missing `development/loop7/refined_plan_v1.md`. I corrected the mutable tracker to point future work at the available plan sources.

## Goal Alignment Summary

ACs: 4/6 addressed | Forgotten items: 0 after tracker update | Unjustified deferrals: 6

AC-1, AC-2, AC-3, and AC-5 have partial progress. AC-4 and AC-6 have not materially started. The unjustified deferral buckets are: oracle fail-closed/64K evidence, DSA/MMLU/dense/TP validation, hybrid/graph-safe scorer, missing head/anchor variants, Tier-2.A bounded implementation, and M4/perf consolidation.

## Tracker Update

I updated `.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md` mutable section only:

- moved task18 to completed/verified with the 64K served evidence;
- kept task1/task2/task5 completed only for unit-verified code;
- moved task3/task4/task6/task7/task8-task17/task19/task20 into Active Tasks with current blockers;
- added the oracle fail-open, graph scorer config, and launcher op-point mismatch as Blocking Side Issues;
- recorded the missing `refined_plan_v1.md` source correction and the partial status of M1 cosine.

## Required Next Implementation Plan

1. Replace the oracle best-effort path with strict oracle-mode behavior: config-borne enable/path/trial-file fields, fail-closed active trial validation, explicit failure records, no out-of-range span filtering, and expected-record-count assertions in `niah_oracle_sweep.py`.

2. Align `serve_double_sparsity.sh` with the Loop-7 measurement op-point or add a required `LOOP7_MEASUREMENT=1` mode that sets `SIGNATURE_DTYPE=int8`, `MEM_FRACTION_STATIC=0.7`, and logs the effective `double_sparsity_config`.

3. Re-run M0 on 8xH200: DS and DSA same-node NIAH at 1K/1.5K/4K/16K/64K, oracle records for 4K/16K/64K with no missing lengths, stride fields recorded, binomial CIs, and MMLU re-anchor. Then re-adjudicate task7 from the corrected evidence.

4. Implement the AC-3 selector set in one pass: `scorer_norm=hybrid_raw_cosine` as the landed candidate, plus independent head-aggregation and anchor-budget flags. Port hybrid scoring into `retrieve_topk_graph_safe()`/Triton, and make graph capture reject unsupported non-off scorer modes instead of silently raw-scoring.

5. Add tests before new measurements: default-off byte equivalence, graph-safe hybrid vs eager equality, graph-replay allocation, TP cross-rank equality for each scorer/anchor mode, and fail-fast guard preservation.

6. Measure task12 fully: per-variant NIAH recall with N>=50 for binding 16K, 64K included, MMLU, dense-DS, within-budget parity, DSA same-node reference, and perf deltas for eager vs graph-safe paths.

7. Execute task13-task17 for the 4K budget-limited regime: add `enable_lifted_budget_decode` and `lifted_budget_top_k`, implement compact-index remap and sparse decode, add safety/correctness tests, then write the landing/disposition record.

8. Finish M4: write the consolidated DS-vs-DSA recall/perf/non-regression report and the final strategic-gate supersession decision record.

## Verification Run During Review

Targeted new unit tests passed:

`python3 -m pytest test/registered/unit/layers/attention/test_selection_recall_oracle.py test/registered/unit/layers/attention/test_oracle_sink_and_force.py test/registered/unit/layers/attention/test_scorer_norm.py -q`

Result: 33 passed, 5 warnings.

