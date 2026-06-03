# Round 4 Review Result

Mainline Progress Verdict: ADVANCED

Claude advanced the Round-4 contract. The oracle hook is no longer best-effort, activation is config-borne, the TP-worker path records through the production long-context selector, and the local Round-4 sink contains measured 4K/16K/64K records with no hard failures. The repeated oracle fail-open blocker from Rounds 0-3 is resolved.

The original Loop 7 plan is still not complete. AC-1 remains partial because oracle-off CUDA graph allocation evidence and dense/default stride baseline closure are still missing; AC-2/AC-3/AC-4/AC-6 still have substantial plan-derived work. Do not close the loop.

## Part 1: Goal Tracker Audit

| AC | Status | Evidence if met | Blocker if not met | Justification if deferred |
|----|--------|-----------------|--------------------|---------------------------|
| AC-1 | PARTIAL | Task1/task2/task3/task5 are implemented and verified. R4 evidence: `selection_kernel.py:985-1069` records `no_active_trial`, `span_out_of_range`, and exception failures; `oracle_artifact_sink.py:72-80` latches config activation; `niah_oracle_sweep.py:177-219` asserts issued trials recorded. Local sink check: 20 trials each for 4K/16K/64K, 4,880 records per length, 0 failure markers, 0 `recall@2048` invariant violations. | Task4 still lacks CUDA allocation-detector evidence under graph replay. Task6 still lacks the full separated baseline closure, including DSA same-node comparison, dense/default stride reference, and MMLU re-anchor. | Not deferred. |
| AC-2 | PARTIAL | R4 gives a binding oracle rerun for budget-vs-scorer attribution: 4K budget-limited, 16K budget-partial, 64K scorer-limited in `m0_oracle_finding_r4.md` and `oracle_budget_vs_scorer_r4.json`. | The served-recall uplift matrix is still missing DSA same-node reference, stated Clopper-Pearson materiality for the final claims, MMLU/within-budget/dense non-regression, N>=50 binding 16K, and final decision record. | Not deferred. |
| AC-3 | PARTIAL | Head aggregation, anchor modes, physical-hybrid reject, and TP=8 CPU/gloo logical-path determinism are completed and verified through Round 3. | Winning scorer/anchor variants are still not ported into the graph-safe Triton selector, and the full NIAH/MMLU/dense/within-budget/DSA/perf matrix remains absent. | Not deferred. |
| AC-4 | NOT MET | The oracle gate is met for 4K and partially met for 16K, so AC-4 is now evidence-justified. | No `enable_lifted_budget_decode` / `lifted_budget_top_k` ABI, validator path, compact remap, lifted decode implementation, correctness/safety tests, or task17 landing/disposition record exists. | Not deferred. |
| AC-5 | MET | 64K `/generate` servability at mem 0.7 was verified in Round 0 with served/admission separated: `ds_niah_baseline_mem07.json` and `m0_baseline.md`; tracker task18 remains completed/verified. | None. | Not deferred. |
| AC-6 | PARTIAL | Default-off/unit safety remains good, and graph-unsafe modes are guarded at startup. | No final conc-1/16 TTFT, decode TPS/req, GPU memory, graph replay success, admission, or Tier-1 spine non-regression report exists for the chosen path(s). | Not deferred. |

Forgotten items detection: after the tracker update in this review, all original tasks are represented in Active, Completed, or the empty Deferred section. No original-plan task is forgotten. I normalized one task-number drift: Claude's summary says "task #12 (oracle fail-closed + 64K)" was completed, but in the original plan task12 is the Tier-2.B measurement matrix and remains active; the completed Round-4 item is task3.

Deferred items audit: the `Explicitly Deferred` table is empty. The learned/distilled selector remains a queued out-of-scope side issue per DEC-5, not a deferred original task. Tier-2.A and M4 are active/pending, not deferred.

Goal completion summary:

```text
Acceptance Criteria: 1/6 met (0 deferred)
Active Tasks: 12 remaining
Estimated remaining rounds: 4-5
Critical blockers: graph-safe scorer port + full AC-3 matrix; DSA/MMLU/N>=50 served-recall matrix; Tier-2.A ABI/decode/disposition; M4 perf/consolidation/final decision record
```

## Part 2: Mainline Drift Audit

Round 4 had a clear singular objective: make the M0 oracle fail-closed and binding. That objective was a true blocking side issue because prior rounds repeatedly inferred or missed 64K oracle evidence. Fixing it directly served AC-1/AC-2 rather than drifting into incidental cleanup.

Claude is still carrying side issues, but the current pattern is progress, not circularity. Rounds 1-3 advanced AC-3 correctness and safety; Round 4 resolved the repeated oracle blocker.

```text
Mainline Progress Verdict: ADVANCED
Blocking Side Issues: 0
Queued Side Issues: 3
```

Queued side issues are plan-marker cleanup, learned/distilled selector follow-on, and the Round-4 analyzer artifact label/taxonomy cleanup added to the tracker by this review.

## Part 3: Implementation Review

Accepted Round-4 fixes:

1. **Oracle fail-open blocker is resolved.**

   Evidence: `_maybe_record_recall_oracle()` now latches config activation, rejects out-of-range spans without filtering, records `no_active_trial`, `span_out_of_range`, and exception failures, and only silently skips non-primary TP ranks (`selection_kernel.py:985-1069`). The sink creates shared directories on write and has fixed config-borne default trial/sink paths (`oracle_artifact_sink.py:41-60`, `:119-134`, `:191-263`, `:318-349`). The sweep resolves shared paths, truncates the sink, forces decode, and fails on missing trials or hard failure markers (`niah_oracle_sweep.py:127-219`).

2. **The 64K oracle result is measured, not inferred.**

   Evidence: `.sglang_ds_oracle/sink.jsonl` contains 14,640 records: 4,880 each for 4K, 16K, and 64K; every length has 20 trial IDs; every layer has 80 samples per length; no `"failure"` records exist; all records preserve `recall_at_index_topk_matches_selected=true`. `development/loop7/oracle_budget_vs_scorer_r4.json` reports 64K r@2048=15.4%, r@4096=19.6%, r@8192=24.4%, supporting the scorer-limited 64K conclusion.

3. **The validator prevents graph replay from silently recording nothing.**

   Evidence: `validate_double_sparsity()` rejects `recall_oracle=true` unless `--disable-cuda-graph` is set (`validator.py:117-130`), and `deepseek_v2.py:2245-2324` threads `recall_oracle` into `retrieve_topk_graph_safe` without forcing the long-context eager logical selector that previously broke DS.

High-signal issue found:

1. **WARNING: the Round-4 aggregate JSON mislabels the uplift metric and cannot represent the 16K "budget-partial" verdict.**

   Evidence: `analyze_oracle.py:106-118` computes `max(r4096, r8192) - r2048` but writes it to `uplift_gate_recall4096_minus_recall2048`. In the committed artifact this makes 16K show `0.2287`, although `r4096-r2048` is only about `0.0816`; the `0.2287` value is actually `r8192-r2048`. The same script has only `budget-limited` and `scorer-limited` verdicts, so `oracle_budget_vs_scorer_r4.json` says 16K is `budget-limited`, while `m0_oracle_finding_r4.md` correctly characterizes it as `budget-partial`.

   Required fix: update `analyze_oracle.py` to emit separate `uplift_4096_minus_2048` and `uplift_8192_minus_2048` fields, rename any max-based field explicitly, and add a `budget-partial` taxonomy for cases like 16K where budget helps materially but caps far below recovery. Regenerate `oracle_budget_vs_scorer_r4.json` before using it as the Tier-2.A gate/disposition artifact. This does not invalidate the 64K measured scorer-limited conclusion.

Additional evidence hygiene:

- `development/loop7/oracle_trials_index.jsonl` and `.sglang_ds_oracle/sink.jsonl` are ignored and not tracked. The committed aggregate is enough for the current review, but the final decision record should either include durable trial/failure-count metadata or intentionally attach/archive the raw run artifacts.
- Some comments still say `recall_oracle` "forces eager" (`config.py:56-59`, `serve_double_sparsity.sh:76-91`), while the implementation deliberately rides the graph-safe selector with CUDA graph disabled (`deepseek_v2.py:2245-2250`). This is not a runtime bug, but clean the wording before merge.

Verification run during review:

```text
python3 -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py test/registered/unit/layers/attention/test_selection_recall_oracle.py test/registered/unit/layers/attention/test_oracle_sink_and_force.py test/registered/unit/layers/attention/test_scorer_norm.py test/registered/unit/layers/attention/test_scorer_variants.py test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py -q

Result: 344 passed, 24 warnings, 9 subtests passed.
```

## Part 4: Goal Tracker Update Requests

I updated only the mutable tracker section:

- bumped Plan Version to 6 for Round 4 Review;
- moved task3 from Active to Completed and Verified;
- kept task4/task6/task7/task8/task12/task13-task17/task19/task20 active or pending;
- recorded the analyzer uplift-label/verdict issue as queued, with a revisit trigger before task13/final decision consumption;
- corrected the resolved oracle-blocker evidence to the local review test count.

I accepted Claude's request to resolve the oracle hook fail-open blocking side issue. I rejected the summary's task-number implication that original-plan task12 is complete; task12 remains active because it is the Tier-2.B measurement matrix.

## Part 5: Progress Stagnation Check

Development is not stagnating. The same oracle fail-open issue did recur across Rounds 0-3, but Round 4 directly fixed it with config-borne activation, explicit failure records, shared-FS files, forced decode, and complete 64K measurements. Recent rounds have resolved previously identified blockers rather than repeating them unchanged.

## Action Items

Mainline Gaps:

1. Fix and regenerate the Round-4 analyzer artifact fields/taxonomy before task13 or the final decision record consumes it.
2. Complete AC-2/task6/task12: DSA same-node reference, MMLU re-anchor, dense-DS/default stride, within-budget parity, 64K, N>=50 for binding 16K, and CI/materiality rule.
3. Complete AC-3: graph-safe Triton scorer/head/anchor port, eager-vs-graph equality, graph replay allocation evidence, and full non-regression matrix.
4. Complete AC-4/task13-task17: explicit lifted-budget ABI, compact remap, lifted decode, correctness/safety tests, production hardening or disposition record.
5. Complete AC-6/task19-task20: perf guardrails, Tier-1 non-regression, consolidated DS-vs-DSA report, and final strategic-gate supersession decision record.

Blocking Side Issues:

None open. The oracle hook fail-open issue is resolved.

Queued Side Issues:

1. Clean plan-specific workflow markers and stale "recall_oracle forces eager selector" comments before merge.
2. Decide whether to archive raw oracle sink/trial-index artifacts or embed their durable counts in the final decision record.
3. Keep learned/distilled selector work out of scope unless explicitly approved.

Not complete: original-plan acceptance work remains active.
