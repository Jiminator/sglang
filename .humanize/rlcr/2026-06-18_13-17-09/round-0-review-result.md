# Round 0 Review Result

Mainline Progress Verdict: ADVANCED

## Goal Alignment Summary

ACs: 9/10 addressed | Forgotten items: 1 | Unjustified deferrals: 0

The implementation substantially advances the mainline: branch hygiene, import closure, pruning, validator gating, calibration, DS-active boot, abort behavior, dependency closure, and dead-code sweeps are all supported by code/evidence I checked. AC-8 is not accepted yet because the saved perf evidence does not run the exact generated-shared-prefix workload it claims to mirror.

I updated the mutable section of `goal-tracker.md`: tasks 1-12 are now Completed and Verified, tasks 13-15 remain Active, and the AC-8 workload issue is tracked as blocking.

## Mainline Gaps

### [P1] AC-8 perf evidence uses the stock GSP defaults, not the loop-11b request grouping

`benchmarks/bench_double_sparsity.py` says the workload "mirrors the loop-11b candidate exactly" at `/sgl-workspace/double-sparisty-v2/sglang/benchmarks/bench_double_sparsity.py:47`, but the command only passes the shared-prefix lengths, range ratio, max concurrency, `--num-prompts`, and seed at `/sgl-workspace/double-sparisty-v2/sglang/benchmarks/bench_double_sparsity.py:68-83`. It does not pass `--gsp-num-groups` or `--gsp-prompts-per-group`.

That matters because stock `bench_serving` defaults generated-shared-prefix to 64 groups and 16 prompts per group at `/sgl-workspace/double-sparisty-v2/sglang/python/sglang/bench_serving.py:2515-2527`, and the dataset actually uses those two fields to generate requests at `/sgl-workspace/double-sparisty-v2/sglang/python/sglang/benchmark/datasets/generated_shared_prefix.py:86-108`. The loop-11b recipe pinned `NUM_GROUPS=1` and passed `--gsp-num-groups "${NUM_GROUPS}" --gsp-prompts-per-group "${NUM_PROMPTS}"` at `/sgl-workspace/sglang/development/benchmark.sh:45-46` and `/sgl-workspace/sglang/development/benchmark.sh:78-90`.

The saved evidence confirms the mismatch: `development/loop12/m6m8_eval_ladder.out:17` shows `gsp_num_groups=64, gsp_prompts_per_group=16, num_prompts=256`; `development/loop12/m6m8_eval_ladder.out:32` reports 1024 successful requests; but the wrapper verdict records only `"num_prompts": 256` at `/sgl-workspace/double-sparisty-v2/sglang/benchmarks/bench_double_sparsity.py:160` and `development/loop12/perf_evidence/verdict.json:11`.

Impact: the 29.34 TPS / 23.29s result may be a good datapoint, but it is not a valid AC-8 proof that the branch reproduces the loop-11b conc-64 workload. AC-8 remains open until the wrapper pins the generated-shared-prefix grouping and the GPU run is repeated.

Directive implementation plan:

1. In `benchmarks/bench_double_sparsity.py`, add `GSP_NUM_GROUPS = 1` and set `gsp_prompts_per_group = args.num_prompts`.
2. Add `--gsp-num-groups 1` and `--gsp-prompts-per-group <args.num_prompts>` to `build_bench_cmd`, keeping stock `sglang.bench_serving`.
3. After parsing the bench result, compute `actual_completed = result.get("completed") or len(result["output_lens"])`.
4. Include `gsp_num_groups`, `gsp_prompts_per_group`, `expected_prompts`, and `actual_completed` in `verdict.json`.
5. Fail the wrapper if `actual_completed != args.num_prompts`; this catches future silent dataset-shape drift.
6. Update `benchmarks/DOUBLE_SPARSITY.md` to show the exact rerun command and actual completed request count.
7. Rerun AC-8 on the DS server, replace `development/loop12/perf_evidence/*`, rerun final sweeps, and push the corrected branch.

## Blocking Side Issues

None separate from the AC-8 mainline gap above.

## Queued Side Issues

### [P3] Shipped comments still contain plan-tracking markers

The plan explicitly says implementation code and comments must not contain plan-specific terminology such as `AC-`, `Milestone`, `Step`, `Phase`, or similar markers at `/sgl-workspace/sglang/development/loop12/plan.md:417`. The shipped branch still has many durable code/test comments with those markers, for example:

- `/sgl-workspace/double-sparisty-v2/sglang/test/registered/unit/managers/test_ds_abort_path.py:1`
- `/sgl-workspace/double-sparisty-v2/sglang/benchmarks/bench_double_sparsity.py:40`
- `/sgl-workspace/double-sparisty-v2/sglang/python/sglang/srt/models/deepseek_v2.py:2080`
- `/sgl-workspace/double-sparisty-v2/sglang/python/sglang/srt/layers/attention/double_sparsity/validator.py:9`
- `/sgl-workspace/double-sparisty-v2/sglang/python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:905`

This is not blocking AC-8, but it contradicts the plan's code-style requirement and should be cleaned before final close-out. Rewrite those comments as durable technical statements without workflow labels.

## Verification Performed

- Read `development/loop12/plan.md`, the round prompt, the goal tracker, and the review pipeline.
- Checked the v2 branch/base/push state: branch `double-sparsity-v2`, base `105e095e005d02a178fb6c5a23bd22ba644c90e4`, HEAD `e6fda2fe9f875a4fae967cc533aff6e585c70269`, remote branch present.
- Verified the v2 diff is 42 files with no `.pensieve/`, `.humanize/`, `development/`, or `SLOS.md`.
- Ran `PYTHONPATH=python python -c "import sglang; import sglang.srt.layers.attention.dsa_backend; import sglang.srt.layers.attention.double_sparsity as ds; print('ok')"` in the v2 clone: passed.
- Ran `PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py test/registered/unit/layers/attention/test_lifted_budget_decode.py test/registered/unit/managers/test_ds_abort_path.py`: 114 passed.
- Checked dropped-module and radix-fixture sweeps: no shipped references found.
- Checked DS-active boot evidence: `selected_tokens=2048`, `total_tokens=5608`, `dense_fallback=0`, bind logs present.
- Checked perf evidence and found the AC-8 GSP grouping/request-count mismatch described above.

Round remains open.
