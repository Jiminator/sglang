Mainline Progress Verdict: ADVANCED

Goal Alignment Summary:
ACs: 6/11 met | Forgotten items: 0 | Deferred items: 0 | Active tasks remaining: 6

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Then read the Pensieve review pipeline, `goal-tracker.md`, round 1-3 summaries and review results, the Round 4 summary, commits `6acdfb94f`, `f2bc1eb6a`, `2220a793f`, changed launcher/benchmark scripts, smoke artifacts under `runs/20260528_dsv32_mvp/`, and the relevant design references in `development/past_implementations/study/07-mvp-proposed-architecture.md`.

Verification run:

```bash
bash -n development/serve_double_sparsity.sh development/serve_native_nsa.sh development/benchmark.sh development/benchmark_baseline.sh
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
# 254 passed, 24 warnings in 11.73s
```

Smoke artifact verification:

- All six raw JSONLs exist locally under `runs/20260528_dsv32_mvp/smoke_results/`.
- Observed durations exceed the 30s smoke window: DSA 178.77/167.87/167.67s, DS 533.34/514.55/514.80s for conc 16/32/64.
- All six sidecars are valid: cluster model path, TP=8, `kv_cache_dtype=fp8_e4m3`, `page_size=64`, `disable_radix_cache=true` on both sides, `NUM_PROMPTS=64`, `WARMUP_SECONDS=0`, `MEASUREMENT_WINDOW_S=30`.
- `benchmark_compare.py --baseline ... --ds ...` rerun on a smoke pair exited 0; committed comparator JSON confirms GPU id 0, TP=8, page 64, radix-off parity, and matching concurrency.

## Acceptance Criteria Audit

| AC | Status | Evidence if met | Blocker if not met | Justification if deferred |
|----|--------|-----------------|--------------------|---------------------------|
| AC-0 | MET | task1 verified in round 0; task2 verified in round 3 via `ac0_capture_positive.json` with 11 `per_token_slot_sha` entries, 61 `per_layer_written_all_true`, no error; unit suite green. | - | - |
| AC-4 | MET | round 1 dry-run + full calibration produced and validated `/models/dsv32-fp8-channel-mask.safetensors`, SHA `7b3207cae888c141173703384bfd7c8974b7adb64b1fddbdacac3ab26c7d6ac6`. | - | - |
| AC-1 | MET | round 3 `/get_server_info`, `/generate`, and invalid-mask rejection artifacts verified; launcher defaults now also point at cluster weights. | - | - |
| AC-1.1 | MET | `ac1_1_genuine_sparsity.json`: long prompt > top_k, `sparsity_rate=0.1045`, `selected_tokens=2048`, `dense_fallback=0`. | - | - |
| AC-1b | NOT MET | - | Chunked-prefill probe has not run; must precede AC-11. | - |
| AC-6 | MET | round 2 review verified regular CUDA graph capture completed all 52 batch sizes, distinct from disabled piecewise CUDA graph. | - | - |
| AC-8 / AC-9 | MET | Round 4 smoke DS+DSA benchmark pair verified: all six JSONLs + sidecars exist locally, duration guard passed, radix-off parity held, cluster model path on both sides. | - | - |
| AC-10 | NOT MET | - | Radix fixtures and no-env-override validator/launcher flip have not been implemented/run; DS final launch still passes `--disable-radix-cache`. | - |
| AC-11 | NOT MET | - | No 3-trial radix-on 120s/600s sweep. #F must be resolved or explicitly handled before this comparison is honest. | - |
| AC-12 | NOT MET | - | Full NIAH 4K/16K/64K + MMLU 5-shot gate has not run. | - |
| AC-Q | NOT MET | - | Paired quality smoke has not run. Existing harness currently assumes simultaneous DS and DSA servers, which contradicts the single-node sequential plan. | - |

## Forgotten Items Detection

No original plan tasks are forgotten after the tracker update. Tasks 1-8 and 10 are completed/verified; tasks 9 and 11-15 remain active. There are no items marked complete in the summary that remain unverified after this review: task7/task8 are now verified and moved to Completed and Verified.

Raw `*.jsonl` artifacts are ignored by repo policy (`.gitignore:179`) but exist locally; the committed sidecars and comparator JSON preserve the reviewed metrics. The final evidence bundle should still account for the raw JSONL storage/transfer policy before task15 closes.

## Deferred Items Audit

The `Explicitly Deferred` section is empty. No deferral is accepted for AC-Q or Tier 2; those are incomplete active work, not optional follow-up.

## Goal Completion Summary

```text
Acceptance Criteria: 6/11 met (0 deferred)
Active Tasks: 6 remaining (task9, task11, task12, task13, task14, task15)
Estimated remaining rounds: 4-6
Critical blockers: #G for AC-Q; #F before AC-11
```

## Mainline Drift Audit

Round 4 had a clear and singular objective: task7/task8, the Tier-1 smoke benchmark pair plus comparator. It advanced mainline ACs rather than circling on side issues. The launcher fixes (#D/#E), DSA mem-fraction knob, and `NUM_PROMPTS` override were side issues, but all were directly blocking valid task7/task8 evidence under real hardware load.

`development/past_implementations/study/07-mvp-proposed-architecture.md` reinforces that the final MVP target is still the DSA baseline comparison with radix/CUDA settings represented and quality gates passing, not merely a smoke benchmark. The smoke is therefore valid progress, not completion.

```text
Mainline Progress Verdict: ADVANCED
Blocking Side Issues: 1 (#G AC-Q sequential harness gap)
Queued Side Issues: 2 (#F AC-11 effective-concurrency, stale calibrate.py operator recipe)
```

## Implementation Review

No high-signal defects were found in the Round 4 benchmark implementation itself.

- `development/serve_double_sparsity.sh` and `development/serve_native_nsa.sh` now default `MODEL_PATH` to `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`.
- `development/serve_native_nsa.sh` keeps radix cache on by default for the future AC-11 path and only adds `--disable-radix-cache` when `DISABLE_RADIX_CACHE=1`.
- The DSA baseline `MEM_FRACTION_STATIC=0.85` knob is justified by the smoke OOM and does not affect the DS launcher default.
- `NUM_PROMPTS` is env-overridable with the full-sweep default preserved at 320.
- The smoke report is correctly labeled as not AC-11 evidence and documents the DS TTFT/effective-concurrency caveat.

One mainline gap was found for the next task:

1. **AC-Q runner is incompatible with the single-node sequential contract.**

   The plan requires DSA references first, then DS, because two TP=8 servers cannot co-reside on one 8-GPU node. But `test/manual/test_dsv32_quality_smoke.py:231-234` skips unless both `DS_BASE_URL` and `DSA_BASE_URL` point at running servers, and `_run_paired` interleaves DSA and DS calls per prompt at `test/manual/test_dsv32_quality_smoke.py:250-260`. This blocks the next Tier-1 item unless Claude rewrites the harness or runs an equivalent sequential workflow.

   Required fix: add explicit DSA capture and DS compare modes, or a small replacement script, that writes the 20 DSA reference outputs under `runs/20260528_dsv32_mvp/`, then later loads that artifact while DS is running and records all four gates.

## Goal Tracker Updates Applied

Updated only the mutable tracker section:

- Moved task7/task8 to Completed and Verified.
- Added Round 4 review to the Plan Evolution Log.
- Marked task9/AC-Q as blocked on #G.
- Added #G as a Blocking Side Issue.
- Moved #F to Queued Side Issues because it blocks future AC-11, not the current AC-Q task.

## Action Items

### Mainline Gaps

1. Complete task9 / AC-Q next. First fix the quality-smoke workflow so it supports single-node sequential DSA-reference capture and DS comparison. Then run all four gates and save the artifact under `runs/20260528_dsv32_mvp/`.
2. Continue Tier 2 in plan order: task11 AC-10 radix flip, task12 AC-1b chunked-prefill probe, task13 AC-11 radix-on sweep/comparator, task14 AC-12, then task15 bundle.

### Blocking Side Issues

1. #G: AC-Q harness currently requires simultaneous DS and DSA servers. This blocks the next mainline task.

### Queued Side Issues

1. #F: DS effective concurrency is KV-pool limited at `mem_fraction_static=0.6`; resolve or explicitly account for it before AC-11 so TTFT is not a queueing artifact.
2. `calibrate.py` operator-facing recipe remains stale after the FP8 sharded calibration redesign.

## Stagnation Check

Not stagnant. The last four reviewed rounds each converted blockers into verified artifacts: AC-4 in round 1, AC-1/AC-1.1/AC-6 in round 2, AC-0/strict AC-1 in round 3, and AC-8/AC-9 in round 4. There are recurring hardware-discovered issues, but they are being closed rather than repeated. The stagnation circuit breaker does not apply.

Original plan work remains pending.
