# Loop 12 — Round 1 Summary

## Mainline objective
Fix the Codex [P1] AC-8 gap: the perf wrapper ran the stock generated-shared-prefix default
(64 groups × 16 = 1024 requests across 64 prefix groups), NOT the loop-11b conc-64 workload
(1 prefix group, all prompts sharing the system prompt). Re-prove AC-8 on the correct workload.

## Outcome: AC-8 now validly PASS; branch corrected and re-pushed

### [P1] perf wrapper GSP grouping — FIXED + re-proven
- `benchmarks/bench_double_sparsity.py`: added `GSP_NUM_GROUPS = 1`; `build_bench_cmd` now passes
  `--gsp-num-groups 1 --gsp-prompts-per-group <num_prompts>` (over stock `bench_serving`). The
  verdict records `gsp_num_groups`, `gsp_prompts_per_group`, `expected_prompts`, `actual_completed`,
  `request_shape_ok`, and the wrapper **fails closed** if `actual_completed != num_prompts` so a
  future dataset-shape drift cannot pass silently.
- **AC-8 rerun on the live DS server (corrected workload)**: `--gsp-num-groups 1
  --gsp-prompts-per-group 256`, **actual_completed = 256** (`request_shape_ok = true`),
  **p50 decode TPS 35.05** (≥ 24.2), **P99 TTFT 22.90 s** (≤ 30.1), `parity: true`. M6 (DS active on
  long context) re-passed in the same boot. Evidence replaced in `development/loop12/perf_evidence/`.
  (The 1-group shape has maximal prefix reuse → cheaper prefill → more decode headroom than the prior
  wrong-shape run; native-DSA on the same base was 26.06 / 46.50 s.)

### [P3] plan-tracking markers in shipped comments — CLEANED
- Stripped `AC-`/`DEC-`/`Milestone`/`Loop-N`/`[R-N]` workflow markers from durable shipped code and
  test comments (`validator.py`, `selection_kernel.py`, `metrics.py`, `channel_mask.py`,
  `page_table_adapter.py`, `deepseek_v2.py`, `calibrate.py`, `test_double_sparsity_unit.py`,
  `test_ds_abort_path.py`, `bench_double_sparsity.py`), rewording them as technical statements.
- Deliberately preserved: `DeepSeek-R1`/`V3/R1` model names (not round markers); pre-existing base-code
  `Step N`/`Phase N` in `tokenizer_manager.py` / `server_args.py` / `model_runner.py` /
  `runner/decode_cuda_graph_runner.py` (not my changes — surgical principle); the
  `benchmarks/DOUBLE_SPARSITY.md` provenance doc's factual "loop-11b reference" baseline citation.

## Files changed (v2 clone, R1)
- `benchmarks/bench_double_sparsity.py` (grouping pin + shape guard + verdict fields)
- `benchmarks/DOUBLE_SPARSITY.md` (corrected perf numbers + command + completed count)
- 8 DS modules/tests (comment-marker strip; no logic change)
- v2 commits: `5dcd73ca6` (wrapper fix + marker strip) + `f05326636` (doc); branch re-pushed.

## Validation
- AC-8 rerun: 256/256 completed, 35.05 TPS / 22.90 s, parity true (1 group). Evidence:
  `development/loop12/perf_evidence/verdict.json`, `m6m8_eval_r1.out`.
- Final sweeps green: AC-1 42-file diff no dev scaffolding; AC-2 0 dropped-module refs + 0 plan
  markers in diff; AC-3 `import sglang` OK; 94 unit tests pass after the comment strip.
- Branch re-pushed to `Jiminator/sglang` (HEAD `f05326636`).

## Remaining Items
None. All 10 ACs pass (AC-8 now on the correct workload); both Codex findings resolved. PR can be
opened at `https://github.com/Jiminator/sglang/pull/new/double-sparsity-v2`.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260619-perf-parity-pin-request-shape
- Notes: Perf-parity evidence must pin the EXACT request shape of the reference workload, not just the
  per-request lengths — the stock generated-shared-prefix dataset is built from
  `--gsp-num-groups` (default 64) × `--gsp-prompts-per-group` (default 16) and ignores
  `--num-prompts`, so omitting those flags benchmarks a different request distribution. Pin
  `--gsp-num-groups 1 --gsp-prompts-per-group <n>` and assert `actual_completed == expected` in the
  perf gate so a silent dataset-shape drift fails closed.
