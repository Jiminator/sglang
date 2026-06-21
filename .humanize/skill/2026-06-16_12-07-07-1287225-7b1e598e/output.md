Read-only review complete. I did not modify code.

**1. RUN_ORDER**

Use paired, adjacent boots because only one TP=8 server fits at a time; the repo notes two TP=8 servers do not fit on one 8-GPU node in [serve_native_nsa.sh](/sgl-workspace/sglang/development/serve_native_nsa.sh:39).

For each op-point separately:

1. `production_envelope`: DSA mem `0.85`, DS mem `0.8`.
2. `same_memory_080`: DSA mem `0.8`, DS mem `0.8`.

For each op-point, run:

```text
c16 t1: DSA -> DS
c16 t2: DS  -> DSA
c32 t1: DSA -> DS
c32 t2: DS  -> DSA
c64 t1: DSA -> DS
c64 t2: DS  -> DSA
```

Each pair must use the same concurrency and same per-concurrency seed. The DS driver defines seeds `[16]=213 [32]=431 [64]=31234` and reuses `SEED="${SEEDS[$CONCURRENCY]}"` inside every trial loop in [benchmark.sh](/sgl-workspace/sglang/development/benchmark.sh:48); baseline does the same in [benchmark_baseline.sh](/sgl-workspace/sglang/development/benchmark_baseline.sh:46).

Log per boot:

- `op_point_id`, `pair_id`, `logical_trial_id`, `side`, `boot_sequence`, side order in pair.
- exact launch command/env, including proof that `PYTORCH_CUDA_ALLOC_CONF` did not contain `expandable_segments`; both server scripts strip that unsafe inherited setting in [serve_double_sparsity.sh](/sgl-workspace/sglang/development/serve_double_sparsity.sh:146) and [serve_native_nsa.sh](/sgl-workspace/sglang/development/serve_native_nsa.sh:85).
- `/server_info` snapshot after ready; `/get_server_info` is deprecated but aliases to `/server_info` in [http_server.py](/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py:621).
- GPU thermal/clock state before boot, at ready, before benchmark, and after benchmark: `nvidia-smi` clocks, pstate, temp, power, memory.
- server log path and benchmark JSONL/meta paths.

If the operator runs all DSA trials first and then all DS trials, or uses the scripts’ internal multi-trial loop under one server boot before the counterpart side runs, label the result `unpaired / block-scheduled`. It may be reported as a diagnostic but not used as the paired drift-controlled comparison.

**2. PER_TRIAL_CAPTURE**

Prefix reuse must be captured from actual request metadata, not inferred from GSP shape. The workload shape is 2253 system tokens + 1843 question tokens = 4096, about 55% reuse in [benchmark.sh](/sgl-workspace/sglang/development/benchmark.sh:35), matching [SLOS.md](/sgl-workspace/sglang/development/SLOS.md:3).

Required per trial:

- per-request `prompt_tokens`, `completion_tokens`, `cached_tokens`.
- per-request `cached_tokens_details.device`, `host`, `storage`, `storage_backend` when present.
- derived `cached_fraction = cached_tokens / prompt_tokens`.
- trial summary: count, min/p25/p50/p75/p95/p99/max and mean cached fraction; nonzero-hit fraction; measured epochs.

These fields exist in response `meta_info`: `cached_tokens` and `cached_tokens_details` are added in [tokenizer_manager.py](/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py:1758). The scheduler computes device/host/storage cached-token details in [schedule_batch.py](/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py:1908). But `bench_serving` currently dumps only `input_lens`, `output_lens`, `ttfts`, `itls`, generated text, and errors in [bench_serving.py](/sgl-workspace/sglang/python/sglang/bench_serving.py:1901), and `RequestFuncOutput` has no `meta_info` field in [bench_serving.py](/sgl-workspace/sglang/python/sglang/bench_serving.py:96). Add a request-metadata sidecar or bench JSONL extension.

AC-5 no-op refusal must record, per DS trial:

- `dense_fallback_total == 0`.
- positive sparse-selection proof: `selected_tokens_mean < total_tokens_mean`, or equivalent `sparsity_rate_mean > 0` with selected/total ratio < 1.

Comparator-recognized JSONL fields are `selected_tokens_mean`, `dense_fallback_total`, and `total_tokens_mean` in [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:255). Existing DS per-request fields are `double_sparsity.sparsity_rate`, `selected_tokens`, and `dense_fallback` from [metrics.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/metrics.py:23), surfaced into `meta_info` by [tokenizer_manager.py](/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py:1799). Server metrics also exist: `sglang_double_sparsity_dense_fallback_total`, `selected_tokens_sum`, `selected_tokens_count`, and `sparsity_rate` in [metrics.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/metrics.py:11). Missing field: per-request or aggregate `total_tokens`; add it directly, or derive it with a recorded formula from `selected_tokens` and `sparsity_rate`.

**3. RECALL_COMPARABILITY**

The frozen baseline is [recall_baseline.json](/sgl-workspace/sglang/development/loop9/runs/20260610_m0/recall_baseline.json:4): lengths `1024`, `4096`, `16384`, each with `6240` samples and `20` trials; total `18720` samples.

Procedure:

1. Boot the served-fp8 DS op-point with recall oracle enabled. Recall mode forces eager only for the diagnostic in [serve_double_sparsity.sh](/sgl-workspace/sglang/development/serve_double_sparsity.sh:68).
2. Run `development/loop7/niah_oracle_sweep.py --lengths 1024 4096 16384 --num 20 --decode-steps 4`; this is the same baseline population used in [run_m0_baselines.sh](/sgl-workspace/sglang/development/loop9/run_m0_baselines.sh:65).
3. Summarize/gate with `oracle_recall_summary.py --baseline development/loop9/runs/20260610_m0/recall_baseline.json --max-delta-pp 0.5`.
4. Require matched length set and per-length sample-count equality. `oracle_recall_summary.py` fails on length-set mismatch and sample-count mismatch in [oracle_recall_summary.py](/sgl-workspace/sglang/development/loop9/oracle_recall_summary.py:122); loop-11’s absorbed summary mirrors this in [absorbed_recall_summary.py](/sgl-workspace/sglang/development/loop11/runs/20260614_m2/absorbed_recall_summary.py:124).

If served-fp8 is not comparable to the frozen baseline, define and record a fresh served-fp8 baseline instead of comparing across op-points. This rule is already stated in [loop11b/plan.md](/sgl-workspace/sglang/development/loop11b/plan.md:97).

**4. SAME_MEMORY_DESIGN**

Run two separate comparator invocations and artifact trees:

- `production_envelope`: DS default `MEM_FRACTION_STATIC=0.8` from [serve_double_sparsity.sh](/sgl-workspace/sglang/development/serve_double_sparsity.sh:48), DSA default `0.85` from [serve_native_nsa.sh](/sgl-workspace/sglang/development/serve_native_nsa.sh:56).
- `same_memory_080`: DS explicit/default `0.8`, DSA launched with `MEM_FRACTION_STATIC=0.8`.

Do not mix paths from both op-points in one `--ac11` invocation. The comparator intentionally ignores cross-side `mem_fraction_static` in [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:527), but enforces it constant within a side in [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:900). That lets each op-point pass as its own matched comparison while preventing accidental within-side mem drift.

Both sides must be radix-ON. The comparator refuses radix-state mismatch through `disable_radix_cache` in [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:286); DSA defaults radix on unless `DISABLE_RADIX_CACHE=1` in [serve_native_nsa.sh](/sgl-workspace/sglang/development/serve_native_nsa.sh:62), and DS defaults to the radix fixture artifact path in [serve_double_sparsity.sh](/sgl-workspace/sglang/development/serve_double_sparsity.sh:123).

**5. TRIAL_FLOOR_CHANGE**

Minimal behavioral edit: change `AC11_MIN_TRIALS = 3` to `AC11_MIN_TRIALS = 2` in [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:310). The two actual refusal checks use only that constant in [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:1199).

Nothing else in `--ac11` implicitly requires three trials: `_median` handles even counts by averaging the two middle values in [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:319), and `_median_metrics` only requires a non-empty set plus matching operating-point fields in [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:326). Also update stale help/comments mentioning `>=3` at [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:1377), but that is documentation, not enforcement.

**6. GAPS**

- Current `benchmark.sh` and `benchmark_baseline.sh` default `TRIALS=3` and describe independent trials in [benchmark.sh](/sgl-workspace/sglang/development/benchmark.sh:55) and [benchmark_baseline.sh](/sgl-workspace/sglang/development/benchmark_baseline.sh:51). Locked DEC-4 needs two repeated stability trials; use the paired wrapper above or update the scripts later.
- Current `--ac11` still gates DS/DSA ratio and exits nonzero on ratio failure in [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:1323). DEC-6 says the publication verdict is DS absolute SLO; ratio should be competitive-position reporting unless the comparator is updated.
- Prefix-reuse distribution is not captured in bench JSONL today; without request-level `cached_tokens`, a reviewer can say “GSP shape is not proof of 55% production reuse.”
- AC-5 no-op refusal is not enforced by `--ac11`; single-trial mode only reports `unknown/triggered/clean` in [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:1104). Publication must refuse externally until comparator gates it.
- DEC-4 says report min/median/max, but `--ac11` reports medians only through `_median_metrics` in [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:326). Add per-trial min/median/max for decode TPS, P99 TTFT, achieved concurrency, and token throughput.
- Aggregate `output_throughput` and `total_throughput` are emitted by bench_serving in [bench_serving.py](/sgl-workspace/sglang/python/sglang/bench_serving.py:1848), but comparator does not carry/report them. Report them, but do not gate them.
- If DS achieved concurrency is below nominal, do not replace the nominal concurrency with a lower one. The comparator reports achieved-vs-nominal in [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:1017); label the row `admission-capped`, judge DS against the absolute 30 TPS / P99 TTFT < 22s SLO, and treat lower-concurrency reruns as diagnostics only.
- Thermal/clock and boot-order data are not in the existing sidecar schema; `_bench_meta_writer.py` records commit, mode, seed, workload, timing knobs, trial id, and server args in [_bench_meta_writer.py](/sgl-workspace/sglang/development/_bench_meta_writer.py:13), but not boot order or GPU clocks.
