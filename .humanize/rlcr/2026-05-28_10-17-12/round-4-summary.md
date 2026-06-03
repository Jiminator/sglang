# Round 4 Summary

## Work Completed
Produced the TIER-1 smoke benchmark pair + comparator (AC-8/AC-9) on 8x H200, fixing the
two Codex-flagged launcher blockers plus two more gaps that surfaced under real bench load.

- **#D — launchers default to the cluster weights.** Both `serve_double_sparsity.sh` and
  `serve_native_nsa.sh` now default `MODEL_PATH` to
  `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2` (env override preserved). Verified on
  both live servers.
- **#E — DSA radix-off smoke knob.** Added `DISABLE_RADIX_CACHE=1` to `serve_native_nsa.sh`
  (appends `--disable-radix-cache`); default 0 keeps radix-on for the later AC-11 sweep.
- **Two more gaps found on hardware (committed as smoke-enablement):**
  - The DSA baseline OOMed at the stock `mem_fraction_static=0.897` in
    `flash_mla_with_kvcache` (~84 GB/rank weights leave no kernel-workspace headroom at the
    4096-ISL shape). Added a `MEM_FRACTION_STATIC` knob (default 0.85).
  - The "shortened window" did not shorten the run: bench_serving runs FULL epochs of
    `NUM_PROMPTS` before re-checking the window, and one epoch at conc 16 / ISL 4096 is
    ~900 s. Made `NUM_PROMPTS` env-overridable (default unchanged 320) so a smoke can use a
    small per-epoch count.
- **Ran the pair (DSA refs first, then DS, single node sequential):** DSA (mem 0.85) and DS
  (mem 0.6, `--disable-radix-cache`) booted radix-off from the cluster weights; smoke shape
  `conc 16/32/64`, `TRIALS=1`, `NUM_PROMPTS=64`, `WARMUP_SECONDS=0`, `MEASUREMENT_WINDOW_S=30`,
  GSP ISL≈4096 / OSL 512.
- **Comparator:** `benchmark_compare.py --baseline --ds` exited 0 for all three
  concurrencies (radix parity held — no `disable_radix_cache` mismatch refusal). Assembled
  `mvp_compare.md` with the smoke context, the three tables, the radix-parity proof, and the
  directional reading + KV caveat.

## Files Changed
- `development/serve_double_sparsity.sh` — `MODEL_PATH` default → cluster weights.
- `development/serve_native_nsa.sh` — `MODEL_PATH` default → cluster weights;
  `DISABLE_RADIX_CACHE` knob; `MEM_FRACTION_STATIC` knob (default 0.85).
- `development/benchmark.sh`, `development/benchmark_baseline.sh` — `NUM_PROMPTS`
  env-overridable (default unchanged).
- `runs/20260528_dsv32_mvp/` — `mvp_compare.md`, `mvp_compare_c{16,32,64}.{md,json}`,
  `dsa_smoke_server_info.json`, `ds_smoke_server_info.json`,
  `smoke_results/*.meta.json` (6 sidecars). Raw `*.jsonl` are gitignored (repo policy
  `.gitignore:179`); metrics are preserved in the sidecars + comparator JSON.
- Commits: `6acdfb94f` (#D/#E launcher parity), `f2bc1eb6a` (mem-fraction + NUM_PROMPTS
  smoke enablement), `2220a793f` (smoke artifacts + comparator). Pushed to remote.

## Validation
- All 6 bench JSONLs have observed `duration` 168–533 s ≥ the 30 s window → the hard
  duration guard passed; smoke explicitly labeled non-AC-11.
- 6 `.meta.json` sidecars confirm `disable_radix_cache=true` on BOTH sides, `tp_size=8`,
  `mem_fraction_static` 0.85 (DSA) / 0.6 (DS), DS on/off correctly, cluster `model_path`.
- `benchmark_compare.py` exit 0 × 3 concurrencies.
- Directional read (smoke only): DS per-token decode competitive-to-better (TPOT P50
  25.7–27.1 ms flat vs DSA up to 193 ms P99 at conc 32/64; DS per-request tok/s on par or
  higher at conc 32/64). DS TTFT worse (P99 120/244/502 s vs DSA 34/70/155 s) — a
  single-node KV-pool/effective-concurrency artifact at mem 0.6 (see #F), not an
  algorithmic regression.

## Remaining Items
- **#F (queued, blocks honest AC-11 TTFT, not the smoke):** DS at mem 0.6 admits only ~2
  concurrent 4096-tok requests (TokenLabelTable + ~84 GB/rank weights → small KV pool), so
  TTFT is queuing-dominated. Resolve before the radix-on AC-11 sweep (scale the workload,
  raise the DS KV budget post-radix-flip if headroom allows — capped, or report effective
  vs nominal concurrency).
- **task9 / AC-Q** paired quality smoke — next (M2 Phase C quality half); needs both
  servers sequentially.
- **TIER-2:** task11 AC-10 radix flip, task12 AC-1b, task13 AC-11, task14 AC-12, task15
  bundle.
- Queued cleanup: stale `calibrate.py` operator recipe docstring.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-dsv32-bench-smoke-sizing
- Notes: Added `BL-20260529-dsv32-bench-smoke-sizing` — (1) the DSA baseline launcher needs
  a mem-fraction knob (stock 0.897 OOMs `flash_mla_with_kvcache` on V3.2 FP8 TP=8 under
  bench load; use 0.85); (2) bench_serving's time-window runs FULL epochs of `NUM_PROMPTS`
  before re-checking elapsed time, so a small window does not shorten the run — make
  `NUM_PROMPTS` small (≥ max concurrency) and `WARMUP_SECONDS=0` for a quick smoke while the
  duration guard still passes. Keep large NUM_PROMPTS + 120s/600s floors for AC-11. The DS
  KV-pool/concurrency constraint is tracked as project state #F in the goal tracker rather
  than a BitLesson (it is config/deployment-specific, not a reusable code pattern).
