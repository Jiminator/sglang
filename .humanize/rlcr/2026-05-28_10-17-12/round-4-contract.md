# Round 4 Contract

## Mainline Objective
**Produce the TIER-1 smoke benchmark pair + comparator (M2 Phase B + the comparator
half of Phase C).** Concretely: fix the two launcher-parity blockers Codex flagged
(#D, #E), reboot DSA then DS from the scripts (radix-off on BOTH sides, cluster model
path), run the smoke DS + DSA `bench_serving` sweeps at conc 16/32/64 with `TRIALS=1`
and a shortened `MEASUREMENT_WINDOW_S` (labeled non-AC-11), and generate the smoke
comparator `mvp_compare.md` with radix parity enforced. Each side produces JSONLs +
valid `.meta.json` sidecars and a fresh `/get_server_info` proving the cluster path.

This is the next concrete command on the plan's critical path (steps 4-6 of Codex's
directive) and yields hardware artifacts under `runs/20260528_dsv32_mvp/`.

## Target ACs (≤ 2)
- **AC-8** — DS smoke benchmark via `benchmark.sh` (radix-off, `TRIALS=1`, shortened
  window, labeled non-AC-11); JSONLs + valid sidecars; the hard duration guard is
  respected.
- **AC-9** — DSA baseline smoke benchmark via `benchmark_baseline.sh` at the matching
  operating point, radix-off, same smoke shape; JSONLs + valid sidecars. The smoke
  comparator (`mvp_compare.md`) consuming both is the AC-8/AC-9 deliverable's report.

## Blocking issues in scope (must be fixed for the mainline to succeed honestly)
- **#D — serve launchers default to the HF model id.** `development/serve_double_sparsity.sh:29`
  and `development/serve_native_nsa.sh:28` default `MODEL_PATH` to `deepseek-ai/DeepSeek-V3.2`.
  Per DEC-6 both must default to `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`
  while keeping env-override support. Otherwise scripted benchmark launches silently
  drift to a download / wrong revision.
- **#E — DSA baseline launcher cannot run the radix-off smoke.** AC-8/AC-9 require
  radix-off on BOTH sides, and `benchmark_compare.py::_match_or_refuse` refuses a
  `disable_radix_cache` mismatch. DS launches with `--disable-radix-cache`;
  `serve_native_nsa.sh` does not and has no knob to. Add a launcher knob that passes
  `--disable-radix-cache` for the TIER-1 smoke, while preserving the radix-ON path for
  the later AC-11 sweep (DSA stays radix-on there).

## Queued / explicitly out of scope this round
- **task9 / AC-Q paired quality smoke** — the natural next round (M2 Phase C quality
  half). It needs both servers booted sequentially (DSA refs first, then DS). Do it
  in Round 5 unless both booted servers are healthy and the benchmark pair finished
  with ample time left in this round; if opportunistically captured, it is a bonus and
  not part of this round's pass/fail.
- **TIER-2:** task11 AC-10 radix flip, task12 AC-1b chunked-prefill probe, task13 AC-11
  3-trial sweep, task14 AC-12 full quality, task15 evidence bundle. Gated on TIER-1.
- Stale `calibrate.py` operator recipe docstring (queued cleanup; provenance recorded).

## Round success criteria
1. `serve_double_sparsity.sh` and `serve_native_nsa.sh` default `MODEL_PATH` to
   `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`, env override still honored (#D).
2. `serve_native_nsa.sh` has a radix-off knob (e.g. `DISABLE_RADIX_CACHE=1` appends
   `--disable-radix-cache`); the radix-on default for AC-11 is preserved (#E).
3. DSA booted radix-off, cluster path; DS booted radix-off (existing `--disable-radix-cache`),
   cluster path. Fresh `/get_server_info` saved for each proving model_path=cluster,
   tp=8, fp8_e4m3, page_size=64, and matching `disable_radix_cache=true`.
4. Smoke DS + DSA `bench_serving` sweeps at conc 16/32/64, `TRIALS=1`, shortened
   `MEASUREMENT_WINDOW_S`; each JSONL's observed duration ≥ the configured window (the
   hard guard does not fire); `.meta.json` sidecars present; artifacts labeled non-AC-11.
5. `benchmark_compare.py --baseline <dsa> --ds <ds>` exits 0 and writes `mvp_compare.md`
   with radix parity satisfied (no `disable_radix_cache` mismatch refusal). At least the
   conc-64 pair is compared; more if time permits.
6. Artifacts saved under `runs/20260528_dsv32_mvp/` (JSONLs + sidecars + server_info +
   `mvp_compare.md`). Goal tracker updated (task7→done, task8→done, #D/#E→resolved);
   `round-4-summary.md` written with a BitLesson Delta. No immutable-section changes.

## Known risks / notes
- The single-trial comparator's `_match_or_refuse` requires `{gpu_id, tp_size,
  page_size, disable_radix_cache, concurrency}` present + equal in BOTH JSONLs (read
  from the nested `server_info`). If `bench_serving` does not emit those fields in the
  JSONL, use `--allow-gpu-mismatch` only for `gpu_id` (same node ⇒ same GPUs anyway);
  the other four must genuinely match. Verify the emitted JSONL shape early.
- Smoke window must be long enough that `bench_serving`'s observed `duration` ≥
  `MEASUREMENT_WINDOW_S` (else `benchmark.sh` FATALs). Tune `NUM_PROMPTS` / window so a
  short smoke still satisfies its own guard.
- Operational: do not kill pre-existing processes I did not create; use a free port;
  verify `nvidia-smi` is clear before each boot; run `pkill`/commit as standalone
  commands (their exit codes abort compound commands).
