# Round 0 Contract

## Mainline Objective
Unblock M1. Land the two parallel M1 workstreams as code, and produce the round's required hardware artifact:
1. **AC-0 producer-bug fix (task1)** — thread `forward_batch` into `dsa_backend._write_token_labels` from all four production call sites, gate the extend-snapshot publish on `forward_batch is not None and forward_mode.is_extend()`, and add a producer-side pytest regression. Pure code; fully completable + verifiable this round.
2. **AC-4 calibration FP8-sharded load change (task3)** — change `calibrate.py` to a native-FP8, device-sharded load (`device_map="auto"`, `torch_dtype="auto"`, no bf16 upcast), fix the `model.device` single-device forward-loop assumption, and add a `--dry-run-blocks N` mode. Then **run the one-block dry-run against the real cluster weights** as the round's hardware artifact (de-risks the full calibration).

## Target ACs (≤ 2)
- **AC-0** — completed at code+regression level this round. The hardware `/generate` capture probe (task2) is sequencing-gated by the first DS boot (needs the mask) and runs in a later round.
- **AC-4** — load-path change + one-block dry-run only this round. Full mask generation + `load_channel_mask` validation (task4) is the next round's mainline, gated on a clean dry-run.

## Blocking Side Issues In Scope
- None known at round start. If the one-block dry-run reveals an FP8 shard-load upcast or a dispatch/device failure, it will be logged as a blocking side issue and is in scope to triage (it directly blocks AC-4).

## Queued Side Issues Out Of Scope
- The full multi-block calibration run (long pole) — deferred to the next round, gated on the dry-run.
- Server boot, benchmarks, comparators, quality gates (task5–task15) — out of scope this round.
- The MHA spy-test signature updates are in scope only as a direct consequence of threading `forward_batch` (test maintenance for the mainline change), not as independent work.

## Round Success Criteria
1. `_write_token_labels` accepts `forward_batch`; all four call sites thread it; the extend-publish gate is explicit; existing affected spy stubs updated.
2. The new producer-side regression passes, and the existing DS unit suite (`test_double_sparsity_unit.py`) still passes.
3. `calibrate.py` loads native FP8 device-sharded (no forced bf16 upcast) and routes forward-loop inputs to the embedding device; `--dry-run-blocks` is wired.
4. A one-block dry-run hardware artifact exists under `runs/20260528_dsv32_mvp/` showing parameter dtypes + device placement (FP8 not upcast) and that the calibration hooks fired on every layer — OR, if the load fails, a recorded failure log + a logged blocking side issue.
5. Goal tracker + this contract committed; `round-0-summary.md` written.
