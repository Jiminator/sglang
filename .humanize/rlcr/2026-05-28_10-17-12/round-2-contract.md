# Round 2 Contract

## Mainline Objective
**First DS boot smoke on DeepSeek-V3.2 FP8 with the calibrated mask, and capture the
evidence that single boot yields.** Boot `serve_double_sparsity.sh` single-node at
TP=8 with `MODEL_PATH` pinned to `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`
(DEC-6), the validated mask (`/models/dsv32-fp8-channel-mask.safetensors`),
`--disable-radix-cache` (still gated, radix-off this round), and
`SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`. Then exercise it once and record the artifacts
the boot produces. SGLang serving uses its own FP8 kernels, so the HF-transformers
hub-kernel fragility from Round 1 does not apply here.

This single boot is the gating action for four plan tasks:
- **task5 (AC-1)** — boot smoke: `/get_server_info` shows DS enabled, TP=8,
  `kv_cache_dtype=fp8_e4m3`, `page_size=64`, the cluster model path (not the HF-id
  default), radix-off; `/generate` returns non-empty text.
- **task2 (AC-0 hardware positive test)** — `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` +
  `/generate` returns non-empty `meta_info["double_sparsity_radix_capture"]` with
  `per_token_slot_sha` populated and `per_layer_written_all_true=True`, no error key.
- **task6 (AC-1.1)** — on a prompt longer than `top_k` (2048), `meta_info["double_sparsity"]`
  shows `0 < sparsity_rate < 1` and `dense_fallback == 0`.
- **task10 (AC-6)** — record the REGULAR CUDA-graph capture/replay status from this
  boot (distinct from the disabled piecewise setting): success OR a documented exception.

## Target ACs (≤ 2 primary)
- **AC-1** (boot smoke) — primary success condition.
- **AC-1.1** (genuine non-trivial sparsity) — primary; same boot.
- AC-0 hardware capture probe (task2) and AC-6 CUDA-graph status (task10) are evidence
  captured opportunistically during this same boot, not separate mainline objectives.

## Blocking Side Issues In Scope
- None known at round start. If the boot is rejected by the DS startup validator
  (DEC-2 guard: mask hash / page-size pairing / radix-off), or the SGLang FP8 serve
  path has its own kernel/dependency gap, that directly blocks AC-1 and is in scope to
  triage (read the validator error verbatim per the AC-1 negative test).

## Queued Side Issues Out Of Scope (do not let these take over)
- Stale `calibrate.py` docstring recipe (`--tp 1`, "--dtype is the model loading dtype")
  — Codex-flagged queued cleanup; does not block the boot. Fix before the next
  calibration handoff, not this round.
- Smoke benchmarks (task7), smoke comparator (task8), paired quality smoke (task9) —
  M2 Phase B/C, the next round, gated on a healthy boot.
- Radix flip (task11), chunked-prefill probe (task12), AC-11 (task13), AC-12 (task14),
  evidence bundle (task15) — M3, later.

## Round Success Criteria
1. DS server boots single-node TP=8 with the cluster `MODEL_PATH` + the mask; the DS
   validator accepts (radix-off); boot log shows all 8 GPUs and no crash.
2. `/get_server_info` captured to `runs/20260528_dsv32_mvp/` and shows DS enabled,
   TP=8, `kv_cache_dtype=fp8_e4m3`, `page_size=64`, radix-off, model path = cluster path.
3. `/generate` returns non-empty text (AC-1) AND non-empty
   `meta_info["double_sparsity_radix_capture"]` with `per_token_slot_sha` populated +
   `per_layer_written_all_true=True`, no error key (AC-0 hardware probe / task2).
4. A `> top_k` prompt yields `meta_info["double_sparsity"]` with `0 < sparsity_rate < 1`
   and `dense_fallback == 0` (AC-1.1), recorded to the run dir.
5. The REGULAR CUDA-graph capture/replay status from this boot is recorded (AC-6),
   distinct from the disabled-piecewise setting — success or a documented exception.
6. Server shut down cleanly; goal tracker updated; `round-2-summary.md` written with a
   BitLesson Delta. If the boot is blocked, the round records the verbatim validator/boot
   error + a logged blocking side issue (no benchmarks attempted until the boot is healthy).
