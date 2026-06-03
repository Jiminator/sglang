# Round 2 Review Result

Mainline Progress Verdict: ADVANCED

ACs: 4/10 addressed | Forgotten items: 0 | Unjustified deferrals: 1

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline, taste-review guidance, Round 0/1 summaries and reviews, `round-2-prompt.md`, `round-2-contract.md`, `round-2-summary.md`, `goal-tracker.md`, the R2 commit diff, changed production/test files, the decode microbench artifact, the launcher, and the NIAH harness.

## Mainline Gaps

1. **AC-3.1 / task4 is still incomplete: the required real-mask NIAH non-regression has no artifact and was moved to `Explicitly Deferred`.**

   Evidence:
   - The plan makes real-mask NIAH part of AC-3.1, not an optional AC-4 extra: `development/loop6/refined_plan_v1.md:47`, `:132`, `:183`, and DEC-8 at `:263`.
   - Task ordering is explicit: task5/AC-4 depends on task4 (`development/loop6/refined_plan_v1.md:183-184`).
   - The only Loop 6 R2 acceptance artifact under `runs/20260530_dsv32_loop6/` is `decode_scoring_microbench.md`; there is no int8-vs-fp16 real-mask NIAH artifact.
   - Claude's summary and tracker explicitly defer the item to the next cluster round.

   Impact: the compact int8 path still has no real-mask recall non-regression against the fp16 Loop-5 DS baseline. Synthetic top-k overlap and the decode microbench are useful, but they do not prove the core AC-3.1 quality gate. Per the review prompt, this deferral is incomplete work and cannot be accepted as AC-3 completion.

## Blocking Side Issues

1. **The standard DS launcher still cannot select the compact int8 table, so the next AC-4/NIAH run can silently validate fp16 labels.**

   Evidence:
   - `development/serve_double_sparsity.sh:53-54` builds `DS_CONFIG` with `top_k`, `page_size`, `channel_mask_path`, and `device_buffer_size` only. It never includes `signature_dtype`.
   - The launch log at `development/serve_double_sparsity.sh:75-85` also does not print the signature dtype.
   - The config default is fp16 (`python/sglang/srt/layers/attention/double_sparsity/config.py:43`), so running the documented script as-is validates the full-precision table, not the compact table.
   - The plan's AC-4 workflow tells Claude to sweep via `serve_double_sparsity.sh` (`development/loop6/refined_plan_v1.md:133`) and AC-4 is defined as validation **with the compact table** (`:58`).

   Impact: even if Claude gets the TP=8 cluster, the next hardware run is easy to mis-run: `MEM_FRACTION_STATIC=... bash development/serve_double_sparsity.sh` still boots fp16 DS. That would invalidate both the real-mask NIAH non-regression and the AC-4 mem-fraction/no-OOM evidence.

   Required fix:
   - Add `SIGNATURE_DTYPE="${SIGNATURE_DTYPE:-fp16}"` to `development/serve_double_sparsity.sh`.
   - Include `"signature_dtype": "${SIGNATURE_DTYPE}"` in `DS_CONFIG`.
   - Echo `signature_dtype` in the launcher log.
   - Run the next DS cluster server as `SIGNATURE_DTYPE=int8 ... bash development/serve_double_sparsity.sh`.

## Queued Side Issues

No separate queued side issues. The remaining findings block the AC-3 -> AC-4 handoff rather than being optional cleanup.

## Goal Alignment Check

AC-1 and AC-2 remain verified. AC-3 advanced: the scale-sidecar proof/sanity bug from Round 1 is fixed, and the decode-scoring microbench passed on H200. AC-3 is still not complete because the real-mask NIAH non-regression is missing. AC-6 advanced: the CPU DSA-default/no-table regression is present, but the hardware product proof remains task7 and is still pending. AC-4 through AC-10 remain pending; AC-10 is still properly gated.

Forgotten items: none. The tracker contains all original-plan tasks. The only rejected deferral is the real-mask NIAH item: the hardware dependency is real, but it still blocks AC-3/task4 and must not live in `Explicitly Deferred` as accepted completion.

## Directive Implementation Plan

1. Fix the launcher before any cluster run:
   - Patch `development/serve_double_sparsity.sh` with `SIGNATURE_DTYPE`, include it in `DS_CONFIG`, and log it.
   - Add a small unit/static test or shell check that `SIGNATURE_DTYPE=int8` produces a config containing `"signature_dtype": "int8"` and that the default remains `"fp16"`.

2. Run the real-mask NIAH non-regression on the TP=8 cluster before AC-4:
   - Boot DSA reference on port 30001 using the existing DSA launcher.
   - Boot DS on port 30000 with the Loop-5 mask and compact labels: `SIGNATURE_DTYPE=int8 MEM_FRACTION_STATIC=0.6 TP_SIZE=8 CHANNEL_MASK_PATH=/models/dsv32-fp8-channel-mask.safetensors bash development/serve_double_sparsity.sh`.
   - Record `/get_server_info` and the DS server log excerpt proving `signature_dtype=int8`.
   - Run `DS_BASE_URL=http://localhost:30000 DSA_BASE_URL=http://localhost:30001 AC12_NIAH_NUM_PROMPTS=20 PYTHONPATH=python python -m pytest test/manual/test_double_sparsity_v32.py -v -k niah`.
   - Copy the generated `development/results/ac12_niah_*.json` files into `runs/20260530_dsv32_loop6/real_mask_niah_int8/`.
   - Write `runs/20260530_dsv32_loop6/real_mask_niah_nonregression.md` comparing int8 DS recall against the fp16 Loop-5 DS baseline artifacts in `runs/20260528_dsv32_mvp/ac12_results/`. Pass only if each comparable length has `int8_ds_recall_pct >= fp16_loop5_ds_recall_pct` and no new DS unservable error where the fp16 baseline served.

3. Only after that artifact passes, proceed to AC-4:
   - Sweep `SIGNATURE_DTYPE=int8 MEM_FRACTION_STATIC=0.6 -> ... -> 0.8`.
   - Log full NVML/torch residual HBM accounting, `/get_server_info`, and a sustained long `/generate` with no OOM or monotonic growth.

## Tracker Update

Updated the mutable section of `goal-tracker.md`:
- Rejected the Round 2 `Explicitly Deferred` classification for real-mask NIAH.
- Left task4 active/partial with real-mask NIAH pending.
- Added a blocking issue for the missing `SIGNATURE_DTYPE` launcher surface.
- Updated task5 notes so AC-4 cannot start without the compact launcher path and NIAH gate.

## Validation Performed

- `python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q -k 'CompactScaleSidecarConsumers or (CompactInt8Signatures and not cuda and not graph_safe and not decode_scoring_overhead)'` -> 17 passed, 262 deselected.
- `python -m pytest test/registered/unit/manual/test_m3b_label_capture_verdict.py -q` -> 13 passed.
- `python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q -k 'decode_scoring_overhead_within_tps_budget'` -> 1 passed, 278 deselected.
- `git diff --check 84d3410b9..e85cd2564` -> clean.

NOT COMPLETE
