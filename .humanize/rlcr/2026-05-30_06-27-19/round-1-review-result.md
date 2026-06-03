# Round 1 Review Result

Mainline Progress Verdict: ADVANCED

ACs: 4/10 addressed | Forgotten items: 0 | Unjustified deferrals: 1

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline, taste-review guidance, Round 0 summary/review, Round 1 prompt/contract/summary, `goal-tracker.md`, the full `84d3410b9` diff, and the changed DS table/write/selection/config/backend/test code.

## Mainline Gaps

1. **AC-3.1 / task4 is still incomplete: the required real-mask NIAH non-regression and compact-vs-fp16 decode-scoring microbench were deferred.**

   Evidence:
   - The plan makes both checks part of AC-3.1, not optional AC-4 extras: real-mask NIAH non-regression and an early compact-vs-fp16 decode-scoring microbench are named in `development/loop6/refined_plan_v1.md:47`, repeated in the implementation hints at `:132`, and in task4 at `:183`.
   - Claude's own summary explicitly defers both to the next hardware round in `.humanize/rlcr/2026-05-30_06-27-19/round-1-summary.md:32`.
   - The new tests only cover synthetic selection overlap, byte accounting, quantization round-trip, and focused Triton/graph-safe scoring paths (`test_double_sparsity_unit.py:9156-9383`). There is no `runs/20260530_dsv32_loop6/` artifact for the Loop-5-mask NIAH comparison or decode-scoring microbench.

   Impact: int8 selection can pass the synthetic top-k gate while still regressing the already-weak real DS recall or exceeding the decode TPS margin. Per the review prompt, this deferral is incomplete work and must not be accepted as AC-3 done.

2. **The CPU-checkable AC-6 / task4 DSA-default-no-table test was not implemented.**

   Evidence:
   - AC-3.1 requires a config/unit test that a DSA-default boot allocates no DS table (`development/loop6/refined_plan_v1.md:47`), and AC-6 separately requires no-DS-flags default behavior to allocate no `TokenLabelTable` (`:79`).
   - Round 1 only added tests for `signature_dtype` default/opt-in, no compact scales in fp16 DS mode, and int8 scoring. Static inspection confirms the new class has no `enable_double_sparsity=False` / no-table boot-path test.
   - The summary line claiming "DSA-default (fp16) allocates no scales" conflates DS fp16 storage with DSA default no-table behavior (`round-1-summary.md:16`).

   Impact: the opt-in/default safety property is still only inferred from code structure, not locked by the requested regression test.

## Blocking Side Issues

1. **Compact int8 scales are missing from existing proof/sanity consumers, so those paths can prove the wrong thing.**

   Evidence:
   - `radix_fixture_config_fingerprint()` fingerprints model path, TP, page size, KV dtype, and channel-mask SHA, but not `signature_dtype` (`python/sglang/srt/layers/attention/double_sparsity/validator.py:328-336`). A radix fixture recorded under fp16 can therefore authorize an int8 boot with the same mask/config surface.
   - `_ds_radix_publish_extend_snapshot()` passes only `table.signatures` and `table.written` into the label-capture proof (`python/sglang/srt/layers/attention/dsa_backend.py:396-404`), and `build_request_capture()` hashes only raw signatures/written (`radix_fixture_capture.py:197-223`, `:279-296`). In compact mode, the semantic label is `signatures * scales`; equal int8 bytes with different scales would false-pass the label-stability proof.
   - `startup_sanity_probe()` snapshots/plants/restores only `table.signatures` and `written`, then calls `retrieve_topk_via_labels()` without `token_scales` (`channel_mask.py:520-557`). I reproduced a false pass by zeroing all compact scales: the probe still passed on raw int8 `10`, while the real compact scorer with `token_scales=table.scales` did not select the needle.

   Impact: the compact path is not actually threaded through every signature consumer. This blocks safe AC-5/AC-7 radix-on proof and leaves an int8 startup sanity path that can pass even when the scale sidecar makes the real selector unable to distinguish the planted needle.

## Queued Side Issues

No separate queued side issues. The scale-sidecar proof gap is not cosmetic; it blocks the upcoming radix-on hardware validation.

## Goal Alignment Check

AC-1 remains verified. AC-2 is now verified: `footprint_feasibility.md` was corrected so int8 same-`label_dim` is the unambiguous binding lever, the fp16 lower-`f` window is optional instrumentation only, and Anchor B is corrected to `≈0.70`.

AC-3 advanced substantially but is not complete: the core int8 table/write/scoring path exists, but the required real-mask NIAH, decode-scoring microbench, and scale-aware proof/sanity consumers are still missing. AC-6 is only partially addressed: fp16 remains default and int8 is opt-in, but the requested DSA-default/no-table unit test and hardware product proof are still pending. AC-4 through AC-10 remain tracked and pending; AC-10 is still properly gated.

Deferred items are not justified as completion blockers: task4 gates task5 in the original plan (`development/loop6/refined_plan_v1.md:183-184`), so the missing AC-3.1 evidence must be completed before AC-4 is claimed. No original-plan item is forgotten from the tracker.

## Directive Implementation Plan

1. Fix compact scale authority across proof/sanity paths before running hardware claims:
   - Add `signature_dtype` to `radix_fixture_config_fingerprint()` using the parsed `DoubleSparsityConfig`; let older artifacts without the field fail closed.
   - Extend `radix_fixture_capture.record_table_snapshot()` and `build_request_capture()` to accept optional `scales`. When present, record per-layer and per-token scale SHAs alongside signature SHAs. Update `compare_cached_prefix()` to compare scale hashes as part of `divergence_kind="label"` or a new explicit `scale` divergence.
   - Pass `table.scales` from `_ds_radix_publish_extend_snapshot()`.
   - Update `startup_sanity_probe()` to snapshot/restore scales when `table.scales is not None`, plant representable int8+scale values, and call `retrieve_topk_via_labels(..., token_scales=table.scales)`.
   - Add CPU regression tests proving a compact sanity probe uses scales and that fp16 radix artifacts cannot authorize int8 boots.

2. Complete the missing AC-3/AC-6 tests and artifacts:
   - Add a focused CPU test for the no-DS/default boot path: with `enable_double_sparsity=False`, no `_double_sparsity_token_label_table` is allocated and DS decode remains inactive.
   - Run the Loop-5-mask NIAH non-regression with `signature_dtype=int8` against the fp16 Loop-5 DS baseline; write the artifact under `runs/20260530_dsv32_loop6/`.
   - Run the compact-vs-fp16 decode-scoring microbench on the served/hardware path and fail it if the compact overhead would push the 33.9 TPS/req Loop-5 margin below 30 TPS/req. Record both absolute timings and the pass/fail threshold.

3. Only after those are green, run AC-4 with the compact table: TP=8 DS boot, `mem_fraction_static` sweep toward 0.8, full NVML/torch residual HBM accounting, `/get_server_info`, and sustained long `/generate` with no OOM or monotonic memory growth.

## Tracker Update

Updated the mutable section of `goal-tracker.md`:
- Marked AC-2/task2 verified.
- Left task3/task4 active as partial pending fixes.
- Added blocking issues for missing AC-3.1/AC-6 evidence and compact scale-sidecar omissions in proof/sanity paths.

## Validation Performed

- Ran targeted CPU tests: `python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q -k 'CompactInt8Signatures and not cuda and not graph_safe'` → 11 passed.
- Static-inspected the new test class for absence of a DSA-default/no-table test.
- Reproduced the compact sanity-probe false pass with zeroed scales: `startup_sanity_probe()` passed without `token_scales`, while the real compact scorer with `token_scales=table.scales` did not select the planted needle.

NOT COMPLETE
