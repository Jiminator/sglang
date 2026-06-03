# Round 2 Contract

## Mainline Objective (exactly one)
**Make the int8 compact path correct through every signature consumer and close the dev-completable AC-3.1/AC-6 evidence gaps.** The semantic label in compact mode is `signatures * scales`, so every place that today treats raw `signatures` as the full label authority must learn about `scales`, and the two non-serving AC-3.1/AC-6 evidence items (DSA-default-no-table test, decode-scoring microbench) must be produced. This finishes AC-3 to the maximum extent possible without the TP=8 served model.

## Target ACs (1–2)
- **AC-3** — completion of the compact-path correctness + non-serving evidence (`coding`).
- **AC-6** — the CPU DSA-default/no-`TokenLabelTable` regression (`coding`).

## Blocking Side Issues in Scope (Codex R1 review — real correctness gap)
The int8 scale sidecar is missing from every existing proof/sanity consumer; Codex *reproduced* a `startup_sanity_probe()` false-pass by zeroing scales. Fix all of:
1. `validator.radix_fixture_config_fingerprint()` — add `signature_dtype`; fp16-recorded artifacts must **fail closed** against an int8 boot (and vice versa).
2. `radix_fixture_capture.record_table_snapshot()` / `build_request_capture()` — accept optional `scales`; when present, record per-(layer/token) scale SHAs alongside signature SHAs. `compare_cached_prefix()` compares scale hashes (label/`scale` divergence).
3. `dsa_backend._ds_radix_publish_extend_snapshot()` — pass `table.scales`.
4. `channel_mask.startup_sanity_probe()` — snapshot/restore `scales` when compact, plant representable int8+scale values, score with `token_scales=table.scales`.
   Plus CPU regressions: a compact sanity probe actually uses scales (zeroing scales must change the outcome), and an fp16 radix artifact cannot authorize an int8 boot.

This blocks the AC-5/AC-7 radix-on proof, so it is genuinely blocking.

## Queued / Explicitly Deferred (justified — hardware dependency, NOT avoidance)
- **Real-mask NIAH non-regression (AC-3.1)** — requires the **TP=8 served DeepSeek-V3.2 FP8 model** (~84 GB/rank × 8 ≈ 672 GB of weights) via `test_double_sparsity_v32.py`. The RLCR dev box has only **2 H200s** (≈282 GB) — V3.2 cannot be served here. This runs on the **cluster** as the **first gate of the AC-4 hardware round** (task4 gates task5), before any AC-4 claim. Deferral reason = hardware unavailability on the dev box, with a concrete trigger (the AC-4 cluster serve).
- AC-4/AC-5/AC-7/AC-8/AC-9 and gated AC-10 — later rounds. Do **not** touch the FlashMLA `indices.shape[-1]==dsa_index_topk` assert (AC-3.3).

## Round Success Criteria
1. `signature_dtype` is part of the radix fixture fingerprint; an fp16 artifact cannot authorize an int8 boot (fail-closed), proven by a CPU regression.
2. Radix-capture records + compares scale SHAs in compact mode; equal int8 bytes with different scales now diverge (CPU regression).
3. `startup_sanity_probe` is scale-aware: a compact probe plants/restores scales and scores with `token_scales`; a CPU regression shows zeroing scales changes the probe result (kills the false-pass Codex found).
4. A CPU test proves a no-DS/default boot (`enable_double_sparsity=False`) allocates **no** `_double_sparsity_token_label_table` and does not enable DS decode.
5. A compact-vs-fp16 **decode-scoring microbench** runs on H200 at production decode shapes, records absolute timings + the explicit pass/fail threshold derived from the Loop-5 33.9→30 TPS/req margin (≈3.83 ms/token budget across 61 layers), with an artifact under `runs/20260530_dsv32_loop6/`.
6. Full DS unit suite stays green (incl. new regressions); commit + push to `jimmy`; `round-2-summary.md` with BitLesson Delta.

## Out-of-Scope Guards
- No FlashMLA decode-assert changes (AC-3.3). fp16 stays the default.
- This is a code + dev-GPU-microbench round; the cluster TP=8 serve (NIAH + AC-4) is the next round and gates the AC-4 claim.
