# Round 2 Summary — scale-aware compact consumers + AC-3.1/AC-6 evidence

## Mainline objective (round contract)
Make the int8 compact path correct through **every** signature consumer (the compact label is `signatures * scales`, so any consumer treating raw `signatures` as authoritative proves the wrong thing), and close the dev-completable AC-3.1/AC-6 evidence gaps. This finishes AC-3 to the maximum extent possible without the TP=8 served model.

## Blocking correctness fix (Codex R1 review — reproduced false-pass)
Codex reproduced a `startup_sanity_probe()` false-pass by zeroing all compact scales. Root cause: the int8 sidecar (`scales`) was wired only into the hot scoring path; the proof/sanity/fingerprint consumers still hashed/scored raw `signatures`. Fixed across all of them (commit `e85cd2564`):

- **`validator.radix_fixture_config_fingerprint`** — added `signature_dtype`. The compare loop (`for k in current: recorded.get(k) != current[k]`) makes an fp16-recorded artifact (or any older artifact missing the key) **fail closed** against an int8 boot.
- **`radix_fixture_capture`** — `record_table_snapshot` / `build_request_capture` accept optional `scales` and record per-(layer,token) **scale SHAs**; `compare_cached_prefix` returns a `"scale"` divergence when equal int8 bytes carry different scales, or on a compact/fp16 mode mismatch. fp16 records are byte-identical (scale keys appear only in compact mode → backward compatible).
- **`dsa_backend._ds_radix_publish_extend_snapshot`** — passes `table.scales`.
- **`channel_mask.startup_sanity_probe`** — the compact plant uses **equal int8 magnitude with the signal living entirely in the scale**, scores with `token_scales`, and snapshots/restores scales. The probe now genuinely exercises the dequant path: a probe that ignored scales would see a flat int8 field and fail to find the needle.

## AC-3.1 / AC-6 evidence completed (dev-box)
- **DSA-default / no-table CPU regression (AC-6/AC-3.1):** `finalize_double_sparsity_bind` is a no-op with `use_double_sparsity=False` → `_bind_double_sparsity_runtime_data` is never invoked → no `TokenLabelTable` allocated.
- **compact-vs-fp16 decode-scoring microbench (AC-3.1), on H200:** the int8 overhead in the decode-time scorer is **+0.029 ms/token worst-case (conc 16)** against the **3.83 ms/token** budget (Loop-5 33.9→30 TPS/req margin) — ~**130× under budget**; at conc 32/64 int8 is *faster* (half the signature bytes → less memory bandwidth). The "TTFT-fixed-at-the-cost-of-TPS" failure mode does **not** occur. Artifact: `runs/20260530_dsv32_loop6/decode_scoring_microbench.md` + a GPU-guarded registered budget test that locks the property.

## Files changed
4 production files (validator, radix_fixture_capture, dsa_backend, channel_mask), the DS test file (+7 tests), and the new microbench artifact. Loop state in `.humanize/rlcr/` (gitignored).

## Validation — 279 DS unit tests pass (272 + 7 new), GPU enabled, no regression
New regressions: `test_compact_sanity_probe_finds_needle_and_restores_scales`, `test_compact_scorer_requires_scales` (the false-pass killer — needle selected only WITH scales), `test_radix_capture_diverges_on_scale_only_change`, `test_radix_capture_scale_mode_mismatch_diverges`, `test_fp16_radix_artifact_cannot_authorize_int8_boot` (fail-closed), `test_dsa_default_finalize_bind_is_noop_no_table`, `test_decode_scoring_overhead_within_tps_budget` (H200). Existing radix-capture + sanity-probe tests still green (backward compatible).

## Remaining items
- **Real-mask NIAH non-regression (AC-3.1) — explicitly deferred (hardware dependency, NOT avoidance):** requires the **TP=8 served DeepSeek-V3.2 FP8 model** (~672 GB weights) via `test_double_sparsity_v32.py`. The RLCR dev box has only **2 H200s** (~282 GB) — V3.2 cannot be served here. It is logged in the tracker's *Explicitly Deferred* section as the **first gate of the AC-4 cluster round** (run with `signature_dtype=int8` before any AC-4 mem-fraction-sweep claim, since task4 gates task5). Every other AC-3.1 item is complete.
- **AC-4 (next):** TP=8 DS boot with `signature_dtype=int8`, mem-fraction sweep 0.6→0.8, full NVML/torch-residual HBM accounting, no-OOM long `/generate` — preceded by the real-mask NIAH non-regression. Then AC-5/AC-6-hardware/AC-7/AC-8/AC-9, then gated AC-10.

## Note for review
The compact path is now correct end-to-end (scorer + proof + fingerprint + sanity probe) with the false-pass killed by a regression. The one outstanding AC-3.1 item (real-mask NIAH) is a genuine serving dependency on the 8-GPU cluster, not avoidance — it's tracked as the gate of the AC-4 round.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-compact-sidecar-consumer-coverage
Notes: Added a lesson capturing the review-found, R1→R2 defect class: when a compaction sidecar (int8 `scales`) changes the semantic value of an existing tensor (`signatures` → `signatures*scales`), every consumer that treats the base tensor as authoritative — proofs, config fingerprints, self-test/sanity probes, fixtures, IPC — must be threaded with the sidecar or it proves/authorizes the wrong thing (Codex reproduced a `startup_sanity_probe` false-pass with zeroed scales). The lesson records the enumerate-every-consumer fix pattern (scorer → bit-stability proof per-element sidecar SHA + explicit divergence → fingerprint storage-mode field fail-closed → sidecar-discriminative self-test) and the backward-compatibility rule (sidecar keys only in compact mode).
