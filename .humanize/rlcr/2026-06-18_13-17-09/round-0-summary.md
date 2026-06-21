# Loop 12 — Round 0 Summary

## Objective
Extract the minimal correct table-free Double Sparsity (DS) runtime from the dev branch onto a fresh
branch off latest `origin/main` in the v2 shipping clone, add one perf eval, and prove parity. All
shipping work is in `/sgl-workspace/double-sparisty-v2/sglang` (branch `double-sparsity-v2`, off
`<BASE>=105e095e0`); loop machinery stays in this dev clone.

## Outcome: COMPLETE — all 10 ACs pass, branch pushed to the fork

| AC | Result |
|----|--------|
| AC-1 branch hygiene | 42-file clean diff vs `<BASE>`; zero `.pensieve`/`.humanize`/`development`/`SLOS.md`; pushed to `Jiminator/sglang` |
| AC-2 exclusions | 0 references to any of the 6 dropped modules in the shipping diff (precise per-module sweep) |
| AC-3 import & prune closure | `import sglang` + `dsa_backend` + `double_sparsity` pkg clean; no shipped file imports a dropped module |
| AC-4 validator ships, radix gate removed | `validate_double_sparsity` ships + gates; all radix-fixture machinery stripped (0 refs) |
| AC-5 calibration | fresh GLM-5.1-FP8 mask via shipped `calibrate.py`, loader-accepted (ChannelMask [78,64,32] page64 fp8_e4m3); calibrate smoke tests pass |
| AC-6 DS active | long-context decode `meta_info["double_sparsity"]`: selected=2048, total=5608, dense_fallback=0; per-layer bind logs present |
| AC-7 abort path | 3/3 unit test — `_maybe_abort_on_ds_error` drives `set_finish_with_abort`+`update_finish_state` same-step |
| AC-8 perf parity | **DS conc-64: 29.34 p50 decode TPS / 23.29s P99 TTFT** — within the loop-11b band (≥24.2 / ≤30.1) and ahead of native DSA on the same base (26.06 / 46.50s) |
| AC-9 dependency closure | no NEW build dep from DS (triton + flash_mla_sparse_fwd resolve; Codex-confirmed) |
| AC-10 no dead code | `metrics.record_selection` + `ds_recall_oracle_enabled` + selection-capture mirror path removed |

114 unit tests pass (slim runtime + calibrate 91, lifted-budget 20, abort 3).

## What was implemented (v2 clone, 6 commits)
- **Port mechanism**: additive minimal closure. Copied the 15 keep-list `double_sparsity/` modules;
  re-applied DS hunks onto current main by triage (11 files applied clean incl. the 895-line
  `deepseek_v2.py`; 10 hand-reconciled for drift).
- **5 modified-upstream files the plan missed, now shipped**: `output_streamer` / `detokenizer_manager`
  / `multi_tokenizer_mixin` (the `per_request_summary` transport that carries `meta_info`),
  `custom_all_reduce_v2` (`override_algo` deterministic AR), `dsa/dequant_k_cache` (lifted-budget).
- **Drift landmines fixed**: gated v2's new `DSATokenToKVPool.move_kv_cache` (would crash under
  DS+radix on the None index-k buffer); DS bind reduced to the module iteration (the pool refs were
  vestigial; `self.token_to_kv_pool` doesn't exist in v2); `forward_mla` `_select_topk_indices`
  dispatch onto the `is_nextn` gate; abort finisher renamed (`_handle_finish_state_updated_req`);
  radix-fixture CLI arg + `apply_radix_fixture_artifact` call removed from `server_args`.
- **Diagnostic prune (entangled)**: removed `_maybe_record_recall_oracle` + score/selection/latent
  capture guards + the 4 diagnostic config fields + the dead selection-capture mirror plumbing
  (dsa_backend flags + cuda_graph buffers) + the radix-fixture recorders from `validator.py`.
- **CUDA-graph selector-width ladder (the perf-critical fix)**: retargeted onto the refactored
  `DecodeCudaGraphRunner` (the old `cuda_graph_runner.py` was split into `runner/`). Captures one
  decode graph per selector width, stamping `_ds_graph_variant_key=(bs,width)` around capture/replay;
  width-encoded `variant_label` keys each graph (no `ShapeKey` change). Without it the captured graph
  scored the full `req_to_token` width (~202k) every step → DS decode 18.8 TPS; with it → 29.3 TPS.
- **One perf eval**: `benchmarks/bench_double_sparsity.py` (thin wrapper over stock `bench_serving`,
  derives p50 decode TPS) + `benchmarks/DOUBLE_SPARSITY.md` provenance doc.
- **Tests**: slim runtime+calibrate extracted from the 70-class dev harness; lifted-budget kept;
  new `test_ds_abort_path.py`.

## Files changed (shipping branch, 42 in diff vs `<BASE>`)
- New: `python/sglang/srt/layers/attention/double_sparsity/` (15 modules); `benchmarks/bench_double_sparsity.py`; `benchmarks/DOUBLE_SPARSITY.md`; `test/registered/unit/managers/test_ds_abort_path.py`; extracted `test_double_sparsity_unit.py` + kept `test_lifted_budget_decode.py`.
- Modified-upstream (DS hunks): `dsa_backend.py`, `dsa/dequant_k_cache.py`, `server_args.py`, `deepseek_v2.py`, `forward_mla.py`, `forward_mha.py`, `model_runner.py`, `model_runner_kv_cache_mixin.py`, `pool_configurator.py`, `runner/decode_cuda_graph_runner.py`, `memory_pool.py`, `memory_pool_host.py`, `logits_processor.py`, `custom_all_reduce_v2.py`, `managers/{schedule_batch,scheduler,io_struct,tokenizer_manager,detokenizer_manager,multi_tokenizer_mixin}.py`, `managers/scheduler_components/{batch_result_processor,output_streamer}.py`.

## Validation
- Cheap gates (CPU): `import sglang`/dsa_backend/double_sparsity OK; AC-2 module sweeps 0; AC-4 radix machinery 0; 114 unit tests pass.
- GPU (8×H200, GLM-5.1-FP8 TP8): mask calibrated + loader-accepted; DS-active boot (radix ON, graphs ON, no fixture/override, no expandable_segments); abort same-step; conc-64 perf 29.34 TPS / 23.29s (evidence in `development/loop12/perf_evidence/`), native-DSA baseline 26.06 / 46.50s (`development/loop12/dsa_evidence/`).
- Branch pushed: `Jiminator/sglang` `double-sparsity-v2` (6 commits).

## Key decisions / deviations (see Goal Tracker Plan Evolution Log)
- **DEC-8 / AC-8 base drift**: branching off LATEST main raised real floors — `sgl-kernel` 0.4.4
  (its flash-attn `only_qv` path; env had 0.4.3 → upgraded via prebuilt wheel) and a triton-3.6.0
  MoE-config fallback that slows BOTH DS and native DSA (TTFT ~46s base-bound). A native-DSA baseline
  on the same base isolated decode (a port gap → the ladder) from TTFT (base drift). The user chose
  to reframe AC-8 as DS-vs-DSA on the same base; once the ladder was restored, DS meets the *original*
  loop-11b band outright AND beats DSA, so the reframe is satisfied a fortiori.
- task5 was initially deferred as a "no-op"; the perf gate proved it required — corrected.

## Remaining items
None blocking. The branch is pushed; a PR can be opened at
`https://github.com/Jiminator/sglang/pull/new/double-sparsity-v2`. Optional follow-ups (not loop-12
scope): bundle triton-3.6.0-tuned MoE configs to recover base TTFT; a CPU unit test for the ladder
width selection.

## BitLesson Delta
Action: add
Lesson ID(s): BL-20260619-ds-selector-width-ladder, BL-20260619-latest-main-base-drift
Notes: The selector-width graph ladder is decode-critical for DS (full-width selection collapses
decode throughput; the runner must key decode graphs by (bs, width)). Branching off latest main drags
newer kernel/triton floors that confound an absolute perf-parity reference — always capture a
same-base native baseline before calling a perf delta a port regression.
