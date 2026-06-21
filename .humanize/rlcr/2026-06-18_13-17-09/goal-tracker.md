# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

Extract the **minimal correct table-free Double Sparsity (DS) runtime** from the development branch
`dev/double-sparsity-standalone` (in clone `/sgl-workspace/sglang`) onto a **fresh branch off latest
`origin/main`** in the shipping clone `/sgl-workspace/double-sparisty-v2/sglang` (`origin =
Jiminator/sglang`), add **one simple performance eval**, and prove the extraction preserved behavior
and performance. Curation + clean-port, not new development: DS behavior is frozen.

The shipping branch must let a client clone it, calibrate a GLM-5.1-FP8 channel mask, enable DS,
serve, and reproduce the loop-11b conc-64 result (≈26.9 TPS / ≈25.1 s P99 TTFT) within a parity
band — DS genuinely active (not a silent dense fallback). "Performant" means **no regression vs.
loop-11b**, NOT the 30-TPS SLO floor (which neither DS nor native DSA meets at conc 64). Every file
and symbol that lands must be reachable from the DS serve path, the calibration tool, the perf
wrapper, or a shipped test. Dev scaffolding (`.pensieve/`, `.humanize/`, `development/`, `SLOS.md`,
oracles, capture sinks, recall/validator harnesses, comparator, manual `dsv32` fixtures) does NOT
ship. Method = **additive minimal closure** (copy new files, hand re-apply DS hunks onto latest
main, keep a file only if its removal breaks `import sglang` / DS server boot / the conc-64 run),
NOT a merge or rebase of the divergent dev branch.

### Acceptance Criteria
<!-- Each criterion must be independently verifiable. Source: development/loop12/plan.md -->

All criteria run in the **v2 clone** against the new branch unless stated otherwise. `<BASE>` is the
recorded SHA of `origin/main` the branch was cut from. Each AC has positive + negative tests in the
plan; condensed here.

- **AC-1 Branch hygiene & diff scope.** Branch on `Jiminator/sglang` cut from latest `origin/main`,
  `<BASE>` recorded. `git diff --name-only <BASE>...HEAD` lists ONLY shipped `double_sparsity/`
  modules, modified-upstream files (incl. `logits_processor.py`), `calibrate.py`, the one perf
  wrapper, minimal feature tests, and the provenance doc. NEG: any `.pensieve/`/`.humanize/`/
  `development/`/`SLOS.md` or any off-allowlist path in the diff → fail.
- **AC-2 Exclusions absent (precise per-module sweeps).** For each dropped module —
  `oracle_artifact_sink`, `selection_recall_oracle`, `radix_fixture_capture`, `score_capture`,
  `selection_capture`, `latent_capture` — `rg -l` over `python/` + `test/` returns ZERO and the
  files are absent; `test/manual/test_dsv32_*`, `test/registered/unit/development/*`, the comparator
  tree, and oracle/recall/ac11/ac12/m3b/accuracy-gate tests absent. NEG: leftover import or a bare
  `capture` grep used → fail.
- **AC-3 Import & prune closure.** `python -c "import sglang"` exits 0; dsa_backend + double_sparsity
  package import exit 0; no shipped file imports a dropped module. NEG: any import error / dropped
  import → fail.
- **AC-4 `validator.py` ships and gates; radix fail-closed gate REMOVED.** `validate_double_sparsity`
  called unconditionally at DS startup; rejects missing/corrupt mask, SHA/schema mismatch,
  hierarchical-cache+DS, disagg+DS, non-graph-safe selector w/o `--disable-cuda-graph`. radix+DS
  boots with NO fixture artifact and NO `SGLANG_DS_RADIX_OVERRIDE`. NEG: corrupt mask accepted, or
  radix+DS needs a gate, or any `apply_radix_fixture_artifact`/recorders/`RADIX_FIXTURE_STATE_*`
  remain → fail.
- **AC-5 Calibration regenerates a valid mask (calibrate.py ships + exercised this run).**
  `calibrate.py` on the documented corpus produces a GLM-5.1-FP8 mask `load_channel_mask` /
  `validate_double_sparsity` accepts; a CPU-runnable calibrate smoke test passes. NEG: bad output
  rejected by loader, or calibrate imports a dropped module → fail.
- **AC-6 Server boots DS genuinely active (command-level).** GLM-5.1-FP8, TP=8, dsa, `glm4_moe`,
  FP8 KV, page 64, CUDA graphs ON, radix ON, NO expandable allocator, mask = freshly-calibrated. A
  decode response's `meta_info["double_sparsity"]` has `selected_tokens>0`,
  `total_tokens>selected_tokens`, `dense_fallback==0`; bind logs present. NEG: meta absent, or
  total==selected, or dense_fallback!=0 → fail.
- **AC-7 Abort path carries the loop-11b fix (command/test-level).** Injecting a DS per-request
  error drives `req.set_finish_with_abort(...)` then `req.update_finish_state(...)` in the SAME
  scheduler step; request finishes with an abort `finished_reason` that step. NEG: relies on
  pre-#25725 `check_finished`, or hangs/wrong state → fail.
- **AC-8 Perf parity (exact metric + numeric band).** Thin wrapper over STOCK `bench_serving`;
  workload mirrors conc-64 exactly (`generated-shared-prefix`, gsp 2253/1843, OSL 512, range-ratio
  1.0, max-concurrency 64, sglang backend, one trial, fixed num-prompts + seed). Emits p50 decode
  TPS (median of `(output_tokens-1)/decode_duration`) + P99 TTFT. **PASS: p50 decode TPS ≥ 24.2 AND
  P99 TTFT ≤ 30.1 s.** Evidence saved (cmd, server args, `<BASE>`, GPU info, bench JSON). NEG:
  below band, or wrapper can't emit the metrics → fail.
- **AC-9 Dependency closure.** No new build dep: `triton` + `flash_mla_sparse_fwd` resolve on base,
  no new `deep_gemm`; `import sglang` works in a clean v2 env. NEG: a shipped module imports a
  symbol absent on latest `origin/main` → fail.
- **AC-10 No dead code (module granularity).** Every shipped DS module reachable from serve path /
  `calibrate.py` / perf wrapper / a shipped test; `metrics.record_selection` + radix-fixture
  recorders removed. NEG: an unreferenced module, or a dead symbol remains → fail.

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 4 (Updated: Round 2 Review)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan from development/loop12/plan.md | - | - |
| 0 | **AC-8 REFRAMED (user decision, 2026-06-18)**: parity gate = DS within band of NATIVE DSA on the SAME latest-main base (port-correctness), NOT vs loop-11b's absolute 26.9/25.1 (a different base). Evidence: on latest main, native DSA decode=26.06 / P99 TTFT=46.5s; DS decode=18.81 / TTFT=39.2s. Decode gap (18.8<26.1) is the deferred selector-width ladder (port-recoverable → task5 retarget now REQUIRED). TTFT is base drift (DS 39 ≈ DSA 46 ≫ 25.1 ref; triton-3.6.0 MoE-config fallback) — DS already beats DSA on TTFT. | User chose "Reframe AC-8 as DS-vs-DSA"; perf gate revealed latest-main base drift confounds the absolute comparison | AC-8 redefined; task5 reopened (ladder port) |
| 0 | **task5 ladder RETARGETED + AC-8 PASS (outright)**: ported the selector-width ladder onto DecodeCudaGraphRunner (helpers + __init__ + per-width capture stamping + replay width-resolution; width-encoded variant_label, no ShapeKey change). DS conc-64 now **29.34 TPS / 23.29s P99 TTFT** — within the ORIGINAL loop-11b band (≥24.2 / ≤30.1) AND ahead of native DSA (26.06/46.50) on the same base. Root cause was full-width selection scoring (req_to_token width ~202756, 40× the 5120 bucket). [[ds-selector-width-graph-ladder-is-decode-critical]] | GPU perf re-run with ladder | AC-8 PASS, AC-6 PASS |
| 0 | **M9 push DONE**: branch `double-sparsity-v2` (6 commits) pushed to `Jiminator/sglang`. Final sweeps green: AC-1 42-file clean diff (no dev scaffolding), AC-2 0 dropped-module refs in diff, AC-3 import OK, AC-4 0 radix machinery, AC-10 record_selection/ds_recall_oracle_enabled gone. ALL 10 ACs pass; 114 unit tests pass. | M9 | AC-1/AC-2/AC-10 |
| 1 | **AC-8 review fix [P1] — perf wrapper used the wrong GSP shape**: it ran the stock generated-shared-prefix default (64 groups × 16 = 1024 reqs) instead of the loop-11b single-group workload. Fixed: pin `--gsp-num-groups 1 --gsp-prompts-per-group <n>`, record gsp grouping + `actual_completed` in verdict, fail closed on shape drift. **Rerun (correct shape): 256/256 reqs, p50 decode TPS 35.05 (≥24.2), P99 TTFT 22.90s (≤30.1), parity true.** Evidence replaced. | Codex R0 review [P1] | **AC-8 now validly PASS** |
| 1 | **Review fix [P3] — stripped plan/loop tracking markers** (AC-/DEC-/Milestone/Loop-N/[R-N]) from durable shipped code + test comments per plan:417 Code Style; DeepSeek-R1 model names + provenance doc left intact. Import + 94 tests green. | Codex R0 review [P3] | AC-1 (clean shipped comments) |
| 1 | **Re-pushed corrected branch** `double-sparsity-v2` (HEAD f05326636) to `Jiminator/sglang`. All final sweeps green. | M9 | AC-1/AC-8 |
| 1 review | **AC-8 accepted; close-out cleanup still open.** Review verified the corrected wrapper and evidence: one GSP group, `gsp_prompts_per_group=256`, `actual_completed=256`, p50 decode TPS 35.05, P99 TTFT 22.90s, parity true. However the P3 cleanup claim is incomplete: newly shipped DS files still contain workflow labels (`Option B`, `Tier-2.A`, `Round 3`), and the provenance doc says both DS and DSA meet the band even though its native-DSA P99 column is 46.50s (>30.1s). | Codex R1 review | AC-8 PASS; task13 complete; task14/task15 remain active for final close-out |
| 0 | Confirmed GPU prerequisites present (GLM-5.1-FP8 weights, corpus, 8 idle H200) → M5/M6/M8 feasible this run | Environment probe | none |
| 0 | **Inventory found 5 modified-upstream files the plan MISSED** — `output_streamer.py`, `detokenizer_manager.py` (per_request_summary transport, closure-required for `meta_info["double_sparsity"]` to reach the client), `multi_tokenizer_mixin.py` (per_request_summary for multi-tok path), `custom_all_reduce_v2.py` (`override_algo` deterministic AR, called by shipped `selection_kernel.py:198`), `dsa/dequant_k_cache.py` (`dequantize_k_cache_paged_out` for lifted-budget graph-safe decode). All SHIP (Codex-confirmed). | DS diff (git, dev HEAD vs base a77449f86) + Codex inventory | AC-1 (diff allowlist now includes these 5), AC-3/AC-6 (closure) |
| 0 | bench_serving.py stays STOCK (DS-meta capture is eval-only) — confirmed drop. | Decision DEC-3 + Codex | AC-1 |
| 0 | Base = latest origin/main `105e095e0`; branch `double-sparsity-v2` (exact name free; only `dev/double-sparsity-v2` was taken). | DEC-8 | AC-1 |
| 0 | KERNEL_DEPS NONE-NEW (triton DS kernels, sgl_kernel.flash_mla.flash_mla_sparse_fwd, CustomAllReduceV2, deep_gemm all exist on base) — AC-9 satisfiable. | Codex inventory | AC-9 |
| 0 | Prune surface broader than plan listed: also config.py (capture fields 30-33/139-142/365-372) + cuda_graph.py (selection_capture_layers buffers), beyond selection_kernel/model_runner/deepseek_v2. | Grep of copied keep-files | AC-2/AC-10 |
| 0 | Port-mechanism drift catches (hand-reconciled): custom_all_reduce_v2 already had a persistent self.override_algo + tc_piecewise rename; v2 added a DSATokenToKVPool.move_kv_cache override that iterates index_k -> MUST gate under DS (radix-ON page moves would crash on the gated None buffer); _ds_req/token_to_kv_pool bind refs are vestigial (set-never-read) and self.token_to_kv_pool doesn't exist in v2 -> bind = module iteration only; forward_mla _select_topk_indices dispatch reconciled onto the new is_nextn gate; batch_result_processor _handle_finished_req renamed to _handle_finish_state_updated_req; cuda_graph_runner capture method renamed to _apply_cuda_graph_metadata. | Hand reconciliation | AC-3/AC-6/AC-7 |
| 0 | Stripped plan terminology from ported code (AC-9/AC-/DEC- comments in batch_result_processor) per Code Style rule. | - | n/a |
| 0 | retrieve_topk_graph_safe entanglement: deepseek_v2 passed recall_oracle=/score_capture= to it though it never accepted them (latent break) -> args removed with the diagnostics. | Prune | AC-3 |
| 0 | **AC-3 IMPORT GATE GREEN**: `import sglang` + dsa_backend + double_sparsity package import cleanly on the v2 tree. AC-3 negative (no dropped-module imports) also clean. v2 commits 75674dab1 (WIP port) + 5f5a9e1b7 (backend + prune). | PYTHONPATH import test | AC-3 |
| 0 | **AC-2 + AC-4 sweeps GREEN**: zero references to any of the 6 dropped modules; zero radix-fixture machinery (validator.py recorders + in-validate gate stripped). Dead metrics.record_selection + ds_recall_oracle_enabled + recall_oracle gate removed. v2 commit b8432f3ce. | Exclusion sweeps | AC-2/AC-4/AC-10 |
| 0 | **M5 mask calibrated + verified**: fresh GLM-5.1-FP8 mask via v2 calibrate.py (loop-11b recipe, `fp8_e4m3`/`label-dim 32`), loads as ChannelMask [78,64,32] page64 fp8_e4m3; content_sha256 `35155ac46ad79fa8…`. AC-5 loader-accept PASS. | v2 load_channel_mask | AC-5 |
| 0 | **DEC-8 consequence (boot)**: the v2 branch off LATEST main `105e095e0` GENUINELY needs sglang-kernel 0.4.4 — its base flash-attention path (`jit_kernel/flash_attention_v3.py`) calls the `only_qv` kernel, which 0.4.3 lacks (boot crashed `flash_attn_varlen_func() got an unexpected keyword argument 'only_qv'`). NOT a DS dep (DS-ported dsa_backend has zero `only_qv`; AC-9 holds). Resolved by upgrading the env 0.4.3 -> 0.4.4 (prebuilt abi3 wheel; both clones still import). The earlier SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK workaround was insufficient (the floor reflects real ABI use) and was removed. | serve boot crash | AC-6 (env) |
| 0 | **task5 verdict (analyze): the v2 CUDA-graph runner needs NO DS changes for correctness.** The dev cuda_graph_runner DS hunks were ENTIRELY the selector-width-ladder optimization (stamp `_ds_graph_variant_key`=(bs,width) around capture/replay). Verified invariant: capture stores at `decode_cuda_graph_metadata[_ds_decode_metadata_key(bs)]` and replay reads the same key; with the runner not stamping, `_ds_graph_variant_key` stays None so both resolve to plain `bs` → graph-safe, self-consistent (the TBO readers at 2874-2876 also use plain bs). Deliberate minimal-closure choice: ship full-width bs-keying; the perf run (M8) is the authoritative gate — revisit porting the ladder only if parity fails. | dsa_backend code inspection (1208/1296/1467) | AC-6 (boot), AC-8 (perf, TBD) |
| 0 review | **AC-8 completion rejected pending workload pin.** Review found the perf wrapper does not pass `--gsp-num-groups` / `--gsp-prompts-per-group`; stock `bench_serving` defaulted to 64 groups × 16 prompts, so the saved evidence completed 1024 requests while the wrapper verdict recorded `num_prompts=256`. The loop-11b recipe pins one GSP group and `prompts_per_group=NUM_PROMPTS`. | Review of v2 wrapper, stock dataset defaults, loop-11b benchmark script, and saved perf evidence | AC-8 pending; task13/task14/task15 remain active |
| 0 review | **Tracker reconciled.** Tasks 1-12 are moved to Completed and Verified based on diff, import, unit, calibration, boot, and abort evidence. Tasks 13-15 remain active because the AC-8 evidence must be regenerated and the provenance/final push must follow that fix. | Round 0 review tracker audit | AC-1 through AC-7, AC-9, AC-10 verified; AC-8 open |
| 1 review | **AC-8 ACCEPTED** (Codex R1 review, 10/10 ACs). Two [P3] close-out items left open: marker sweep incomplete + a false native-DSA band line in the doc. | Codex R1 review | AC-8 verified; AC-1 doc/comment polish open |
| 2 | **[P3] close-out DONE.** Stripped the markers the R1 sweep missed — Option B (config/page_table_adapter/validator×2), Tier-2.A (validator×2), Round 3 (calibrate), "round 1"→"first radix pass" (topk_kernel) — + removed metrics.py EOF blank line (git diff --check clean). Fixed DOUBLE_SPARSITY.md: native-DSA column is same-base CONTEXT only (46.50s TTFT not in band; pre-grouping-pin run), not a corrected-shape baseline; accepted result is the DS run (256/256, request_shape_ok=true, 35.05/22.90). [[perf-parity-must-pin-exact-request-shape]] | Codex R1 review [P3] ×2 | AC-1 (clean diff + accurate doc) |
| 2 | **Re-pushed** `double-sparsity-v2` HEAD 323cb7802 to `Jiminator/sglang`. Final sweep: 42 files, 0 dev-scaffolding, 0 dropped-module refs, 0 plan/workflow markers in diff, git diff --check clean, import + 114 unit tests pass. ALL 10 ACs pass; both Codex reviews' findings resolved. | M9 | AC-1/AC-8 |

#### Active Tasks
<!-- Mainline tasks only: each task must directly advance the current round objective and carry routing metadata -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| None | - | - | - | - | All mainline tasks are completed and verified |

### Blocking Side Issues
<!-- Only issues that directly block current mainline progress belong here -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
| None | - | - | - |

### Queued Side Issues
<!-- Non-blocking issues stay queued and must NOT replace the round objective -->
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| None | - | - | - |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|
| AC-1, AC-2 | task1 Inventory DS footprint → port allowlist | 0 | 0 review | v2 diff against `<BASE>` is 42 files and excludes `.pensieve`/`.humanize`/`development`/`SLOS.md`; allowlist includes DS modules, modified-upstream files, tests, perf wrapper, provenance doc |
| AC-1 | task2 Cut branch off latest origin/main; record `<BASE>` | 0 | 0 review | v2 branch `double-sparsity-v2` at `e6fda2fe9f875a4fae967cc533aff6e585c70269`; `<BASE>=105e095e005d02a178fb6c5a23bd22ba644c90e4`; remote branch present |
| AC-3 | task3 Copy pure-new `double_sparsity/` modules + calibrate.py | 0 | 0 review | 15 `python/sglang/srt/layers/attention/double_sparsity/` modules present; import gate passes |
| AC-3, AC-6, AC-7 | task4 Re-apply DS hunks + abort fix | 0 | 0 review | `import sglang`, `dsa_backend`, and `double_sparsity` pass; DS boot/abort evidence present |
| AC-3, AC-6 | task5 Retarget CUDA-graph hunks to runner/ + runner_backend/ | 0 | 0 review | selector-width ladder present in `runner/decode_cuda_graph_runner.py`; boot/perf evidence exercises CUDA graphs |
| AC-2, AC-3, AC-4, AC-10 | task6 Prune dev-only refs while porting | 0 | 0 review | dropped-module and radix-fixture sweeps return no shipped references; `record_selection` / `ds_recall_oracle_enabled` absent |
| AC-3, AC-5 | task7 Extract slim runtime + calibrate tests | 0 | 0 review | `test_double_sparsity_unit.py`, `test_lifted_budget_decode.py`, and abort test suite pass |
| AC-2, AC-3, AC-5, AC-9, AC-10 | task8 Cheap closure gates | 0 | 0 review | `PYTHONPATH=python` import gate passes; 114 unit tests pass; exclusion sweeps clean |
| AC-9 | task9 Audit dependency closure | 0 | 0 review | runtime imports resolve in v2 env; no new DS-only build dependency observed in shipped diff |
| AC-5 | task10 Calibrate mask on corpus → valid GLM-5.1-FP8 mask | 0 | 0 review | loop12 evidence records fresh mask accepted as ChannelMask `[78,64,32]`, page64, `fp8_e4m3`, content SHA `35155ac46ad79fa8...` |
| AC-6 | task11 Boot GLM-5.1-FP8 and assert meta activity | 0 | 0 review | loop12 boot evidence shows `selected_tokens=2048`, `total_tokens=5608`, `dense_fallback=0`, plus bind logs |
| AC-7 | task12 Fault-inject DS error and assert same-step abort finish | 0 | 0 review | `test/registered/unit/managers/test_ds_abort_path.py` passes with the selected unit suite |
| AC-8 | task13 Thin perf wrapper over stock bench_serving; conc-64 one trial; p50 decode TPS + P99 TTFT; check band; save evidence | 1 | 1 review | wrapper pins `--gsp-num-groups 1 --gsp-prompts-per-group 256`; evidence shows `actual_completed=256`, `request_shape_ok=true`, p50 decode TPS 35.05, P99 TTFT 22.90s, parity true; branch pushed at `f05326636` |
| AC-5, AC-8 | task14 Mask + corpus provenance doc (paths + mask content SHA) | 2 | 2 review | `benchmarks/DOUBLE_SPARSITY.md` now labels native DSA as same-base context only, notes 46.50s TTFT is not in band, and identifies the accepted corrected-shape DS result: 256/256, `request_shape_ok=true`, 35.05 TPS / 22.90s |
| AC-1, AC-2, AC-10 | task15 Final dead-code sweep + exclusion re-check; push to fork | 2 | 2 review | v2 branch `double-sparsity-v2` HEAD equals remote `323cb7802`; diff remains 42 files, no dev-scaffolding paths, no dropped-module/radix/dead-symbol refs, no plan/workflow markers in added diff lines, `git diff --check` clean, import gate and 114 focused tests pass |

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
