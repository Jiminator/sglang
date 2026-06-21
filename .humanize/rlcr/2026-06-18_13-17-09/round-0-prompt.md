Read and execute below with ultrathink

## Goal Tracker Setup (REQUIRED FIRST STEP)

Before starting implementation, you MUST initialize the Goal Tracker:

1. Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/goal-tracker.md
2. If the "Ultimate Goal" section says "[To be extracted...]", extract a clear goal statement from the plan
3. If the "Acceptance Criteria" section says "[To be defined...]", define 3-7 specific, testable criteria
4. Populate the "Active Tasks" table with MAINLINE tasks from the plan, mapping each to an AC and filling Tag/Owner
5. Record any already-known side issues in either "Blocking Side Issues" or "Queued Side Issues"
6. Write the updated goal-tracker.md

## Round Contract Setup (REQUIRED BEFORE CODING)

Before starting implementation, create @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/round-0-contract.md with:

1. **One mainline objective** for this round
2. **Target ACs** (1-2 ACs only)
3. **Blocking side issues in scope** for this round
4. **Queued side issues out of scope** for this round
5. **Round success criteria**

Use this contract to keep the round focused. Do NOT let non-blocking bugs or cleanup work replace the mainline objective.

**IMPORTANT**: The IMMUTABLE SECTION can only be modified in Round 0. After this round, it becomes read-only.

---

## Implementation Plan

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList).

Every task MUST start with exactly one lane tag:
- `[mainline]` for plan-derived work that directly advances the round objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` tasks are the primary success condition for the round
- `[blocking]` tasks may be resolved in the round only if they truly block mainline progress
- `[queued]` tasks must NOT become the round objective and do NOT need to be cleared before moving on
- If a new issue is not blocking the current objective, tag it `[queued]` and keep moving on the mainline

## Task Tag Routing (MUST FOLLOW)

Each task must have one routing tag from the plan: `coding` or `analyze`.

- Tag `coding`: Claude executes the task directly.
- Tag `analyze`: Claude must execute via `/humanize:ask-codex`, then integrate Codex output.
- Keep Goal Tracker "Active Tasks" columns **Tag** and **Owner** aligned with execution (`coding -> claude`, `analyze -> codex`).
- If a task has no explicit tag, default to `coding` (Claude executes directly).

# Loop 12 — Cut the Clean Double Sparsity Shipping Branch

> Convergence: `converged`. Codex first-pass + second-pass reviews ran; all required changes were
> folded in; every code-level claim was verified by read-only inspection (file:line evidence below);
> all open decisions were resolved by the user. No pending decisions remain.

## Goal Description

Extract the **minimal correct table-free Double Sparsity (DS) runtime** from the development branch
`dev/double-sparsity-standalone` (in clone `/sgl-workspace/sglang`) onto a **fresh branch off latest
`origin/main`** in the shipping clone `/sgl-workspace/double-sparisty-v2/sglang` (`origin =
Jiminator/sglang`), add **one simple performance eval**, and prove the extraction preserved behavior
and performance. This is a **curation + clean-port**, not new development: DS behavior is frozen.

The shipping branch must let a client clone it, **calibrate a GLM-5.1-FP8 channel mask, enable DS,
serve, and reproduce the loop-11b conc-64 result** (≈26.9 TPS / ≈25.1 s P99 TTFT) within a parity
band — DS genuinely active (not a silent dense fallback). "Performant" means **no regression vs.
loop-11b**, NOT the 30-TPS SLO floor (which neither DS nor native DSA meets at conc 64). Every file
and symbol that lands must be reachable from the DS serve path, the calibration tool, the perf
wrapper, or a shipped test. The development scaffolding (`.pensieve/`, `.humanize/`, `development/`,
`SLOS.md`, oracles, capture sinks, recall/validator harnesses, comparator, manual `dsv32` fixtures)
does **not** ship — it is how the feature was developed, not the feature.

Method is **additive minimal closure** (copy new files, hand re-apply DS hunks onto latest main,
keep a file only if its removal breaks `import sglang` / DS server boot / the conc-64 run), NOT a
merge or rebase of the divergent dev branch.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

All criteria run in the **v2 clone** against the new branch unless stated otherwise. `<BASE>` is the
recorded SHA of `origin/main` the branch was cut from.

- AC-1: **Branch hygiene & diff scope.** The branch exists on `Jiminator/sglang`, cut from latest
  `origin/main`, with `<BASE>` recorded in the branch and in `development/loop12/`.
  - Positive Tests (expected to PASS):
    - `git diff --name-only <BASE>...HEAD` lists ONLY: the shipped `double_sparsity/` modules, the
      modified-upstream files (incl. `logits_processor.py`), `calibrate.py`, the one perf wrapper
      script, the minimal feature tests, and the mask/corpus provenance doc.
    - `git ls-remote --heads origin <branch>` resolves; the branch name is free (or a documented
      fallback name).
  - Negative Tests (expected to FAIL):
    - Any path under `.pensieve/`, `.humanize/`, `development/`, or the file `SLOS.md` appears in
      the diff → fail.
    - Any file outside the allowlist appears in the diff → fail.
- AC-2: **Exclusions absent (precise sweeps, not generic grep).** Dev-only scaffolding is fully gone.
  - Positive Tests:
    - For each dropped module name — `oracle_artifact_sink`, `selection_recall_oracle`,
      `radix_fixture_capture`, `score_capture`, `selection_capture`, `latent_capture` — `rg -l` over
      `python/` and `test/` returns ZERO matches and the module files are absent.
    - `test/manual/test_dsv32_*`, `test/registered/unit/development/*`, the
      `test/registered/debug_utils/comparator/*` tree, and the oracle/recall/ac11/ac12/m3b/
      accuracy-gate tests are absent.
  - Negative Tests:
    - A leftover `import ... selection_capture` or a residual `_maybe_record_recall_oracle`
      reference exists → fail.
    - The exclusion check uses a bare `capture` grep (false-fails on legitimate CUDA-graph
      `is_current_stream_capturing` / graph capture) → reject that check; use module-name sweeps.
- AC-3: **Import & prune closure.** No shipped file imports a dropped module.
  - Positive Tests:
    - `python -c "import sglang"` exits 0; `python -c "import sglang.srt.layers.attention.dsa_backend"`
      exits 0; importing the `double_sparsity` package exits 0.
    - Per-module grep over shipped files finds no import of any dropped module.
  - Negative Tests:
    - Any import error on `import sglang` → fail.
    - Any shipped runtime/test file imports a dropped module → fail.
- AC-4: **`validator.py` ships and gates; the radix fail-closed gate is REMOVED.**
  - Positive Tests:
    - `validate_double_sparsity` is present and called unconditionally at DS startup
      (`server_args.py` `check_server_args` path); it rejects with a clear error: missing/corrupt
      mask, mask content-SHA/schema mismatch, hierarchical-cache + DS, disaggregation + DS, and a
      non-graph-safe selector without `--disable-cuda-graph`.
    - The server boots with **radix cache ON + DS** using NO fixture-state artifact and NO
      `SGLANG_DS_RADIX_OVERRIDE` env (the gate is gone, radix+DS just works).
  - Negative Tests:
    - A corrupt or SHA-mismatched mask is silently accepted → fail.
    - radix+DS is rejected or requires a fixture artifact/override → fail (the gate must be gone).
    - Any reference to `apply_radix_fixture_artifact`, `record_radix_fixture_passed`,
      `write_radix_fixture_state`, `radix_fixture_config_fingerprint`, or `RADIX_FIXTURE_STATE_*`
      remains in shipped code → fail.
- AC-5: **Calibration regenerates a valid mask (calibrate.py ships and is exercised this run).**
  - Positive Tests:
    - `calibrate.py` runs against the documented external corpus path and produces a GLM-5.1-FP8
      channel-mask safetensors that `load_channel_mask` / `validate_double_sparsity` accepts
      (schema, dtype, page-size pairing, self-consistent content SHA).
    - A `calibrate` unit/smoke test passes (logic-level, CPU-runnable).
  - Negative Tests:
    - calibrate output is rejected by the loader (bad schema/dtype/shape) → fail.
    - `calibrate.py` imports a dropped module (oracle/capture) → fail.
- AC-6: **Server boots with DS genuinely active (command-level).** Run GLM-5.1-FP8, TP=8, dsa
  backend, `glm4_moe`, FP8 KV, page 64, CUDA graphs ON, radix cache ON, NO expandable allocator,
  `--double-sparsity-config` pointing at the **freshly-calibrated** mask.
  - Positive Tests:
    - A decode response's `meta_info["double_sparsity"]` has `selected_tokens > 0`,
      `total_tokens > selected_tokens`, and `dense_fallback == 0`.
    - Startup logs show `double_sparsity bind shape check passed` (per layer) and
      `double_sparsity bind_runtime_data completed` (per selector).
  - Negative Tests:
    - `meta_info["double_sparsity"]` absent, or `total_tokens == selected_tokens` (dense fallback),
      or `dense_fallback != 0` → fail (DS is not genuinely active).
- AC-7: **Abort path carries the loop-11b fix (command/test-level).**
  - Positive Tests:
    - Injecting a DS per-request error (via the error-containment path) drives
      `req.set_finish_with_abort(...)` then `req.update_finish_state(...)` in the **same scheduler
      step**; the request finishes with an abort `finished_reason` that step.
  - Negative Tests:
    - The abort relies on the pre-#25725 `check_finished` finisher → fail.
    - The faulted request hangs or finishes with the wrong state → fail.
- AC-8: **Perf parity (exact metric + numeric band).** A thin wrapper calls **stock** `bench_serving`.
  - Positive Tests:
    - Workload mirrors conc-64 exactly: `--dataset-name generated-shared-prefix`,
      `--gsp-system-prompt-len 2253 --gsp-question-len 1843` (ISL 4096, ~55% prefix),
      `--gsp-output-len 512`, `--gsp-range-ratio 1.0`, `--max-concurrency 64`, `--backend sglang`,
      one trial, fixed `--num-prompts` and seed.
    - The wrapper emits **p50 decode TPS** (median over requests of
      `(output_tokens - 1) / decode_duration`, where `decode_duration = last_token_time -
      first_token_time`; derivable from bench_serving per-request detail / ITLs) and **P99 TTFT**.
    - p50 decode TPS ≥ `0.90 × 26.9 ≈ 24.2` AND P99 TTFT ≤ `1.20 × 25.1 ≈ 30.1 s`.
    - Evidence saved: the exact command, server args, `<BASE>` SHA, GPU info, and the bench JSON.
  - Negative Tests:
    - p50 decode TPS < 24.2 or P99 TTFT > 30.1 s → fail (regression).
    - The wrapper cannot emit p50 decode TPS or P99 TTFT → fail.
- AC-9: **Dependency closure (kernel/build deps).** The DS port introduces no new build dependency.
  - Positive Tests:
    - Shipped DS modules resolve all kernel symbols on the base (`triton` and `flash_mla_sparse_fwd`
      — 14 refs — resolve; no `deep_gemm` is newly required); `import sglang` works in a clean v2 env.
  - Negative Tests:
    - A shipped DS module imports a kernel/symbol absent on latest `origin/main` → fail.
- AC-10: **No dead code (module granularity).**
  - Positive Tests:
    - Every shipped DS module is reachable from the serve path, `calibrate.py`, the perf wrapper, or
      a shipped test. Confirmed-dead symbols are removed: `metrics.record_selection` and the
      radix-fixture recorder helpers.
  - Negative Tests:
    - A shipped module is unreferenced by any of the above → fail.
    - `metrics.record_selection` (verified dead) or any radix-fixture recorder remains → fail.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
The full minimal DS runtime: the `double_sparsity/` package **minus** the six dropped capture/oracle
modules; `validator.py` with the radix-gate machinery stripped (validation + mask load + mode gates
kept); `calibrate.py` (with its corpus reader); `metrics.py` with `record_selection` pruned;
`logits_processor.py` gaining the `per_request_summary` field; the modified-upstream DS hunks
re-applied onto latest main (CUDA-graph hunks retargeted to the `runner/` + `runner_backend/`
layout; abort fix carried). Plus: one perf wrapper around stock `bench_serving`; the minimal feature
tests (runtime + calibrate + lifted-budget if it ships); and a short doc recording mask + corpus
provenance (external paths + the calibrated mask's content SHA). The freshly-calibrated mask is
consumed by the serve/perf run.

### Lower Bound (Minimum Acceptable Scope)
Everything in the Upper Bound **except** `lifted_budget.py` and its test — droppable as a coupled
pair (keep-both-or-drop-both) **only if** the performant serving config does not enable lifted-budget
decode AND the import/boot/perf closure still holds without it. No other reductions are acceptable:
`calibrate.py`, `validator.py` (gate-stripped), the `meta_info["double_sparsity"]` activity signal,
and radix-on-without-a-gate are all required by the resolved decisions and the acceptance criteria.

### Allowed Choices
- Can use: additive minimal closure (copy new files + hand re-apply/retarget hunks); stock
  `bench_serving` plus a thin wrapper that defines the parity metric; an external documented
  calibration corpus path and an external documented mask path (with recorded SHA); branching off
  latest `origin/main` with `<BASE>` recorded; a documented fallback branch name if taken.
- Cannot use: a merge or rebase of the dev branch onto main; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
  for serving; committing `development/`, `.humanize/`, or `.pensieve/`; pushing to the public
  upstream `sgl-project/sglang`; the radix fixture-state gate (must be removed); porting the
  `--warmup-seconds`/`--measurement-window-seconds` window flags into `bench_serving` (option B is
  rejected); a bare `capture` grep as the exclusion check.

> **Note on Determinism:** Several choices are fixed by the resolved decisions and are NOT open:
> calibrate.py ships and is exercised this run; the radix gate is removed; perf eval = stock
> `bench_serving` + wrapper; parity band = decode TPS ≥ −10% / P99 TTFT ≤ +20%; the eval runs
> against the freshly-calibrated mask. For these, upper and lower bounds converge.

> **Note on Deterministic Designs**: If the draft specifies a highly deterministic design with no choices (e.g., "must use JSON format", "must use algorithm X"), then the path boundaries should reflect this narrow constraint. In such cases, upper and lower bounds may converge to the same point, and "Allowed Choices" should explicitly state that the choice is fixed per the draft specification.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Order matters: **apply and prune per file, then run global closure** — pruning before re-applying
hunks would reintroduce dev references (Codex second-pass finding).

```
1. Inventory the real DS footprint on the dev clone (new files + modified-upstream hunks + tests).
2. Branch off latest origin/main in the v2 clone; record <BASE>.
3. Copy the pure-new runtime files; copy calibrate.py.
4. Re-apply each modified-upstream DS hunk onto main's CURRENT file, pruning dev-only branches AS
   you touch each file:
     - drop score_capture/selection_capture/latent_capture guard blocks,
     - delete the entangled _maybe_record_recall_oracle() path (+ its param on absorbed_topk_select
       + callers) so oracle_artifact_sink/selection_recall_oracle disappear,
     - strip the radix-gate code from validator.py (keep validate + mask load + mode gates),
     - retarget CUDA-graph hunks to model_executor/runner/ + runner_backend/,
     - add the per_request_summary field to logits_processor.py,
     - carry the abort fix (set_finish_with_abort + update_finish_state).
5. Static/import/unit closure (cheap, before GPU): import sglang clean; no dropped-module imports;
   slim runtime + validator + calibrate unit tests pass.
6. Calibrate: run calibrate.py on the documented corpus -> a GLM-5.1-FP8 mask the loader accepts.
7. DS boot: serve GLM-5.1-FP8 (TP8, dsa, glm4_moe, FP8 KV, page64, CUDA graphs, radix) on that mask;
   assert meta_info["double_sparsity"] selected<total, dense_fallback==0.
8. Abort test (fault injection) -> abort in the same scheduler step.
9. Perf: thin wrapper over stock bench_serving -> p50 decode TPS + P99 TTFT within the band.
10. Dead-code sweep (module granularity) -> remove record_selection + any unreferenced module.
11. Push to fork (owner-authorized).
```

The two expensive 8×H200 steps (boot, perf) come AFTER the cheap closure gates, so most defects are
caught without burning GPU time. One TP=8 server at a time; tear down and wait for GPU idle between
boot and perf.

### Relevant References
- `python/sglang/srt/layers/attention/double_sparsity/__init__.py:28` — re-exports
  `validate_double_sparsity` (proves `validator.py` is runtime, must ship).
- `python/sglang/srt/server_args.py:7214-7223` — `check_server_args` calls
  `apply_radix_fixture_artifact` (REMOVE) + `validate_double_sparsity` (KEEP) at DS startup.
- `python/sglang/srt/layers/attention/double_sparsity/validator.py` — `validate_double_sparsity`
  (keep), radix recorders `record_radix_fixture_passed`/`write_radix_fixture_state`/
  `radix_fixture_config_fingerprint`/`apply_radix_fixture_artifact` + `RADIX_FIXTURE_STATE_*` (strip).
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` —
  `_maybe_record_recall_oracle()` (~line 602–700, imports `oracle_artifact_sink`:631 /
  `selection_recall_oracle`:688) called from the hot path (~line 1210); `score_capture` guard
  (1186-1199). Entangled prune target.
- `python/sglang/srt/model_executor/model_runner.py:3256-3263` — `selection_capture` guard block
  (conditional on `enable_double_sparsity`); `:3240-3250` — DS summary transfer to
  `LogitsProcessorOutput`.
- `python/sglang/srt/models/deepseek_v2.py` — `_publish_ds_request_summary` /
  `meta_info_for_request` (DS-active signal source); `latent_capture` guard (2381-2392).
- `python/sglang/srt/layers/attention/double_sparsity/metrics.py` — `meta_info_for_request`,
  `record_error`, `mark_channel_mask_valid` LIVE; `record_selection` DEAD (prune).
- `python/sglang/srt/layers/logits_processor.py:111` — add `per_request_summary` dataclass field
  (the modified file the draft missed).
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:225` — DS abort:
  `set_finish_with_abort` + `update_finish_state` (both APIs exist on target main:
  `schedule_batch.py:1539, 1403`).
- v2 main drift: `model_executor/cuda_graph_runner.py` is GONE → logic split into
  `model_executor/runner/{base,decode,prefill}_cuda_graph_runner.py` +
  `model_executor/runner_backend/{base,full,breakable,tc_piecewise}_cuda_graph_backend.py`
  (#23906, #28081). Retarget DS CUDA-graph hunks here.
- `test/registered/unit/layers/attention/test_lifted_budget_decode.py` — keep as-is (zero
  scaffolding). `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — extract the
  runtime + calibrate test classes; drop its radix-fixture/capture-dependent classes.
- Stock `bench_serving` already has the `generated-shared-prefix` dataset + `--gsp-*` flags; the
  wrapper only fixes args and derives p50 decode TPS from per-request detail.

## Dependencies and Sequence

### Milestones
1. **M1 — Inventory & branch.** Inventory the real DS footprint; cut the branch off latest
   `origin/main` in the v2 clone; record `<BASE>`. (Gates AC-1.)
2. **M2 — Copy pure-new files.** Copy the new `double_sparsity/` runtime modules + `calibrate.py`
   into the v2 tree. (Depends on M1.)
3. **M3 — Re-apply + prune per file.** Re-apply each modified-upstream DS hunk onto main's current
   file, pruning dev-only branches as each file is touched: drop capture guard blocks; delete the
   entangled recall-oracle path; strip the radix gate from `validator.py`; retarget CUDA-graph hunks
   to `runner/` + `runner_backend/`; add the `logits_processor.py` field; carry the abort fix.
   (Depends on M2; gates AC-3, AC-4, AC-7.)
4. **M4 — Cheap closure gates.** `import sglang` clean; no dropped-module imports; precise exclusion
   sweeps; slim runtime + validator + calibrate unit tests pass. Run BEFORE any GPU. (Depends on M3;
   gates AC-2, AC-3, AC-5, AC-9, AC-10 first pass.)
5. **M5 — Calibrate the mask.** Run `calibrate.py` on the documented corpus → a GLM-5.1-FP8 mask the
   loader accepts; record its content SHA. (Depends on M4; gates AC-5.)
6. **M6 — DS boot active.** Serve GLM-5.1-FP8 (TP8, dsa, glm4_moe, FP8 KV, page64, CUDA graphs,
   radix) on the calibrated mask; assert the activity signal. (Depends on M5; gates AC-6. One TP=8
   server at a time.)
7. **M7 — Abort fault-injection test.** Drive the DS abort path; assert same-step abort finish.
   (Depends on M6; gates AC-7.)
8. **M8 — Perf parity.** Tear down the boot server, wait for GPU idle, run the wrapper → p50 decode
   TPS + P99 TTFT within band; save evidence. (Depends on M6; gates AC-8.)
9. **M9 — Final sweep & push.** Module-granularity dead-code sweep; full exclusion re-check; push to
   the fork (owner-authorized). (Depends on M4, M6, M8; gates AC-1, AC-2, AC-10.)

Dependency shape: M1→M2→M3→M4 is the build-and-verify-cheap chain; M4 gates the expensive GPU work
(M5→M6→{M7, M8}); M9 closes out after the cheap gates and the GPU evidence. The `lifted_budget`
keep/drop coupling is resolved during M4 (closure) and confirmed at M6/M8 (does the performant
config use it?).

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Inventory the real DS footprint (new files, modified-upstream hunks, tests); produce the port allowlist | AC-1, AC-2 | analyze | - |
| task2 | Cut branch off latest `origin/main` in the v2 clone; record `<BASE>`; verify free branch name | AC-1 | coding | task1 |
| task3 | Copy pure-new `double_sparsity/` runtime modules + `calibrate.py` into the v2 tree | AC-3 | coding | task2 |
| task4 | Re-apply DS hunks for `dsa_backend.py`, `server_args.py`, `deepseek_v2.py`, `forward_mla.py`, `forward_mha.py`, `model_runner*.py`, `pool_configurator.py`, `mem_cache/*`, `managers/*`, adding `logits_processor.py` field and carrying the abort fix | AC-3, AC-6, AC-7 | coding | task3 |
| task5 | Retarget CUDA-graph DS hunks from the old `cuda_graph_runner.py` to `runner/` + `runner_backend/` on current main | AC-3, AC-6 | analyze | task4 |
| task6 | Prune dev-only references while porting: capture guard blocks, the entangled `_maybe_record_recall_oracle()` path, and the radix-gate machinery in `validator.py`; prune dead `record_selection` | AC-2, AC-3, AC-4, AC-10 | coding | task4 |
| task7 | Extract the slim runtime + calibrate test set from `test_double_sparsity_unit.py`; keep `test_lifted_budget_decode.py`; add a calibrate smoke test; drop scaffolding tests | AC-3, AC-5 | coding | task6 |
| task8 | Cheap closure gates: `import sglang`, no dropped-module imports, precise exclusion sweeps, unit tests; resolve `lifted_budget` keep/drop | AC-2, AC-3, AC-5, AC-9, AC-10 | coding | task5, task6, task7 |
| task9 | Audit dependency closure (triton / `flash_mla_sparse_fwd`; confirm no new build dep) against latest main | AC-9 | analyze | task8 |
| task10 | Run `calibrate.py` on the documented corpus → valid GLM-5.1-FP8 mask; record content SHA | AC-5 | coding | task8 |
| task11 | Boot GLM-5.1-FP8 (TP8, dsa, glm4_moe, FP8 KV, page64, CUDA graphs, radix) on the calibrated mask; assert `meta_info["double_sparsity"]` activity | AC-6 | coding | task10 |
| task12 | Fault-inject a DS per-request error; assert same-step `set_finish_with_abort` + `update_finish_state` | AC-7 | coding | task11 |
| task13 | Write the thin perf wrapper over stock `bench_serving`; run conc-64 one trial; emit p50 decode TPS + P99 TTFT; check the band; save evidence | AC-8 | coding | task11 |
| task14 | Write the mask + corpus provenance doc (external paths + calibrated mask content SHA) | AC-5, AC-8 | coding | task10 |
| task15 | Final module-granularity dead-code sweep + full exclusion re-check; push to fork (owner-authorized) | AC-1, AC-2, AC-10 | coding | task8, task11, task13 |

## Claude-Codex Deliberation

### Agreements
- Additive minimal closure (copy + hand re-apply hunks) is the right porting method; not a
  merge/rebase of the divergent dev branch.
- `validator.py` must ship (it is runtime); `calibrate.py` quality/oracle/capture/comparator modules
  and dev tests do not belong in the runtime closure by default.
- `meta_info["double_sparsity"]` is the correct "DS genuinely active" proof.
- The missed modified file `logits_processor.py` (one dataclass field) and the CUDA-graph hunk
  retarget to `runner/` + `runner_backend/` must be in the plan.
- The two expensive 8×H200 steps run only after cheap import/unit closure gates pass.

### Resolved Disagreements
- **`validator.py` classification (draft said drop):** Verified RUNTIME — `__init__.py:28` exports
  `validate_double_sparsity`; `server_args.py:7214-7223` calls it at startup. Resolution: ships, with
  the radix-gate machinery stripped. Rationale: removing it breaks DS init.
- **Drop = file deletion (draft implied):** Verified dropping dev modules needs ~900 lines of
  pruning (entangled `_maybe_record_recall_oracle`, capture guard blocks). Resolution: prune-while-
  porting per file. Rationale: deletion alone leaves dangling references.
- **DS-active transport set (Codex first-pass overstated):** Codex named `output_streamer`,
  `detokenizer_manager`, `multi_tokenizer_mixin`; verified they carry NO DS code. Only
  `logits_processor.py` is the extra file. Resolution: add just that field. Rationale: evidence.
- **Milestone order (Claude v1 pruned before re-applying hunks):** Codex flagged this reintroduces
  dev refs. Resolution: apply+prune per file, then global closure. Rationale: correctness of order.
- **AC-2/AC-3 exclusion checks (Claude v1 used broad grep):** Codex flagged false-fails on legit
  CUDA-graph capture + missed attribute reads. Resolution: precise per-module-name sweeps.
- **AC-5/AC-6/AC-7 testability (Claude v1 under-specified):** Resolution: made command/test-level
  with explicit injection + state assertions (now AC-6/AC-7) and command-level boot (AC-6).
- **AC-7/AC-8 perf metric (Claude v1 "within noise"):** Resolution: exact p50 decode TPS formula +
  numeric band (decode TPS ≥ −10% / P99 TTFT ≤ +20%, user-set).
- **AC-8 granularity (Claude v1 per-symbol):** Resolution: relaxed to per-module reachability;
  helpers allowed; only confirmed-dead symbols (`record_selection`) pruned.
- **`lifted_budget` contradiction (Claude v1 dropped it in lower bound but kept its test):** Codex
  caught the conflict. Resolution: keep-both-or-drop-both, closure-decided.
- **`cuda_graph_runner.py` drift:** Verified refactored away on main. Resolution: retarget hunks to
  the new runner layout (a known high-risk hand-port, not a blocker).
- **Abort/finisher API:** Verified `set_finish_with_abort` + `update_finish_state` exist on target
  main; the loop-11b fix carries cleanly.

### Convergence Status
- Final Status: `converged`
- Rounds: Codex first-pass (assumptions/risks) + one second-pass reasonability review. All
  REQUIRED_CHANGES were folded in; the residual UNRESOLVED items were all genuine user decisions and
  are now resolved (see Pending User Decisions — all marked with a final decision).

## Pending User Decisions

All decisions are RESOLVED. None block implementation.

- DEC-1: Calibration tooling — does `calibrate.py` ship?
  - Claude Position: Lean runtime-only (verified `calibrate.py` is cleanly droppable, zero runtime refs).
  - Codex Position: N/A — open question; do not keep `calibrate.py` merely to solve mask provenance.
  - Tradeoff Summary: Shipping calibration is a larger surface but lets a client recalibrate for new
    models/quant; runtime-only ships only pre-calibrated models.
  - Decision Status: **SHIP `calibrate.py`, and exercise it this run** (regenerate the mask via
    calibrate, then serve/eval against it). Calibration tooling is in-scope; its corpus reader and
    calibrate tests ship.
- DEC-2: "DS is active" signal.
  - Claude Position: Keep `meta_info["double_sparsity"]` (already runtime) + startup bind logs;
    prune dead `record_selection`.
  - Codex Position: Reasonable; DEC-2 can be closed.
  - Tradeoff Summary: The per-request meta path already exists in the DS request lifecycle; it is the
    cheapest host-side proof DS is genuinely sparse.
  - Decision Status: **RESOLVED — keep `meta_info["double_sparsity"]` + bind logs; prune
    `record_selection`** (`metrics.py` ships as load-bearing for the signal).
- DEC-3: Perf-eval fidelity.
  - Claude Position: Option A — stock `bench_serving` + thin wrapper that defines the metric.
  - Codex Position: Acceptable; collapse to stock + wrapper if the wrapper defines the parity metric.
  - Tradeoff Summary: A = smallest diff, wider noise band; B = port window flags for tighter parity,
    slightly larger diff.
  - Decision Status: **A — stock `bench_serving` + thin wrapper.** Option B (porting window flags)
    is rejected.
- DEC-4: Channel-mask provenance.
  - Claude Position: External documented path + recorded SHA, or commit a sanitized artifact.
  - Codex Position: Self-contained branch artifact vs external artifact path — UNRESOLVED; do not
    keep `calibrate.py` just to solve this.
  - Tradeoff Summary: Committing a multi-MB mask bloats the repo; an external path needs a recorded
    SHA; regeneration needs `calibrate.py` + a corpus.
  - Decision Status: **Regenerate the mask via the shipped `calibrate.py`** (couples to DEC-1); the
    branch consumes the calibrated mask by documented path with its content SHA recorded.
- DEC-5: Test floor.
  - Claude Position: Keep `test_lifted_budget_decode.py`; extract the slim runtime + (now) calibrate
    test set from `test_double_sparsity_unit.py`; drop oracle/recall/ac11/ac12/m3b/manual tests.
  - Codex Position: Extract, do not copy (the keeper test imports calibrate/capture/oracle).
  - Tradeoff Summary: Copying drags scaffolding; extraction keeps a real runtime guard.
  - Decision Status: **RESOLVED — extract runtime + calibrate tests (calibrate now ships), keep
    lifted-budget test if the module ships, drop the rest.**
- DEC-6: Radix-cache acceptance / the fail-closed radix gate.
  - Claude Position: Three options — ship/regenerate fixture artifact, use override env, or radix-off.
  - Codex Position: UNRESOLVED — split radix acceptance; add a fail-closed negative test.
  - Tradeoff Summary: The loop's radix fixture-state gate (in `validator.py`) is dev-built safety
    machinery; on a validated shipping branch it is dead/awkward product behavior.
  - Decision Status: **Remove the gate so radix + DS just work** (drop `apply_radix_fixture_artifact`,
    the recorders, the fingerprint, and `RADIX_FIXTURE_STATE_*`); rely on loop-11b validation.
- DEC-7: Perf parity tolerance (single trial, GPU variance).
  - Claude Position: Propose decode TPS ≥ −10% / P99 TTFT ≤ +20% of loop-11b (26.9 TPS / 25.1 s).
  - Codex Position: UNRESOLVED — needs human agreement (one trial + GPU variance).
  - Tradeoff Summary: Tighter risks flakiness; looser only catches big regressions; report-only has
    no automated gate.
  - Decision Status: **decode TPS ≥ 24.2 (−10%) AND P99 TTFT ≤ 30.1 s (+20%).**
- DEC-8: Base commit.
  - Claude Position: Pin to `105e095e0` (already checked out, lowest drift risk).
  - Codex Position: UNRESOLVED — pin exact vs refresh to latest; do not leave "current main" ambiguous.
  - Tradeoff Summary: Pinning minimizes hand-reconciliation; latest is a cleaner PR base but may add
    fresh drift beyond the mapped CUDA-graph refactor.
  - Decision Status: **Branch off latest `origin/main`** (`git fetch` first); record the resolved
    `<BASE>` SHA in the branch and in `development/loop12/`. The closure check absorbs any added drift.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.

### Hard Operational Constraints (carry forward — do not relitigate)
- **NEVER** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving.
- **One TP=8 server at a time.** Tear down before booting the next; wait for GPU idle (between M6 boot and M8 perf).
- Do **not** run blanket `nvidia-smi` GPU PID kills; do **not** `pkill -f` with a pattern that matches the parent shell.
- All serving/perf work happens in the **v2 clone** (`/sgl-workspace/double-sparisty-v2/sglang`); loop machinery and this plan stay in **this** clone (`/sgl-workspace/sglang`). The shipping branch must never receive `development/`, `.humanize/`, or `.pensieve/`.
- Push only to the **fork** (`Jiminator/sglang`), owner-authorized; **never** the public upstream `sgl-project/sglang`.
- The serve/perf run needs GLM-5.1-FP8 weights + the freshly-calibrated channel mask present at run time; calibration needs the documented external corpus path.

### Operating Notes
- The reviewer diffs the **v2 branch against its base (`<BASE>`)**, not against this dev branch.
- Round work edits files under the v2 clone; round summaries / goal-tracker live under this clone's `development/loop12/`.
- The classification of files into IN/OUT in the draft is a **starting hypothesis**; the authoritative selector is the closure check (import / DS boot / conc-64 run). Log every dropped and every kept file — no silent omissions or inclusions.

--- Original Design Draft Start ---

# Loop 12 Draft — Cut the clean Double Sparsity shipping branch

> Written 2026-06-18. Loop 11/11b produced the *validated* table-free Double Sparsity (DS)
> candidate on GLM-5.1-FP8 (8×H200): DS serves, is correct, and was measured against the client
> SLOs (PASS @ conc 16/32, honest FAIL @ conc 64 = 26.9 TPS / 25.1 s P99 TTFT, where native DSA
> also fails). That work lives on `dev/double-sparsity-standalone` — a **development** branch caked
> in loop machinery: `.pensieve/`, `.humanize/`, `development/` (every loop's logs, evals, oracles,
> calibration sweeps), `SLOS.md`, and a `double_sparsity/` package where roughly half the files are
> dev-only oracle/capture/validation scaffolding.
>
> Loop 12 does ONE thing: **cut the branch we actually ship the feature from.** A fresh branch off
> `Jiminator/sglang` `main`, carrying *only* the runtime needed to serve DS plus *one* simple
> performance eval — no loop scaffolding, no evals we don't need, no dead code. The bar is: a client
> can clone this branch, enable DS, serve GLM-5.1-FP8, and the feature works and performs exactly as
> the loop-11b candidate did.
>
> Feed this through `gen-plan` once scope is confirmed.

---

## What this is (and is NOT) — read first

**This is a curation + clean-port loop, not new development.** The algorithm is done and validated.
We are not changing DS behavior, not re-opening the selection/scoring pipeline, not chasing the
conc-64 number. We are extracting the *minimal correct runtime* from a messy dev branch onto a clean
base, and proving the extraction preserved behavior and performance.

**"Performant and shippable" is parity, not a new SLO.** The validated candidate does NOT meet the
30 TPS floor at conc 64 (it lands ≈26.9 TPS; so does DSA — that is the honest envelope edge). Loop
12 does not try to fix that. "Performant" here means **no regression vs. the loop-11b numbers**: the
clean-ported branch must reproduce the validated conc-64 result within noise. The perf eval is a
**regression gate**, not a pass/fail SLO gate.

**A clean, smaller diff is the deliverable.** Every file and symbol that lands on the shipping branch
must be reachable from the DS serving path or the one perf script. If it isn't, it doesn't ship.

---

## Objective

Produce `double-sparsity-v2` (or next free name) on `Jiminator/sglang`, branched from current `main`,
containing:

1. **The minimal DS runtime** — exactly the source needed to serve table-free Double Sparsity on
   GLM-5.1-FP8 with the performant knobs the client requires (TP, CUDA graphs, radix cache), and
   nothing else.
2. **One performance eval script** — a single, simple, one-trial benchmark that mirrors the conc-64
   workload we have been running (4096 ISL / 512 OSL, gsp ~55% prefix), emitting client-visible
   decode TPS + P99 TTFT.
3. **Proof it works** — server boots with DS enabled, DS is genuinely active (not silently falling
   back to dense), and the perf eval reproduces the loop-11b conc-64 numbers within noise.

Explicitly **excluded** from the branch: `.pensieve/`, `.humanize/`, `development/` (the whole tree),
`SLOS.md`, every loop log, and all DS dev-only scaffolding (oracles, capture sinks, recall/validator
harnesses, calibration sweeps, the AC-11 comparator + evidence gates, manual `test_dsv32_*`
fixtures). None of these are part of the feature; they are how we *developed* it.

---

## Working model — two clones, do not cross the streams

The user has set up two independent checkouts so loop machinery and shipping code never mix:

- **`/sgl-workspace/sglang`** (THIS clone) — branch `dev/double-sparsity-standalone`. Holds all dev
  history, logs, `.humanize/`/`.pensieve/`, and `development/loop12/` (this draft + the loop's own
  plan/summaries). **The loop-12 RLCR machinery and this draft live here and NEVER get committed to
  the shipping branch.** This is also the *source of truth* we copy DS code FROM.
- **`/sgl-workspace/double-sparisty-v2/sglang`** (the v2 clone) — `origin =
  https://github.com/Jiminator/sglang.git`, currently on clean `main` @ `105e095e0`. **All shipping
  code changes, the new branch, and all serving/perf runs happen HERE.** Confirmed clean: zero DS
  references in `python/sglang/` today.

Implication for the RLCR loop: round work edits and tests files under the v2 clone; round summaries
and goal-tracker live under this clone's `development/loop12/`. The reviewer diffs the v2 branch
against its base (`main`), not this dev branch.

---

## Branch setup

In the v2 clone:

1. `git fetch origin`, then branch from `origin/main` (current `main`, `105e095e0` or newer):
   `git switch -c double-sparsity-v2 origin/main`.
2. If `double-sparsity-v2` already exists on the fork, fall back to a free name
   (`double-sparsity-shipping`, `double-sparsity-v2a`, …) — verify against `git ls-remote --heads
   origin` before creating.
3. Push to `Jiminator/sglang` (the fork) once the branch is real and owner-authorized. **Never push
   to the public upstream `sgl-project/sglang`.**

---

## Scope — what to port (IN) and what to drop (OUT)

The DS footprint on the dev branch is two groups: **new files** (the `double_sparsity/` package +
DS tests) and **modified upstream files** (wiring). The classification below is the *starting
hypothesis* from a filename/role scan — it is NOT authoritative. **The authoritative selector is the
minimal-closure test (see "Port strategy"): a file ships only if removing it breaks `import sglang`,
breaks server boot with DS enabled, or breaks the conc-64 perf run.**

### New runtime files — copy wholesale (IN, pending closure check)
`python/sglang/srt/layers/attention/double_sparsity/`:
- `__init__.py`, `config.py` — package + DS config
- `absorbed_latent.py`, `absorbed_latent_kernel.py` — the served table-free scoring path
- `selection_kernel.py`, `topk_kernel.py`, `selector.py` — selection runtime
- `channel_mask.py` — runtime mask apply (the calibrated mask is consumed at serve time)
- `cuda_graph.py`, `page_table_adapter.py` — CUDA-graph + page-table runtime (a required knob)
- `lifted_budget.py`, `error_containment.py` — decode budget + runtime safety
- `metrics.py` — **DECISION (DEC):** ships only if the "DS is active" signal is kept (see open
  decisions). If the no-op instrumentation is dropped, this and its plumbing go too.

### New files — drop (OUT: dev-only eval / oracle / calibration)
`oracle_artifact_sink.py`, `selection_recall_oracle.py`, `radix_fixture_capture.py`,
`score_capture.py`, `selection_capture.py`, `latent_capture.py`, `validator.py`, and **`calibrate.py`
(DEC — see open decisions: a client serving a *new* model needs a way to produce a channel mask; a
client running the shipped GLM-5.1 mask does not).** All `test/manual/test_dsv32_*`,
`_dsv32_quality_smoke_lib.py`, `_m3b_label_capture_verdict.py`, and the
`test/registered/unit/development/*` (AC-11 comparator, bench-meta writer, option-b scripts) +
`test/registered/unit/.../test_selection_recall_oracle.py` / `test_oracle_sink_and_force.py` /
`test_ac12_helpers.py` / `test_m3b_label_capture_verdict.py` / `test_accuracy_gate_compare.py` are
loop-validation harnesses, not feature tests.

### Modified upstream files — re-apply DS hunks onto current `main` (IN, pending closure check)
These cannot be blind-copied: `main` has drifted. Re-apply only the DS-relevant hunks, reconciled
against main's current code.
- `python/sglang/srt/layers/attention/dsa_backend.py` — DS additions to the DSA backend
- `python/sglang/srt/server_args.py` — the `--enable-double-sparsity` (+ related) CLI knobs
- `python/sglang/srt/model_executor/{model_runner.py, model_runner_kv_cache_mixin.py,
  cuda_graph_runner.py, pool_configurator.py}` — DS enablement, KV/pool, CUDA-graph wiring
- `python/sglang/srt/models/deepseek_v2.py` and
  `models/deepseek_common/attention_forward_methods/{forward_mla.py, forward_mha.py}` — model
  forward integration (GLM-5.1 `glm4_moe` reuses the DeepSeek MLA path)
- `python/sglang/srt/mem_cache/{memory_pool.py, memory_pool_host.py}` — DS KV pool
- `python/sglang/srt/managers/{schedule_batch.py, scheduler.py, io_struct.py,
  tokenizer_manager.py, scheduler_components/batch_result_processor.py}` — DS request lifecycle.
  **Carry forward the loop-11b fix: the DS abort path must call `req.update_finish_state()` (NOT the
  pre-#25725 `check_finished`)** — verify it matches main's current finisher API.

### Tests — keep only feature tests (IN)
The unit tests that exercise *shipped runtime*: `test_double_sparsity_unit.py`,
`test_lifted_budget_decode.py` (keep only if `lifted_budget` ships and the test needs no oracle
fixtures). Drop everything that imports an oracle/capture/comparator. **DECISION:** decide the test
floor — the minimum set that guards the runtime — rather than porting tests reflexively.

### `bench_serving.py` — stays stock (OUT of the diff)
The DS-meta capture we added to `bench_serving.py` is eval instrumentation. The perf eval (below)
uses **stock main `bench_serving`** for client-visible TPS/TTFT, so this file is left untouched from
`main`. (gsp dataset + `--gsp-*` already exist in main; only our window-measurement flags don't —
see perf-eval decision.)

---

## Port strategy — additive minimal closure (NOT a merge/rebase)

The dev branch and `Jiminator/sglang main` do not share a clean recent ancestor, and a merge would
drag the whole mess across. So **build up, don't tear down**:

1. **Start from the blank base** (`main`). Add DS, don't subtract scaffolding.
2. **New files:** copy from this clone's tree into the v2 tree.
3. **Modified files:** re-apply each DS hunk onto main's *current* version of the file, reconciling
   drift by hand (e.g. the `check_finished → update_finish_state` rename already landed in main).
4. **Closure check = the gate.** After each addition, in the v2 clone: `python -c "import sglang"`
   clean → server boots with DS enabled → conc-64 perf run produces numbers. Any file not required
   by that chain does not ship. Log what was dropped (no silent omissions, no silent inclusions).
5. **Dead-code sweep before done:** grep every shipped DS module/symbol for a live reference from
   the serving path or the perf script; anything unreferenced is removed, not "kept just in case."

---

## The one performance eval

A single script in the v2 branch (e.g. `benchmarks/` or a top-level `bench_double_sparsity.sh` —
pick a home that reads as product tooling, not loop scaffolding). It must:

- Mirror the **conc-64** workload exactly: `--dataset-name generated-shared-prefix`,
  `--gsp-system-prompt-len 2253 --gsp-question-len 1843` (ISL 4096, ~55% prefix),
  `--gsp-output-len 512` (OSL), `--gsp-range-ratio 1.0`, `--max-concurrency 64`, **one trial**,
  `--backend sglang`.
- Emit the client-visible numbers the SLO is defined on: **decode TPS (p50)** and **P99 TTFT**.
- Be dead simple — one model, one concurrency, one trial. No sweep, no comparator, no DSA side, no
  evidence gates.

**DECISION (perf-eval fidelity):** stock main `bench_serving` lacks our `--warmup-seconds /
--measurement-window-seconds` steady-state flags. Two options:
- **(A, recommended) Simple/stock:** plain `--num-prompts N --max-concurrency 64`, no window flags,
  nothing ported into `bench_serving`. Simplest; numbers include some ramp/drain so parity is
  judged with a wider noise band.
- **(B) Faithful:** port *only* the small, self-contained window-measurement flags into the
  branch's `bench_serving` so the eval reports steady-state decode TPS exactly like loop-11b.
  More faithful parity, slightly larger diff.

**Parity target (regression gate):** the eval reproduces the loop-11b conc-64 result — DS ≈ **26.9
TPS / ≈25.1 s P99 TTFT** — within noise. PASS = "matches the validated candidate," NOT "≥30 TPS."

---

## Acceptance criteria (rough — gen-plan will formalize as AC-X)

1. Branch `double-sparsity-v2` (or free name) exists on `Jiminator/sglang`, branched from current
   `main`; `git diff main...branch` touches **only** DS runtime + the one perf script + minimal
   feature tests.
2. **Exclusions verified absent** on the branch: no `.pensieve/`, no `.humanize/`, no `development/`,
   no `SLOS.md`, no oracle/capture/calibration/comparator files, no `test/manual/test_dsv32_*`, no
   `test/registered/unit/development/*`.
3. `python -c "import sglang"` is clean; the DS feature tests pass.
4. Server boots in the v2 clone with DS enabled on GLM-5.1-FP8 (8×H200, dsa backend, glm4_moe), with
   TP + CUDA graphs + radix cache on; DS is **genuinely active** (selection path runs; not a silent
   dense fallback).
5. The perf eval runs the conc-64 workload, one trial, and reports decode TPS + P99 TTFT **within
   noise of the loop-11b candidate (≈26.9 TPS / ≈25.1 s)** — regression gate, not SLO gate.
6. **No dead code:** every shipped DS module/symbol is referenced from the serving path or the perf
   script (dead-code sweep clean).

---

## Constraints (hard-won — carry forward, do not relitigate)

- **NEVER** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving.
- **One TP=8 server at a time.** Tear down before booting the next; wait for GPU idle.
- Do **not** run blanket `nvidia-smi` GPU PID kills; do **not** `pkill -f` with a pattern that
  matches the parent shell.
- All serving/perf work happens in the **v2 clone**; loop machinery + this draft stay in **this**
  clone. The shipping branch must never receive `development/`, `.humanize/`, or `.pensieve/`.
- Push only to the **fork** (`Jiminator/sglang`), owner-authorized; never the public upstream.
- The perf eval needs the GLM-5.1-FP8 weights + a calibrated channel mask present at run time
  (operational dependency — see open decisions on whether the branch can regenerate the mask).

---

## Open decisions for the user / gen-plan

- **DEC — calibration tooling:** does the shipping branch include `calibrate.py` (so a client can
  produce a channel mask for a new model / new quant), or is it runtime-only consuming a pre-made
  GLM-5.1 mask? Shipping calibration is a meaningfully larger surface; leaving it out means the
  branch only serves models we pre-calibrated. *Lean: runtime-only for v2, document mask provenance;
  revisit if the client needs to recalibrate.*
- **DEC — "DS is active" signal:** keep a lightweight runtime signal that DS is genuinely sparse
  (the `metrics.py` no-op fields, or just a startup log line), or drop all metric plumbing for the
  smallest diff? *Lean: keep a minimal, cheap signal (one log line / one meta field) — zero GPU
  sync, host-side — so "is DS actually on?" is answerable in production without the full dev gate.*
- **DEC — perf-eval fidelity:** option A (stock/simple) vs option B (port window-measurement flags).
  *Lean: A, unless tight parity to the loop-11b TPS is required to sign off.*
- **DEC — channel mask provenance:** where does the GLM-5.1 mask the eval uses come from on a clean
  branch (committed artifact? documented external path? regenerated via calibrate if it ships)?
- **DEC — test floor:** minimum feature-test set that guards the runtime without dragging oracle
  fixtures.

--- Original Design Draft End ---

---

## BitLesson Selection (REQUIRED FOR EACH TASK)

Before executing each task or sub-task, you MUST:

1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task to select relevant lesson IDs
3. Follow the selected lesson IDs (or `NONE`) during implementation

Include a `## BitLesson Delta` section in your summary with:
- Action: none|add|update
- Lesson ID(s): NONE or comma-separated IDs
- Notes: what changed and why (required if action is add or update)

Reference: @/sgl-workspace/sglang/.humanize/bitlesson.md

---

## Goal Tracker Rules

Throughout your work, you MUST maintain the Goal Tracker:

1. **Before starting a round**: Re-anchor on the original plan and current round contract
2. **Before starting a task**: Mark the relevant mainline task as "in_progress" in Active Tasks
   - Confirm Tag/Owner routing is correct before execution
3. **Active Tasks** are MAINLINE tasks only - side issues do not belong there
4. **Blocking Side Issues** are reserved for issues that truly stop mainline progress
5. **Queued Side Issues** are non-blocking and must not take over the round
6. **After completing a mainline task**: Move it to "Completed and Verified" with evidence (but mark as "pending verification")
7. **If you discover the plan has errors**:
   - Do NOT silently change direction
   - Add entry to "Plan Evolution Log" with justification
   - Explain how the change still serves the Ultimate Goal
8. **If you need to defer a task**:
   - Move it to "Explicitly Deferred" section
   - Provide strong justification
   - Explain impact on Acceptance Criteria
9. **If you discover new issues**:
   - Add to "Blocking Side Issues" only if mainline progress is blocked
   - Otherwise add to "Queued Side Issues" or keep them as `[queued]` tasks/backlog

---

Note: You MUST NOT try to exit `start-rlcr-loop` loop by lying or edit loop state file or try to execute `cancel-rlcr-loop`

After completing the work, please:
0. If you have access to the `code-simplifier` agent, use it to review and optimize the code you just wrote
1. Finalize @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/goal-tracker.md (this is Round 0, so you are initializing it - see "Goal Tracker Setup" above)
2. Write your round contract into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/round-0-contract.md
3. Commit your changes with a descriptive commit message
4. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/round-0-summary.md
