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
