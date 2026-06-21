# Ask Codex Input

## Question

You are doing a FIRST-PASS planning critique for a curation/clean-port task in the SGLang repository. Output is consumed by an automated planning pipeline — be concrete and terse.

## Repository context
- This is SGLang (LLM serving). You are running inside the DEV clone at /sgl-workspace/sglang on branch `dev/double-sparsity-standalone`. You may read any file here.
- A validated "table-free Double Sparsity (DS)" feature was developed on this branch over loops 11/11b on GLM-5.1-FP8 (8xH200). The algorithm is DONE and validated. This task does NOT change DS behavior.
- The DS runtime package is `python/sglang/srt/layers/attention/double_sparsity/` (you can `ls` it: it contains both runtime files AND dev-only oracle/capture/validation scaffolding mixed together).
- A SECOND clean clone exists at `/sgl-workspace/double-sparisty-v2/sglang` on clean `main` (~105e095e0), with ZERO DS references today. All shipping changes happen THERE.
- Goal of Loop 12: extract the MINIMAL correct DS runtime from the dev branch onto a fresh branch off the v2 clone's `main`, plus ONE simple perf eval, and prove parity (no regression vs loop-11b numbers: DS ~26.9 TPS / ~25.1s P99 TTFT at concurrency 64). "Performant" = parity/regression-gate, NOT a new 30-TPS SLO.
- Key method: ADDITIVE MINIMAL CLOSURE, not a merge/rebase. Start from blank `main`, copy new DS files, re-apply DS hunks onto main's CURRENT version of modified files (main has drifted), and a file ships ONLY if removing it breaks `import sglang`, breaks server boot with DS enabled, or breaks the conc-64 perf run.

## The raw draft (authoritative human intent)
<<<DRAFT
# Loop 12 Draft — Cut the clean Double Sparsity shipping branch

This is a curation + clean-port loop, not new development. The algorithm is done and validated. We extract the minimal correct runtime from a messy dev branch onto a clean base, and prove the extraction preserved behavior and performance. "Performant and shippable" = parity (no regression vs loop-11b numbers), not a new SLO. The validated candidate lands ~26.9 TPS at conc 64 (DSA also fails the 30-TPS floor there); Loop 12 does NOT try to fix that. A clean, smaller diff is the deliverable: every file/symbol that lands must be reachable from the DS serving path or the one perf script.

OBJECTIVE: Produce `double-sparsity-v2` (or next free name) on `Jiminator/sglang`, branched from current `main`, containing: (1) minimal DS runtime to serve table-free DS on GLM-5.1-FP8 with TP + CUDA graphs + radix cache; (2) ONE perf eval script mirroring the conc-64 workload (4096 ISL / 512 OSL, gsp ~55% prefix) emitting decode TPS + P99 TTFT; (3) proof it works — server boots with DS enabled, DS genuinely active (not silent dense fallback), perf eval reproduces loop-11b conc-64 numbers within noise.

EXCLUDED from branch: `.pensieve/`, `.humanize/`, `development/` (whole tree), `SLOS.md`, every loop log, all DS dev-only scaffolding (oracles, capture sinks, recall/validator harnesses, calibration sweeps, the AC-11 comparator + evidence gates, manual test_dsv32_* fixtures).

WORKING MODEL — two clones: THIS dev clone (/sgl-workspace/sglang, branch dev/double-sparsity-standalone) is the SOURCE we copy DS code FROM and holds loop machinery + development/loop12/ — NEVER committed to shipping branch. The v2 clone (/sgl-workspace/double-sparisty-v2/sglang, origin Jiminator/sglang, clean main) is where ALL shipping changes/new branch/serving/perf runs happen. Reviewer diffs v2 branch against its base (main), not this dev branch.

BRANCH SETUP (in v2 clone): git fetch origin; git switch -c double-sparsity-v2 origin/main. If name exists on fork, fall back to a free name (verify via git ls-remote --heads origin). Push to Jiminator/sglang (the fork) once real and owner-authorized. NEVER push to public upstream sgl-project/sglang.

SCOPE — starting hypothesis (NOT authoritative; the minimal-closure test is authoritative):
NEW RUNTIME FILES — copy wholesale (IN, pending closure): double_sparsity/{__init__.py, config.py, absorbed_latent.py, absorbed_latent_kernel.py, selection_kernel.py, topk_kernel.py, selector.py, channel_mask.py, cuda_graph.py, page_table_adapter.py, lifted_budget.py, error_containment.py}. metrics.py: DECISION — ships only if "DS is active" signal kept.
NEW FILES — DROP (dev-only): oracle_artifact_sink.py, selection_recall_oracle.py, radix_fixture_capture.py, score_capture.py, selection_capture.py, latent_capture.py, validator.py, and calibrate.py (DECISION — lean runtime-only). All test/manual/test_dsv32_*, _dsv32_quality_smoke_lib.py, _m3b_label_capture_verdict.py, test/registered/unit/development/* (AC-11 comparator, bench-meta writer, option-b scripts), and oracle/recall/ac12/label-capture/accuracy-gate-compare tests.
MODIFIED UPSTREAM FILES — re-apply DS hunks onto current main (IN, pending closure): dsa_backend.py; server_args.py (--enable-double-sparsity + related knobs); model_executor/{model_runner.py, model_runner_kv_cache_mixin.py, cuda_graph_runner.py, pool_configurator.py}; models/deepseek_v2.py + models/deepseek_common/attention_forward_methods/{forward_mla.py, forward_mha.py} (GLM-5.1 glm4_moe reuses DeepSeek MLA path); mem_cache/{memory_pool.py, memory_pool_host.py}; managers/{schedule_batch.py, scheduler.py, io_struct.py, tokenizer_manager.py, scheduler_components/batch_result_processor.py}. CARRY FORWARD loop-11b fix: DS abort path must call req.update_finish_state() (NOT pre-#25725 check_finished) — verify vs main's current finisher API.
TESTS — keep only feature tests: test_double_sparsity_unit.py, test_lifted_budget_decode.py (keep only if lifted_budget ships and test needs no oracle fixtures). Drop everything importing an oracle/capture/comparator. DECISION — decide test floor.
bench_serving.py — stays STOCK (out of diff). The perf eval uses stock main bench_serving for client-visible TPS/TTFT.

PORT STRATEGY — additive minimal closure (NOT merge/rebase): start from blank main; copy new files; re-apply each DS hunk onto main's current file version reconciling drift by hand; closure check = the gate (after each addition in v2: python -c "import sglang" clean -> server boots with DS enabled -> conc-64 perf run produces numbers; any file not required does not ship; log what's dropped); dead-code sweep before done (grep every shipped DS module/symbol for a live reference from serving path or perf script; unreferenced -> removed).

THE ONE PERF EVAL: single script in v2 (benchmarks/ or top-level bench_double_sparsity.sh — read as product tooling). Mirror conc-64 exactly: --dataset-name generated-shared-prefix, --gsp-system-prompt-len 2253 --gsp-question-len 1843 (ISL 4096, ~55% prefix), --gsp-output-len 512, --gsp-range-ratio 1.0, --max-concurrency 64, ONE trial, --backend sglang. Emit decode TPS (p50) + P99 TTFT. Dead simple — one model, one concurrency, one trial. DECISION (perf-eval fidelity): stock main bench_serving lacks --warmup-seconds/--measurement-window-seconds steady-state flags. (A, recommended) stock/simple: plain --num-prompts N --max-concurrency 64, no window flags, wider noise band. (B) faithful: port only the small self-contained window-measurement flags into branch's bench_serving for exact steady-state TPS, slightly larger diff. Parity target: DS ~26.9 TPS / ~25.1s P99 TTFT within noise — PASS = matches validated candidate, NOT >=30 TPS.

ROUGH ACCEPTANCE CRITERIA: (1) branch exists on fork branched from current main; git diff main...branch touches ONLY DS runtime + one perf script + minimal feature tests. (2) exclusions verified ABSENT (no .pensieve/.humanize/development/SLOS.md/oracle/capture/calibration/comparator/test_dsv32_*/test/registered/unit/development/*). (3) python -c "import sglang" clean; DS feature tests pass. (4) server boots in v2 with DS enabled on GLM-5.1-FP8 (8xH200, dsa backend, glm4_moe), TP + CUDA graphs + radix cache on; DS genuinely active (not silent dense fallback). (5) perf eval runs conc-64 one trial, reports decode TPS + P99 TTFT within noise of loop-11b (~26.9 TPS / ~25.1s) — regression gate. (6) no dead code — every shipped DS module/symbol referenced from serving path or perf script.

CONSTRAINTS (hard-won, do not relitigate): NEVER set PYTORCH_CUDA_ALLOC_CONF=expandable_segments for serving. ONE TP=8 server at a time (tear down before next, wait GPU idle). Do NOT run blanket nvidia-smi GPU PID kills; do NOT pkill -f a pattern matching parent shell. All serving/perf in v2 clone; loop machinery + draft stay in this clone. Push only to fork, owner-authorized, never public upstream. Perf eval needs GLM-5.1-FP8 weights + calibrated channel mask present at run time.

OPEN DECISIONS: (DEC calibration tooling) ship calibrate.py or runtime-only? Lean runtime-only for v2. (DEC "DS is active" signal) keep lightweight runtime signal (metrics.py no-op fields or startup log line) or drop all metric plumbing? Lean keep minimal cheap host-side signal. (DEC perf-eval fidelity) A vs B. Lean A. (DEC channel mask provenance) where does the GLM-5.1 mask the eval uses come from on a clean branch (committed artifact? documented external path? regenerated via calibrate)? (DEC test floor) minimum feature-test set guarding the runtime without oracle fixtures.
DRAFT>>>

## Your task
Inspect the actual DS package and the modified upstream files in THIS clone where useful (you have repo access). Then critique the plan. Be concrete: name files, name risks the draft under-weights. Output STRICTLY in this format with these exact section headers:

CORE_RISKS:
- highest-risk assumptions and failure modes (e.g. closure-check feasibility on shared GPUs, drift reconciliation hazards, "DS genuinely active" detection method, mask provenance blocking the eval)

MISSING_REQUIREMENTS:
- likely omitted requirements or edge cases (e.g. dependency-graph completeness for the closure check, how to PROVE DS active vs dense fallback, how parity "within noise" is quantified, what guards the re-applied abort-path fix)

TECHNICAL_GAPS:
- feasibility/architecture gaps (e.g. hand-reconciling DS hunks across drifted files, ordering of closure additions, sgl-kernel/build deps for the DS kernels)

ALTERNATIVE_DIRECTIONS:
- viable alternatives with tradeoffs (e.g. squash-export vs hand-port, automated import-closure tool vs manual, single perf script vs pytest perf gate)

QUESTIONS_FOR_USER:
- questions needing explicit human decisions (tie back to the 5 open DECs; add any new ones)

CANDIDATE_CRITERIA:
- candidate acceptance criteria phrased for deterministic verification (positive + negative test ideas)

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-18_10-51-37
- Tool: codex
