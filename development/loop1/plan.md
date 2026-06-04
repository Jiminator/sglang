# Hill-Climb GLM-5.1-FP8 to Client SLOs Using SGLang Flags Only

## Goal Description

Find the best-achievable SGLang launch configuration — expressed purely as CLI flags and `SGLANG_*` environment variables, with **no** changes to SGLang source, model weights, the benchmark script, the workload/dataset, or the SLO definitions — that drives `zai-org/GLM-5.1-FP8` on a single node of 8× H200 (TP8) as close as possible to (or past) the client SLOs, measured with `development/benchmark.sh`.

Workload (fixed; from `development/CLIENT_SLOS.md`, baked into `development/benchmark.sh`): 4096 ISL (2253-token shared system prompt + 1843-token question) / 512 OSL, max-concurrency 64, ~55% shared-prefix cache hit, 320 prompts, fixed seed.

Client SLO targets are treated as optimization **directions** (per user decision): the deliverable is the best-achievable config plus an honest remaining-gap report; a run is additionally flagged "target met" only if both targets are crossed.
- Per-user output speed ≥ 30 tokens/s — official metric: **median ITL ≤ 33.3 ms** (client formula 1000 / ITL_ms).
- P99 TTFT < 22 s.

GLM-5.1 shares DeepSeek-V3.2's model structure (DeepSeek Sparse Attention "DSA" + MoE), so DSv3.2 tuning techniques and known incompatibilities apply. The starting point is the GLM-5.1 cookbook command (EAGLE/MTP speculative decoding, TP8, `--mem-fraction-static 0.85`). The output of this work is a reproducible hill-climb: a sweep table over launch knobs, the best config found, its measured metrics, and the quantified gap to target.

### Verified Facts (checked against this repo; ground the criteria below)
- `development/benchmark.sh` runs `sglang.bench_serving` once over 320 prompts at concurrency 64 with `--output-details`. Its JSONL summary emits, among others: `p99_ttft_ms`, `mean_ttft_ms`, `median_ttft_ms`, `mean_itl_ms`, `median_itl_ms`, `p95_itl_ms`, `p99_itl_ms`, `mean_tpot_ms`/`median_tpot_ms`/`p99_tpot_ms`, `output_throughput`, `total_throughput`, `concurrency` (Little's-Law average), `max_concurrent_requests`, `accept_length` (speculative acceptance length, SGLang only), and `completed`; `--output-details` adds per-request `ttfts`, `itls`, and `errors` arrays. There is **no** prefix-cache-hit field — cache hit is a workload property observable only via server logs.
- `--kv-cache-dtype` defaults to `auto`, which for DSA models resolves to **bf16 on H200 (SM90)** — FP8 KV cache on H200 requires an explicit `--kv-cache-dtype fp8_e4m3`, which also flips the default DSA kernels to `flashmla_kv` on Hopper.
- `--max-running-requests` defaults to auto, **but to 48 whenever a speculative algorithm is enabled** — below the workload concurrency of 64, so it must be raised to ≥ 64 when speculative decoding is on.
- `--page-size` defaults to auto (model/backend dependent). `--dsa-prefill-backend` / `--dsa-decode-backend` accept `flashmla_sparse`, `flashmla_kv`, `flashmla_auto`, `fa3`, `tilelang`, `aiter`, `trtllm` (deprecated `--nsa-*` aliases exist); on H200 the valid fast kernels are `flashmla_sparse`/`flashmla_kv`/`flashmla_auto`/`fa3` (`trtllm` is Blackwell-only, `aiter` is AMD-only). `--enable-dp-attention` requires `--dp-size` == TP size. `--schedule-policy` defaults to `fcfs` (`lpm` available). `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` defaults to 2048. `SGLANG_ENABLE_SPEC_V2` defaults to True.
- Prior empirical knowledge on this DSA+MoE structure: EP / MoE all-to-all backends (e.g. deepep), alternate MoE runner backends, `--enable-torch-compile`, NGRAM speculative decoding, and pd-multiplexing (pdmux) crash at scheduler init and are irrelevant to a fixed TP8 path — excluded a priori.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. AC-3, AC-6, and AC-7 are hard constraints; the SLO thresholds inside AC-2 are optimization directions (per the user decision), so AC-2's binding requirement is rigorous measurement, improvement over baseline, and honest gap reporting — not crossing a fixed threshold.

- AC-1: Reproducible benchmark harness, validated baseline, and admission/capacity check.
  - Positive Tests (expected to PASS):
    - The cookbook starting config — with `--max-running-requests` raised to ≥ 64 (since the speculative default of 48 caps admission) — launches cleanly on 8× H200, and `development/benchmark.sh` runs unchanged to completion with `completed == 320` and all `errors` empty.
    - Baseline metrics (`p99_ttft_ms`, `median_itl_ms`, `mean_tpot_ms`, `output_throughput`, `accept_length`, `max_concurrent_requests`) are captured to a result JSONL.
    - The server startup log is recorded showing the resolved attention backend (`dsa`), the actual KV-cache dtype, page size, and a KV/token capacity that covers ~ 64 × (4096 + 512) ≈ 294,912 concurrent tokens with margin for decode growth and speculation.
  - Negative Tests (expected to FAIL):
    - A baseline whose admitted concurrency never reaches 64 (e.g. `--max-running-requests` left at the speculative default of 48) is rejected as invalid.
    - A run with any errored request, server OOM, or scheduler-init crash is rejected.
    - A "baseline" that does not record the resolved backend / dtype / page size / KV capacity from logs is rejected.

- AC-2: Best-achievable config measured and reported against the official SLO metrics (directional).
  - Positive Tests (expected to PASS):
    - The reported best config comes from a fresh-server run of unchanged `development/benchmark.sh`, and records `median_itl_ms` (official per-user-speed metric), `p99_ttft_ms`, plus `mean_tpot_ms`, `output_throughput / 64`, `p99_itl_ms`, `accept_length`, observed `max_concurrent_requests` (≈ 64), `completed` (== 320), and empty `errors`.
    - The reported config is a measured improvement over the AC-1 baseline toward the targets.
    - If `median_itl_ms ≤ 33.3` AND `p99_ttft_ms < 22000`, the run is additionally flagged "target met".
  - Negative Tests (expected to FAIL):
    - Declaring success / "target met" without a measured benchmark JSONL is rejected.
    - Numbers taken from a server that inherited a warmed prefix cache or CUDA-graph state from a prior candidate are rejected.
    - Presenting a config that regresses versus baseline as the "winner" is rejected.
    - Reporting only `median_itl_ms` while ignoring the documented speculative-burst caveat (the `mean_tpot_ms` cross-check) is flagged.

- AC-3: Flags-only constraint (hard).
  - Positive Tests (expected to PASS):
    - `git diff` over SGLang source, the model, `development/benchmark.sh`, `development/CLIENT_SLOS.md`, and any dataset is empty.
    - The winning config is fully reproducible from environment variables plus CLI flags alone.
  - Negative Tests (expected to FAIL):
    - Any source / benchmark / dataset / SLO modification that affects measured performance fails this criterion.
    - A config that cannot be reproduced from flags + env alone fails.

- AC-4: Page-size flexibility (page size is a free knob).
  - Positive Tests (expected to PASS):
    - The sweep includes page size 64 and at least one other page size.
    - The winner uses whichever page size performs best, with no preference for 64.
    - The chosen config is demonstrated to launch and benchmark successfully at ≥ 2 distinct page sizes.
  - Negative Tests (expected to FAIL):
    - A config with no demonstrated support for varying page size (only ever launched at one hard-coded size) fails.
    - Rejecting or penalizing a faster non-64 page size purely because it is not 64 is rejected.

- AC-5: Documented hill-climb, confirmation reruns, and reproducibility metadata.
  - Positive Tests (expected to PASS):
    - A sweep table records, per candidate: the changed knob(s), `p99_ttft_ms`, median / mean / p99 `itl_ms`, `mean_tpot_ms`, `output_throughput`, `accept_length`, observed concurrency, progress-toward-target, and a one-line rationale.
    - The best config is confirmed by ≥ 2 fresh-server reruns (3 if it lands within ~ 5% of either target).
    - The report records SGLang commit, model revision, container/image, CUDA/driver, GPU clock/persistence mode, the exact launch command and env vars, the result-file paths, and the resolved KV/token capacity AFTER the final (`cuda-graph-max-bs`, `max-running-requests`, speculative, page-size, kv-dtype) combination.
  - Negative Tests (expected to FAIL):
    - A single lucky pass with no confirmation rerun is rejected as evidence.
    - A sweep table missing the changed-knob or metric columns is incomplete.
    - A winning config reported without full reproducibility metadata is rejected.

- AC-6: Out-of-scope axes excluded (hard).
  - Positive Tests (expected to PASS):
    - The winning config contains none of: EP / MoE all-to-all backends (`--moe-a2a-backend`, deepep), alternate MoE runner backends, `--enable-torch-compile`, NGRAM speculative decoding, or pd-multiplexing (pdmux).
    - Parallelism stays TP8, optionally with `--enable-dp-attention --dp-size 8`, with no expert-parallel mode.
  - Negative Tests (expected to FAIL):
    - Any winning config containing one of those axes fails.
    - Spending sweep budget actively crash-probing those axes (instead of excluding them a priori) is discouraged and flagged.

- AC-7: Accuracy-risk escalation ladder respected.
  - Positive Tests (expected to PASS):
    - Non-accuracy-affecting knobs (scheduler capacity, Hopper-valid DSA backends under the default bf16 KV path, speculative parameters, DP/TP attention, page size, schedule policy) are exhausted before any accuracy-risk knob is introduced.
    - When needed, accuracy-risk knobs are introduced in priority order FP8 KV cache → IndexCache → raised `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`.
    - Every accuracy-risk knob present in a reported config is explicitly flagged as such.
  - Negative Tests (expected to FAIL):
    - Introducing IndexCache or a raised dense-prefill threshold before FP8 KV cache, or before lower-risk knobs are exhausted, violates the ladder.
    - An accuracy-risk knob used but not flagged in the report fails.
  - AC-7.1: Capacity exception.
    - Positive: if the AC-1 capacity check shows the default bf16-KV path cannot admit concurrency 64, FP8 KV cache (`--kv-cache-dtype fp8_e4m3`) — the first rung of the ladder — may be enabled early, documented as a capacity necessity.
    - Negative: skipping straight to IndexCache or the dense-prefill threshold to "save capacity" (bypassing FP8 KV) is rejected.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A complete structured hill-climb that, following the knob-priority ladder, sweeps scheduler-capacity knobs, Hopper-valid DSA backends, speculative-decoding parameters (with speculation on vs off compared fairly at concurrency 64), DP-vs-TP attention, page size, and schedule policy under the default bf16 KV path; escalates to accuracy-risk knobs (FP8 KV → IndexCache → raised dense-prefill threshold) only when lower-risk knobs are exhausted; and produces a fully populated sweep table, the best config with ≥ 2 confirmation reruns, a quantified remaining-gap analysis per target, and complete reproducibility metadata.

### Lower Bound (Minimum Acceptable Scope)
A reproducible baseline run plus at least one fresh-server flags-only config that measurably improves over the baseline toward the targets on unchanged `development/benchmark.sh` (zero errors, concurrency 64 admitted), reported with its exact command, resolved backend / dtype / page size, result JSONL, and `median_itl_ms` / `p99_ttft_ms`.

### Allowed Choices
- Can use: any SGLang CLI flag and `SGLANG_*` env var; TP8 with optional DP attention (`--enable-dp-attention --dp-size 8`); speculative decoding on/off and parameter tuning (`--speculative-num-steps` / `--speculative-eagle-topk` / `--speculative-num-draft-tokens`, `SGLANG_ENABLE_SPEC_V2`); Hopper-valid DSA kernels (`flashmla_sparse`, `flashmla_kv`, `flashmla_auto`, `fa3`) via `--dsa-prefill-backend` / `--dsa-decode-backend`; any page size; `--schedule-policy` (including `lpm`); capacity knobs (`--max-running-requests`, `--mem-fraction-static`, `--cuda-graph-max-bs`, `--chunked-prefill-size`, token-capacity flags); accuracy-risk knobs strictly per the AC-7 ladder (FP8 KV cache, IndexCache, raised dense-prefill threshold); reasoning/tool-call parsers retained for production parity.
- Cannot use: SGLang source edits or any change to `development/benchmark.sh` / workload / dataset / SLO definitions; quality / correctness / tokenizer weakening; EP / MoE all-to-all backends; alternate MoE runner backends; `--enable-torch-compile`; NGRAM speculative; pd-multiplexing; Blackwell-only (`trtllm`) or AMD-only (`aiter`) DSA kernels on H200; cherry-picking warmed-cache epochs or reusing server state across candidates.

> **Note on Directional Design**: Because the SLOs are optimization directions (not hard gates), the binding acceptance is the *process + best-achievable config + honest gap reporting*, with the SLO thresholds as the climb objective. The upper and lower bounds describe how exhaustive the climb is, not whether a fixed threshold is crossed. The hard, non-negotiable constraints are the flags-only scope (AC-3), the out-of-scope exclusions (AC-6), the accuracy-risk ladder (AC-7), and run validity (zero errors, concurrency 64 admitted, no inherited server state).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
The workload is latency-bound at concurrency 64, and the P99 TTFT budget (22 s) is very loose relative to a single 4096-token prefill — so prefill work can be deprioritized to protect decode inter-token latency. One possible climb order:
1. Establish a reproducible baseline plus a capacity check: fresh server, `--max-running-requests` ≥ 64, and verify the resolved backend / dtype / page size and KV/token capacity from the startup log.
2. Scheduler capacity / admission: `--max-running-requests` (64 / 80 / 96), `--mem-fraction-static` (0.85 → 0.9 for more KV headroom), `--cuda-graph-max-bs` (≥ max running requests so the decode batch is graph-captured), `--chunked-prefill-size` (smaller chunks protect decode latency given the loose TTFT budget).
3. Hopper-valid DSA backends for prefill and decode separately, under the bf16 KV path first.
4. Speculative decoding: tune `--speculative-num-steps` / `--speculative-eagle-topk` / `--speculative-num-draft-tokens`, watch `accept_length`, and compare speculation on vs off at concurrency 64 (fair only with `--max-running-requests` ≥ 64). At high concurrency the verification overhead may outweigh speculation's benefit — measure both.
5. DP vs TP attention: keep TP as the low-latency reference, try `--enable-dp-attention --dp-size 8` for high-concurrency throughput.
6. `--page-size` sweep (including 64 and ≥ 1 other) and `--schedule-policy lpm` to exploit the shared prefix.
7. Accuracy-risk ladder (only if lower-risk knobs are exhausted, per AC-7): FP8 KV (`--kv-cache-dtype fp8_e4m3` — ~ 2× KV capacity, flips DSA kernels to `flashmla_kv` on H200) → IndexCache (`--json-model-override-args '{"index_topk_pattern": "..."}'`) → raised `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`.

Methodology: fresh server per candidate and per confirmation rerun; run `development/benchmark.sh` unchanged (it already passes `--output-details`); take the gate numbers from the exact `benchmark.sh` command, using the `--output-details` arrays only as diagnostics (e.g. inspecting the TTFT distribution, where the first cold wave dominates P99). Never gate on prefix-cache-hit rate — no benchmark field exists; observe it only via server logs as supporting evidence. Verify each candidate's resolved attention/DSA backend, KV-cache dtype, and page size from the server logs rather than assuming them from the flags.

### Relevant References
- `development/benchmark.sh` — the fixed benchmark harness (do not modify).
- `development/CLIENT_SLOS.md` — workload and SLO definitions (do not modify).
- `.claude/skills/sglang-sota-performance/SKILL.md` — methodology; use its benchmarking/comparison portions only (its code-patching steps are out of scope here).
- `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` — GLM-5.1 deployment and tuning tips (DP attention, MTP, mem-fraction).
- `docs/basic_usage/deepseek_v32.md` — DSA backend/kernel knobs, DP/TP attention, MTP, IndexCache (GLM-5.1 shares this structure).
- `docs_new/docs/advanced_features/hyperparameter_tuning.mdx` — general tuning guidance.
- `python/sglang/bench_serving.py` — metric definitions and JSONL schema.
- `python/sglang/srt/server_args.py`, `python/sglang/srt/environ.py` — flag and env-var names and defaults.

## Dependencies and Sequence

### Milestones
1. Harness, baseline, and capacity validation (targets AC-1, AC-3).
   - Phase A: fresh-server cookbook launch with `--max-running-requests` ≥ 64.
   - Phase B: run unchanged benchmark, capture baseline metrics, verify resolved backend / dtype / page size and KV capacity from logs.
2. Capacity / admission tuning (targets AC-2). Depends on Milestone 1.
   - Sweep `--max-running-requests`, `--mem-fraction-static`, `--cuda-graph-max-bs`, `--chunked-prefill-size`.
3. DSA-backend tuning under bf16 KV (targets AC-2). Depends on Milestone 2.
   - Sweep Hopper-valid prefill/decode kernels.
4. Speculative-decoding and DP/TP-attention tuning (targets AC-2). Depends on Milestone 3.
   - Tune speculative parameters; compare speculation on/off and DP vs TP attention at concurrency 64.
5. Page-size and schedule-policy sweep (targets AC-4, AC-2). Depends on Milestone 4.
6. Accuracy-risk escalation, only if needed (targets AC-7, AC-2). Depends on Milestone 5.
   - FP8 KV → IndexCache → raised dense-prefill threshold (or FP8 KV early if the Milestone 1 capacity check forces it).
7. Confirmation and final report (targets AC-5, AC-6). Depends on Milestone 6.
   - ≥ 2 fresh reruns of the best config; sweep table; gap analysis; reproducibility metadata; confirm out-of-scope axes are absent.

Dependencies are relative: each tuning milestone builds on the previously discovered best config, and the accuracy-risk milestone runs only after the lower-risk milestones are exhausted (except the AC-7.1 capacity exception).

## Task Breakdown

Each task includes exactly one routing tag:
- `coding`: implemented by Claude (operationally: launching servers, running the benchmark, recording results, writing the report)
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Stand up fresh-server cookbook baseline with `--max-running-requests` ≥ 64; run unchanged `benchmark.sh`; capture baseline JSONL plus server-log resolved backend / dtype / page-size / KV-capacity | AC-1, AC-3 | coding | - |
| task2 | Validate KV/token capacity covers ~ 64 × 4608 tokens; if short, sweep `--mem-fraction-static` / token-capacity flags | AC-1 | coding | task1 |
| task3 | Sweep scheduler-capacity knobs (`--max-running-requests`, `--cuda-graph-max-bs`, `--chunked-prefill-size`), fresh server each | AC-2 | coding | task2 |
| task4 | Sweep Hopper-valid DSA prefill/decode backends under bf16 KV | AC-2 | coding | task3 |
| task5 | Tune speculative parameters and compare speculation on/off and DP-vs-TP attention at concurrency 64 | AC-2 | coding | task4 |
| task6 | Sweep page size (including 64 and ≥ 1 other) and `--schedule-policy lpm`; confirm the config supports varying page sizes | AC-4, AC-2 | coding | task5 |
| task7 | If lower-risk knobs are exhausted without meeting targets, escalate the accuracy-risk ladder FP8 KV → IndexCache → dense-prefill threshold, flagging each | AC-7, AC-2 | coding | task6 |
| task8 | Analyze the running sweep table to identify the binding bottleneck and recommend the next highest-value knob each round | AC-2, AC-5 | analyze | task3 |
| task9 | Run ≥ 2 fresh confirmation reruns of the best config (3 if within ~ 5% of a target); record resolved KV capacity after the final combination | AC-5 | coding | task7 |
| task10 | Produce the final report: sweep table, best command, gap-to-target analysis, full reproducibility metadata; confirm out-of-scope axes absent | AC-5, AC-6 | coding | task9 |

## Claude-Codex Deliberation

### Agreements
- `--max-running-requests` must be ≥ 64 — the speculative default of 48 caps admission below the workload concurrency.
- Flags-only; benchmark / dataset / SLO untouched; fresh server per candidate and per confirmation; no inherited cache or CUDA-graph state.
- Per-user speed under speculative decoding needs robust measurement; report `mean_tpot_ms` and `output_throughput / 64` alongside the chosen scalar.
- The DSA-backend sweep is limited to Hopper-valid kernels; `trtllm` (Blackwell) and `aiter` (AMD) are excluded on H200.
- EP / a2a MoE backends, alternate MoE runners, torch-compile, NGRAM, and pdmux are out of scope for TP8 and/or known to crash on the DSA+MoE structure.
- KV/token capacity must be validated from server logs (~ 64 × 4608 tokens), not assumed; the best config needs ≥ 2 confirmation reruns.

### Resolved Disagreements
- Per-user speed metric: Codex argued median ITL is gameable under speculative bursts and pushed `mean_tpot_ms` / throughput-÷-64; Claude initially proposed `mean_tpot_ms` as primary. Resolution: the user selected `median_itl_ms ≤ 33.3` as the official scalar (client-faithful 1000/ITL), with `mean_tpot_ms` and `output_throughput / 64` retained as documented robustness cross-checks. Rationale: honor the client's literal formula while surfacing the speculative-burst caveat.
- Page-size-64 as acceptance: Codex flagged "winning config must be page-size 64" as over-constraining; Claude relaxed it to a swept knob. Resolution: the user chose "page size is free" — the winner uses the best size; the config must still demonstrate ≥ 2 page sizes launch.
- Throughput denominator: resolved to divide by 64 (the target concurrency) and additionally require observed `max_concurrent_requests` to reach ≈ 64.

### Convergence Status
- Final Status: `converged` (2 convergence rounds; round-2 REQUIRED_CHANGES: none).

## Pending User Decisions

All decisions are resolved; none remain `PENDING`.

- DEC-1: Per-user-speed scalar for the 30 TPS check.
  - Claude Position: `mean_tpot_ms` as primary (robust to speculative bursts), report median ITL.
  - Codex Position: median ITL alone is insufficient under speculation; require `mean_tpot_ms` and/or `output_throughput / 64`.
  - Tradeoff Summary: median ITL is the client's literal 1000/ITL formula but can look optimistic when speculation emits token bursts; TPOT/throughput are robust but not the client's stated formula.
  - Decision Status: **median ITL ≤ 33.3 ms is the official metric**; `mean_tpot_ms` and `output_throughput / 64` are reported as cross-checks, with the speculative-burst caveat documented.
- DEC-2: Which accuracy-risk knobs may be used.
  - Claude Position: FP8 KV is on the table per the draft; treat IndexCache and the raised dense-prefill threshold as accuracy-risk extras.
  - Codex Position: FP8 KV, IndexCache, and the dense-prefill threshold all affect accuracy on a latency-only benchmark; gate them on explicit user acceptance.
  - Tradeoff Summary: these knobs can unlock speed/capacity but carry possible quality loss the benchmark cannot detect.
  - Decision Status: **All allowed, but only after exhausting other options, in priority order FP8 KV → IndexCache → raised dense-prefill threshold** (encoded as the AC-7 ladder).
- DEC-3: Page-size-64 requirement.
  - Claude Position: prefer 64, allow an alternate if 64 cannot meet the target.
  - Codex Position: page size 64 is a valid knob, not a valid acceptance criterion unless the client requires it.
  - Tradeoff Summary: the draft "significantly prefers" 64 but also wants support for varying page sizes.
  - Decision Status: **Page size is free** — the winner uses the best-performing size; the config must still demonstrate ≥ 2 page sizes launch.
- DEC-4: Keep reasoning/tool-call parsers in the launch command.
  - Claude Position: keep `--reasoning-parser glm45 --tool-call-parser glm47` for production parity (negligible benchmark-latency impact; the workload has no tool calls).
  - Codex Position: parsers can be dropped for benchmark-only latency, but should match production if parity matters.
  - Tradeoff Summary: parity vs marginal simplicity; no measurable benchmark cost.
  - Decision Status: **Keep parsers** (resolved by default; user may override).
- DEC-5: Number of confirmation reruns.
  - Claude Position / Codex Position: at least two fresh reruns; three if within ~ 5% of a target.
  - Tradeoff Summary: more reruns increase confidence at higher cost; noise near a threshold needs more samples.
  - Decision Status: **≥ 2 fresh reruns, 3 if within ~ 5% of a target** (resolved during convergence).
- DEC-6: Whether the SLOs are hard gates or directions.
  - Claude Position: default to hard gates unless the client says otherwise.
  - Codex Position: treat SLOs as hard gates unless the client explicitly says otherwise.
  - Tradeoff Summary: hard gates give a crisp pass/fail; directions allow reporting the best-achievable config when the target is not fully reached.
  - Decision Status: **Optimization directions** — deliver the best-achievable config plus a quantified gap report; flag "target met" only if both thresholds are crossed.

## Implementation Notes

### Code Style Requirements
- Implementation artifacts and comments (any helper scripts, config snippets, or notes produced) must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting artifacts.
- Use descriptive, domain-appropriate naming instead (e.g. `baseline`, `capacity_sweep`, `dsa_backend`, `confirm_run`).

### Operational / Reproducibility Notes
- This is a tuning / operations task: the intended output is a set of launch commands, benchmark result files, a sweep table, and a final report — not SGLang source changes.
- For every run, capture the resolved attention/DSA backend, KV-cache dtype, page size, and KV/token capacity from the server startup log; record the SGLang commit and model revision; keep the benchmark result JSONLs.
- Take gate numbers from the exact unchanged `development/benchmark.sh` command; use `--output-details` arrays as diagnostics only.
- Median-ITL caveat: under speculative decoding, report `mean_tpot_ms` alongside `median_itl_ms` so the official metric is not read in isolation.
- Never gate on prefix-cache-hit rate (no benchmark field exists); use server logs only as supporting evidence.

--- Original Design Draft Start ---

Task: Hillclimb GLM 5.1 FP8 on a specific workload to meet target using only sglang flags.

Workload and Target: development/CLIENT_SLOS.md

Benchmark Script: development/benchmark.sh
Out-of-scope: Code changes that affect sglang performance, we are testing out-of-box performance

Revelant Skills: .claude/skills/sglang-sota-performance

Starting Point (cookbook):
```
SGLANG_ENABLE_SPEC_V2=1 sglang serve \
  --model-path zai-org/GLM-5.1-FP8 \
  --tp 8 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.85
```


Relevant and Useful Sources:
- docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx 
- https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.1
- docs/basic_usage/deepseek_v32.md
- docs_new/docs/advanced_features/
- docs_new/docs/advanced_features/hyperparameter_tuning
- https://sgl-project-sglang-93.mintlify.app/optimization/performance-tuning

Notes:
- Assume Fp8 kv cache is on the table.
--- Original Design Draft End ---
