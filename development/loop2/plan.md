# GLM-5.1-FP8 Profile-Driven Flags-Only Hill-Climb (Loop 2)

## Goal Description

Drive `zai-org/GLM-5.1-FP8` (an MLA + DeepSeek-Sparse-Attention / DSA, 256-expert MoE model) on one node of 8× H200 (TP8, CUDA) as close as possible to the rebased client SLO, using **only** SGLang `sglang serve` CLI flags and `SGLANG_*` environment variables — **no source, kernel, test, workload, or SLO edits that affect SGLang performance**. All gate numbers come from the fixed, unmodified `development/benchmark.sh`.

Unlike Loop 1 (which inferred its bottleneck from benchmark deltas), Loop 2 is **profile-driven**: every candidate is accompanied by a decode-phase torch-profiler trace and a bottleneck analysis, and the profile — not blind sweeping — decides the next knob. The central deliverable is an explicit, profiler-grounded answer to: **is the ~24–27 TPS decode ceiling hard MoE-GEMM compute (which would confirm expert parallelism is required, and that is out of scope here), or is there any flags-only overlap / fusion / scheduling / attention-kernel headroom left?**

### Official acceptance metric (rebased from Loop 1 — read first)

The official per-user-speed metric is the client's verbatim TPS formula applied to run totals:

```
TPS = total_output_tokens / (total_latency − TTFT) = Σ output_tokens / Σ (latency − ttft) ≈ 1000 / mean_tpot_ms
```

- **Target: client TPS ≥ 30 per user, AND P99 TTFT < 22 s.**
- `median ITL` / `1000/ITL` is a **speculation-inflated cross-check ONLY** (EAGLE/MTP bursts inflate it ~2.3×); it is **never** the official verdict.
- Page size 64 is **not** a requirement (DSA pins the effective page size to 64 on CUDA regardless of `--page-size`).
- FP8 KV cache is fully permitted (use freely if it helps; Loop 1 found it regresses, to be re-confirmed by profile).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive tests (expected to PASS when the criterion is met) and negative tests (expected to FAIL / be rejected when the work is done correctly). "Tests" here are deterministic checks against the produced artifacts (benchmark JSONL/logs, profiler traces, analysis tables, the final report, and `git diff`), since the deliverable is an evidence-backed tuning study, not a code feature.

- AC-1: The official **ranking/selection** metric is the client TPS formula `Σ output_tokens / Σ (latency − ttft)`, with target `client TPS ≥ 30`. `P99 TTFT < 22 s` is a documented SLO target that is **reported per candidate but is NOT a disqualifier** for loop-2 selection (owner decision DEC-2 — report-only). `median ITL` is reported only as a speculation-inflated cross-check. The TPS metric definition itself is fixed and is not softened; loop success is "best-achievable + a defensible profiler-grounded ceiling verdict" rather than a strict 30-TPS pass/fail (owner decision DEC-1).
  - Positive Tests (expected to PASS):
    - The result parser computes per-run client TPS as `Σ output_tokens / Σ (latency − ttft)` and the ranking column in the sweep table is that TPS.
    - Every reported candidate row carries P99 TTFT (ms) with an explicit (informational) note of whether it is below 22 000 ms — recorded, but not used to drop the candidate from selection.
    - `median ITL` (and any `1000/ITL`) appears in a column explicitly labeled "cross-check (speculation-inflated)".
  - Negative Tests (expected to FAIL):
    - A verdict that declares "best per-user speed" on the basis of `1000/median_ITL` is rejected.
    - A candidate ranked as "best" using mean-of-per-request-rate (token_i/decode_i) instead of the Σ/Σ total-ratio is rejected.
    - A candidate that omits its P99 TTFT value entirely (it must still be reported, even though it does not disqualify) is rejected.

- AC-2: Gate (official) measurement integrity is preserved: fresh server per candidate, unchanged `development/benchmark.sh`, flags-only server launch, and the gate run is **not** profiled.
  - Positive Tests:
    - Each gate candidate is launched on a fresh server (teardown → launch → readiness wait → unchanged `development/benchmark.sh` → parse → GPU-mem drain), with the resolved server flags/env logged.
    - Gate runs report `completed == 320`, `errors == 0`, and observed max concurrency ≈ 64.
    - The gate benchmark command is byte-for-byte the unmodified `development/benchmark.sh`; `git diff` on `development/benchmark.sh` is empty.
  - Negative Tests:
    - A gate TPS number captured on a server that was simultaneously running the torch profiler is rejected (profiler overhead perturbs the gated metric).
    - A gate number from a server that reused warmed state / prefix cache from a previous candidate (not a fresh server) is rejected.
    - A candidate with `completed < 320` or `errors > 0` is not eligible to be the recommended config.
    - A number from a profile-only diagnostic run used as the official/scoring metric is rejected — profile-only runs are **non-scoring** and can never substitute for a gate run, even when they replay the identical workload.
  - AC-2.1: Finalist configs are confirmed for run-to-run stability.
    - Positive: each config promoted to "recommended" (safe and best-achievable) has its gate run repeated 2–3 times on fresh servers, with the reported client TPS / P99 TTFT being the confirmed value (and observed variance noted).
    - Negative: a single-run number promoted to "recommended winner" with no repeat is flagged as unconfirmed.

- AC-3: Each evaluated candidate has a decode-phase torch-profiler trace plus a bottleneck analysis, captured on a **profile-only diagnostic run** (separate from the gate run) that replays the same generated-shared-prefix workload at concurrency 64 with identical server flags.
  - AC-3.1: The EAGLE decode loop is attributed correctly — the "decode step" groups `ForwardMode.DECODE + TARGET_VERIFY + DRAFT_EXTEND(_V2)`, not bare `DECODE`.
    - Positive: the analysis explicitly states it does not rely on `--profile-by-stage`'s decode bucket alone (which classifies `TARGET_VERIFY` as prefill via `is_extend()`, `forward_batch_info.py:109-118`), and instead groups the verify/draft forwards into the decode loop (e.g. by post-filtering step spans or capturing a steady-state window).
    - Negative: an analysis that reports a "decode" share using only `ForwardMode.DECODE` events (omitting TARGET_VERIFY / DRAFT_EXTEND) under an EAGLE config is rejected.
  - AC-3.2: Each bottleneck analysis produces: (a) kernel time breakdown by category — MoE GEMMs vs MLA/DSA attention + DSA indexer vs all-reduce/comms vs sampling/draft-model/EAGLE-verify vs other; (b) top-N kernels by total time; (c) overlap opportunities (idle/exposed gaps); (d) fuse-pattern candidates.
    - Positive: each profiled candidate records, alongside its benchmark row, the top-3 kernels by time, the dominant bottleneck category as a % of the decode step, and any overlap/fusion/scheduling headroom observed. Categories are produced via an explicit kernel→category map (not raw kernel names alone).
    - Positive: the analysis reports BOTH summed-kernel-time share AND exposed (non-overlapped, critical-path) wall-time share, so overlapped kernels are not double-counted as the bottleneck.
    - Negative: a "bottleneck = X" claim with no kernel table, no category map, and no exposed-vs-summed distinction is rejected as hand-wavy.
  - AC-3.4: Profiler artifact hygiene (owner decision DEC-4): the extracted insights are persisted, the raw traces are not.
    - Positive: for each profiled candidate, the extracted insights (kernel category table, top-N kernels, summed vs exposed shares, overlap/fuse notes, batch/rank evidence) are written to a per-candidate markdown file under a dedicated profiling-analysis directory (e.g. `development/loop2/profiling/`); after extraction, the raw profiling artifacts (trace JSON/`.gz`, Nsight `.nsys-rep`, etc.) are deleted to save disk.
    - Positive: the captured tool is recorded; if NVTX-marker or Nsight captures fail or are unavailable, the run is not blocked — the torch profiler trace is the required floor and the analysis proceeds on it alone.
    - Negative: leaving large raw trace artifacts behind after insights are extracted, or blocking a candidate because an optional diagnostic tool (Nsight/NVTX) failed while the torch profiler succeeded, is rejected.
  - AC-3.3: Captured traces are representative of steady-state concurrency-64 decode.
    - Positive: trace evidence shows steady-state batch size ≈ 64 (or the analysis justifies the observed batch), the warmup/cold-prefill window is excluded (via `start_step`), and the captured rank(s) are recorded (all TP ranks, or one rank with justification).
    - Positive: when more than one TP rank is captured, per-rank kernel shares are normalized per rank and any rank imbalance (e.g. MoE/routing skew) is noted separately from the aggregate decode-loop attribution.
    - Negative: a trace dominated by cold-start prefill, or a single synthetic short-in/long-out probe used as a stand-in for the conc-64 shared-prefix workload, is rejected as the decode-phase evidence for a candidate.

- AC-4: The DSA `--dsa-prefill-backend` × `--dsa-decode-backend` cross-product over the Hopper-relevant kernels {`flashmla_sparse`, `flashmla_kv`, `flashmla_auto`, `fa3`} under bf16 is attempted, with each cell's outcome recorded under a deterministic failure taxonomy and exact source/log citation. The matrix is exhausted (every launchable cell fully measured), not pruned (owner decision DEC-3).
  - AC-4.1: Each attempted cell is classified as exactly one of: `parser-reject` / `startup-reject` / `readiness-timeout` / `first-request runtime failure` / `full-benchmark result (TPS, accept, p99 TTFT)`, with the exact log line or source `path:line` recorded.
    - Positive: `--dsa-decode-backend flashmla_auto` is attempted and recorded as a runtime failure citing the decode dispatch assertion (`python/sglang/srt/layers/attention/dsa_backend.py:1726`, `assert False, Unsupported self.dsa_decode_impl`), since decode has no `auto` branch (auto is prefill-only, `dsa_backend.py:2273-2289`).
    - Positive: `flashmla_kv` (decode and/or prefill) under bf16 is actually **attempted** (not skipped on theory) and recorded as a launchable full-benchmark result; the analysis notes it internally quantizes the whole cache (`dsa_backend.py:1846-1848`), explaining any regression.
    - Negative: any matrix cell marked "skipped (theory)" with no launch attempt and no source/log citation is rejected (no silent skips).
  - AC-4.2: The matrix is exhausted, not pruned (owner decision DEC-3 — no pruning).
    - Positive: **every** matrix cell is launch-attempted and recorded under the AC-4.1 taxonomy; every cell that launches successfully is fully gate-benchmarked AND profiled, regardless of the profiled attention/indexer share.
    - Positive: each launchable cell's delta vs the incumbent is attributed to a specific kernel change using its profile (the profile explains the delta; it does not gate whether the cell is run).
    - Negative: skipping the gate-benchmark or profile of any launchable cell on "attention is a small fraction" grounds is rejected — pruning is disabled for this loop.
    - Negative: skipping a cell's launch attempt / failure-taxonomy record is rejected (no silent skips).

- AC-5: The EAGLE tree axis (`--speculative-eagle-topk > 1`) is attempted and its outcome recorded with source citation; the EAGLE-verify share of the decode loop is quantified from a profile to characterize the incumbent chain (topk=1) config.
  - Positive: `--speculative-eagle-topk 2` (and/or 4) with matched `--speculative-num-draft-tokens`/`--speculative-num-steps` is attempted and recorded as a deterministic launch-reject citing the validation `ValueError` (`python/sglang/srt/arg_groups/speculative_hook.py:383-390`: rejected because DSA forces `attention_backend=dsa` + `page_size=64`, and topk>1 with page_size>1 is only allowed for `flashinfer`/`fa3`).
  - Positive: a decode-phase profile of the incumbent topk=1 config reports the EAGLE-verify (TARGET_VERIFY) + draft-model share of the decode loop, so the "bigger verify batches cost more than they buy" prior is confirmed/denied with profiler evidence.
  - Negative: declaring the speculative-tree axis "exhausted" with no launch attempt and no `path:line` citation is rejected; and concluding tree speculation "wouldn't help" purely from a benchmark guess (without the verify-share profile) is rejected.

- AC-6: The central question is answered explicitly with profiler evidence: hard MoE-GEMM compute ceiling vs flags-only headroom.
  - Positive: the report states a verdict and supports "hard MoE compute" only if (a) MoE GEMMs dominate the exposed decode/verify wall-time, (b) MLA/DSA attention + indexer and sampling/verify overhead are small slices, and (c) there is no material idle/comm/scheduling gap addressable by an allowed flag.
  - Positive: if any allowed-flag headroom exists (e.g. exposed comm/idle gaps, a missed overlap/fusion/scheduling path), it is named with the profiler row that supports it and tested as a candidate.
  - Negative: a "ceiling is MoE compute" conclusion asserted from benchmark deltas alone, with no exposed-time category breakdown, is rejected; a conclusion that contradicts its own kernel table is rejected.

- AC-7: No silent skips — coverage is evidence-bounded.
  - Positive: a finite candidate backlog is enumerated; every launchable DSA matrix cell and every in-scope axis is measured (gate + profile), and any axis that is infeasible flags-only is closed with a cited launch/runtime outcome (`path:line` or log line).
  - Positive: any *profile-directed follow-up* knob that is NOT pursued (e.g. a DSA env/threshold probe) is skipped only with a cited profiler-grounded justification (the relevant category was shown immaterial in the profile) — never by assertion.
  - Negative: a "covered all remaining gaps" claim backed only by assertion (no profile/log/source citations) is rejected.

- AC-8: The hard scope constraints hold throughout.
  - Positive: `git diff` over `python/`, `sgl-kernel/`, `test/`, and `development/benchmark.sh` since the Loop 2 setup commit is empty for performance-affecting source; parallelism stays TP8 (`tp_size=8, ep_size=1, dp_size=1, moe_a2a_backend=none`), verified from resolved server args.
  - Positive: no candidate uses EP / MoE a2a (`--moe-a2a-backend`, deepep), alternate MoE runners, `--enable-torch-compile`, NGRAM speculative, pd-multiplexing, or Blackwell-only (trtllm) / AMD-only (aiter) DSA kernels.
  - Negative: any candidate enabling an out-of-scope axis, or any performance-affecting edit to SGLang source / kernels / tests / the benchmark harness, is rejected.

- AC-9: A final report consolidates results with reproducibility metadata and recommended config(s).
  - Positive: the report includes the best safe (no-accuracy-risk) config and the best achievable config (accuracy-risk flagged if any). Each recommended config carries its **exact official result from an unprofiled, fresh-server run**: Σ/Σ client TPS (the selection metric, with its status vs the 30-TPS target), P99 TTFT (ms, informational per DEC-2), request count (320), error count (0), observed concurrency (≈64), accept length, the resolved launch command/flags+env, and the resolved memory headroom (`max_total_num_tokens`, KV dtype, resolved DSA prefill/decode backend).
  - Positive: finalist numbers are confirmed across 2–3 fresh-server gate repeats with variance noted (per AC-2.1).
  - Positive: the report states the profiler-grounded central-question verdict and full reproducibility metadata — SGLang version/commit + branch, model snapshot hash, container/OS, CUDA/NCCL/torch/driver versions, GPU type/count + clock/power mode (if available), and the harness scripts used.
  - Negative: a report that names a "winner" without a fresh-server, 320/0-error gate run, or without its accompanying decode-profile insights markdown, is rejected.
  - Negative: a recommended config whose stated TPS/TTFT comes from a profiled run (rather than a clean gate run) is rejected.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A profile-driven sweep that: (1) establishes the incumbent (Loop 1 `combo`) gate baseline and a decode-phase profile; (2) runs the full DSA prefill×decode cross-product — attempting every cell, recording deterministic outcomes, and fully gate-benchmarking + profiling every launchable cell (no pruning, DEC-3); (3) attempts the EAGLE-tree axis and records its launch outcome plus the incumbent verify-share profile; (4) follows profile-directed follow-ups (overlap/scheduling/fusion knobs, and — only where the profile shows the relevant category is material — DSA env/threshold probes, with accuracy-risk flagged); and (5) delivers an evidence-backed central-question verdict and a final report with both recommended configs. Each evaluated candidate has a gate row + decode profile + bottleneck-insights markdown.

### Lower Bound (Minimum Acceptable Scope)
At minimum: the incumbent `combo` config is re-confirmed as a fresh-server gate run (320/0-error, client TPS + P99 TTFT) and profiled at concurrency 64; the Loop-1-open DSA cells are attempted and recorded (`--dsa-decode-backend flashmla_auto`; `flashmla_kv` decode/prefill under bf16) with source/log citations; the EAGLE-tree axis is attempted and its launch-reject recorded with citation; and the central question is answered with at least the incumbent decode profile's exposed-time category breakdown. Any unrun matrix cell or axis is skipped only with a cited profiler-grounded justification — never silently.

### Allowed Choices
- Can use: any `sglang serve` CLI flag and `SGLANG_*` env var permitted by scope; FP8 KV cache; DSA prefill/decode backend selection among the listed kernels; EAGLE chain (topk=1) tuning; `--chunked-prefill-size`, `--schedule-policy`, `--mem-fraction-static`, `--max-running-requests`, `--cuda-graph-max-bs`, overlap/scheduling-related flags. For diagnostics (owner decision DEC-4): the torch profiler via `/start_profile`–`/stop_profile`, `sglang.bench_serving` profiling flags, `SGLANG_TORCH_PROFILER_DIR`; a profile-only **copy** of the benchmark command (the gate run still uses the unmodified `development/benchmark.sh`); optionally `--enable-layerwise-nvtx-marker` and Nsight Systems captures. The **torch profiler is the required floor** — if NVTX/Nsight are unavailable or fail, proceed on the torch profiler alone rather than blocking. Profiler insights are extracted to a markdown file and the raw trace artifacts are then deleted (disk hygiene).
- Cannot use: any performance-affecting edit to SGLang source, kernels, tests, the workload/dataset, the benchmark harness, or the SLO definition; EP / MoE a2a backends (`--moe-a2a-backend`, deepep); alternate MoE runner backends; `--enable-torch-compile`; NGRAM speculative; pd-multiplexing; Blackwell-only (trtllm) / AMD-only (aiter) DSA kernels on H200; `--speculative-eagle-topk > 1` as a *benchmarkable* config on this DSA path (it is launch-rejected — record the rejection, do not work around it by changing the attention backend).

## Feasibility Hints and Suggestions

> **Note**: Reference only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
1. **Two run types, never mixed.** (a) *Gate run* = unmodified `development/benchmark.sh` on a fresh server, no profiler — produces the official TPS / P99 TTFT. (b) *Profile-only diagnostic run* = a separate fresh server with identical flags, replaying the same generated-shared-prefix workload at concurrency 64 via a copied `bench_serving` command (or `/start_profile`+`/stop_profile`), with `start_step` to skip warmup/cold-prefill and `num_steps` to bound a steady-state decode window.
2. **Group the decode loop correctly.** Under EAGLE, sum `DECODE + TARGET_VERIFY + DRAFT_EXTEND(_V2)` as the speculative decode loop; do not trust `--profile-by-stage`'s "decode" bucket alone (it classifies `TARGET_VERIFY` as prefill).
3. **Categorize with a map, report two shares, then discard the trace.** Use `with_stack`/`record_shapes` and an explicit kernel→category map (MoE GEMM / MLA+DSA attn+indexer / comms / sampling+draft+verify / other). Report summed-kernel-time share *and* exposed (critical-path, non-overlapped) share. Use a CUDA-graph-on trace for the verdict; optionally a graph-off trace only to improve source attribution. Write the extracted insights to a per-candidate markdown file under `development/loop2/profiling/`, then delete the raw trace artifacts (DEC-4 disk hygiene). The torch profiler is the floor; NVTX/Nsight are best-effort.
4. **Anchor first, then exhaust.** Profile the incumbent `combo` (and the accuracy-risk IndexCache config) first to characterize the decode loop; then fully run+profile every launchable DSA matrix cell (no pruning, DEC-3) and follow profile-directed follow-up knobs.
5. **Attempt-and-record the known non-starters** rather than skipping: `flashmla_auto` decode (runtime assert), `flashmla_kv` bf16 (launchable, slow), EAGLE topk>1 (launch ValueError). Each closes its axis with a deterministic, cited outcome.

### Relevant References
- `development/benchmark.sh` — the fixed gate workload (generated-shared-prefix, 4096 ISL / 512 OSL, conc 64, ~55% prefix hit, seed 31234). Do not modify.
- `development/loop2/CLIENT_SLOS.md` — the rebased official SLO (client TPS ≥ 30, P99 TTFT < 22 s).
- `development/loop1/` — `FINAL_REPORT.md`, `sweep_table.md`, `analysis_notes.md`, `run_candidate.sh`, `parse_result.py` — prior incumbent, dead-knob list, and reusable harness pattern.
- `python/sglang/srt/server_args.py` — `DSA_CHOICES` (`:282-290`); DSA forces `attention_backend=dsa` (`:1828-1854`) and `page_size=64` (`:1918-1920`, `:2848-2854`).
- `python/sglang/srt/layers/attention/dsa_backend.py` — decode dispatch has no `auto` branch, asserts on unknown (`:1661-1726`); `flashmla_auto` is prefill-only (`:2273-2289`); `flashmla_kv` quantizes bf16 cache (`:1815-1874`).
- `python/sglang/srt/arg_groups/speculative_hook.py:383-390` — `eagle_topk>1` + `page_size>1` rejection (non-flashinfer/fa3 backends).
- `python/sglang/srt/model_executor/forward_batch_info.py:78-149` — `ForwardMode` enum; `is_extend()`/`is_prefill()` include `TARGET_VERIFY`; `is_decode()` is `DECODE` only.
- `python/sglang/srt/entrypoints/http_server.py:960-993`, `python/sglang/srt/managers/io_struct.py:1732-1754` — `/start_profile`/`/stop_profile` and `ProfileReqInput` parameters.
- `python/sglang/srt/environ.py:241-255` — `SGLANG_TORCH_PROFILER_DIR`, `SGLANG_PROFILE_WITH_STACK`, etc.
- `.claude/skills/sglang-sota-performance/SKILL.md`, `.claude/skills/generate-profile/SKILL.md`, `.claude/skills/llm-torch-profiler-analysis/SKILL.md` — profiling/analysis workflow and kernel/overlap/fuse tables. Note: their default decode capture uses a synthetic short-in/long-out probe; for this task the profile-only run must instead replay the conc-64 shared-prefix workload.

## Dependencies and Sequence

### Milestones
1. **Harness + baseline**: stand up the Loop 2 fresh-server harness (reusing the Loop 1 pattern) and the profile-only diagnostic run; result parser computes client TPS.
   - Phase A: confirm the incumbent `combo` gate run (320/0-error, client TPS + P99 TTFT).
   - Phase B: capture the incumbent decode-phase profile (conc 64) and produce the first bottleneck analysis (category breakdown, top kernels, exposed vs summed share, EAGLE-verify share).
2. **Gap-closing sweeps** (depend on Milestone 1's baseline profile):
   - Phase A: DSA prefill×decode cross-product — run the Loop-1-open cells first (`decode=flashmla_auto`, `flashmla_kv` bf16), then fully run+profile every remaining launchable cell (no pruning, DEC-3); record the deterministic outcome of every cell.
   - Phase B: EAGLE-tree attempt (`topk>1`) recorded as launch-reject; incumbent verify-share characterized from profile.
   - Phase C: profile-directed follow-ups (overlap/scheduling/fusion knobs; DSA env/threshold probes pursued or skipped per profiler evidence — accuracy-risk flagged).
3. **Verdict + report** (depends on Milestones 1–2):
   - Step 1: answer the central question with the exposed-time category evidence.
   - Step 2: write the final report with recommended config(s) and reproducibility metadata.

<Dependencies are relative: profiles gate which sweep cells run; the verdict depends on the assembled profile evidence.>

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Stand up the Loop 2 fresh-server gate harness (reuse Loop 1 `run_candidate.sh` pattern) and a separate profile-only diagnostic run that replays the conc-64 shared-prefix workload with identical flags; ensure the parser computes client TPS and labels median ITL as cross-check | AC-1, AC-2 | coding | - |
| task2 | Re-confirm the incumbent `combo` config as a fresh-server gate run (320/0-error, client TPS, P99 TTFT) | AC-2, AC-9 | coding | task1 |
| task3 | Capture the incumbent decode-phase profile at conc 64 and produce the bottleneck analysis: kernel→category map, top-N kernels, summed vs exposed share, EAGLE-verify share, grouping DECODE+TARGET_VERIFY+DRAFT_EXTEND; write insights to a markdown file under `development/loop2/profiling/` and delete the raw trace | AC-3, AC-5 | analyze | task2 |
| task4 | Run the DSA prefill×decode cross-product: attempt the Loop-1-open cells (`decode=flashmla_auto` → runtime assert; `flashmla_kv` bf16 → launchable/slow) and all remaining cells; record deterministic outcome + source/log citation per cell; gate-benchmark AND profile every launchable cell (no pruning, DEC-3) | AC-4 | coding | task3 |
| task5 | Attempt the EAGLE-tree axis (`--speculative-eagle-topk 2`/`4`) and record the launch `ValueError` with `path:line`; confirm/deny the verify-cost prior from the incumbent profile | AC-5 | coding | task3 |
| task6 | Profile-directed follow-ups: name and test any allowed-flag headroom the baseline profile reveals (overlap/scheduling/fusion); pursue or skip DSA env/threshold knobs per profiler evidence, flagging accuracy-risk | AC-6, AC-7 | analyze | task3 |
| task7 | Maintain the coverage ledger: confirm every launchable cell is fully gate+profile measured and every infeasible axis is closed with a cited outcome; attribute each cell's delta to a kernel change via its profile | AC-4, AC-7 | analyze | task3, task4 |
| task8 | Answer the central question (hard MoE-GEMM vs flags-only headroom) with the exposed-time category evidence and the defensibility checklist | AC-6 | analyze | task3, task4, task5, task6 |
| task9 | Verify scope constraints: `git diff` empty for performance-affecting source/kernel/test/benchmark; resolved args show TP8 / no out-of-scope axes | AC-8 | analyze | task2 |
| task10 | Write the final report: recommended safe + best-achievable configs (accuracy-risk flagged), verdict, full reproducibility metadata, sweep table | AC-9 | coding | task7, task8, task9 |

## Claude-Codex Deliberation

### Agreements
- The official metric is client TPS `Σ output_tokens / Σ (latency − ttft)`; median ITL is a speculation-inflated cross-check only.
- Gate runs (unmodified `development/benchmark.sh`, fresh server, unprofiled) must be kept separate from profile-only diagnostic runs; profiling the gated run would perturb the very TPS being gated.
- Under EAGLE, the decode loop must group `DECODE + TARGET_VERIFY + DRAFT_EXTEND(_V2)`; `--profile-by-stage` classifies `TARGET_VERIFY` as prefill and cannot be trusted as the decode bucket.
- Kernel categories require an explicit kernel→category map (raw kernel names collapse into generic CUTLASS/Triton/GEMM buckets); the analysis must report both summed-kernel-time and exposed (critical-path) shares.
- The known non-starters should be attempted-and-recorded, not skipped: `flashmla_auto` decode (runtime assert, `dsa_backend.py:1726`), `flashmla_kv` bf16 (launchable, quantizes cache), EAGLE topk>1 (launch `ValueError`, `speculative_hook.py:383-390`).
- A finite candidate backlog is required to bound "every remaining knob"; every launchable cell/axis is measured and every infeasible axis is closed with a cited launch/runtime outcome.

### Resolved Disagreements
- **EAGLE-tree feasibility**: the draft framed topk>1 as a benchmarkable sweep ("watch accept_length vs verify cost"). Source confirms it is hard-rejected at launch on this DSA path (`speculative_hook.py:383-390`). Resolution: the axis is closed by recording the deterministic launch-reject with citation, plus a profiled verify-share of the incumbent topk=1 config — this satisfies the draft's "measured or profiler-justified, no silent skip" rule without an out-of-scope backend swap.
- **Profile workload**: the profiling skills default to a synthetic short-in/long-out decode probe, which is not the conc-64 shared-prefix workload. Resolution: profile-only runs replay the real workload at conc 64 (copied bench command / `/start_profile`), preserving gate integrity.
- **SLO gate framing (round 2 → owner override)**: Codex argued TPS ≥ 30 and P99 TTFT < 22 s are stated official gates and should not be open decisions. The owner then resolved DEC-1/DEC-2: the client-TPS *metric definition* stays fixed (AC-1), loop success = best-achievable + a defensible profiler-grounded ceiling verdict (not strict 30-TPS pass/fail), and P99 TTFT is **report-only** (recorded per candidate, not a disqualifier). Selection is on client TPS.
- **Cross-product coverage (round 2 → owner override)**: Codex flagged that pruning must not skip required cross-product cells. The owner then resolved DEC-3 to **no pruning** — every launchable cell is fully gate-benchmarked and profiled (AC-4.2); the profile attributes each cell's delta but never gates whether the cell runs.
- **Profile-only runs are non-scoring (round 2)**: made explicit in AC-2 — diagnostic profiling can never substitute for an unprofiled fresh-server gate run.

### Convergence Status
- Final Status: `converged` — two Codex passes (first-pass critique + second-pass reasonability review); all `REQUIRED_CHANGES` from the review were applied (non-scoring diagnostics, prune-scope restriction, exact gate-result fields, finalist repeats, rank-normalized profiler shares). All four user decisions (DEC-1…DEC-4) are now **RESOLVED** by the owner: loop success = best + ceiling proof; P99 TTFT report-only; no DSA-matrix pruning; all diagnostic tools allowed with the torch profiler as the non-blocking floor + raw-artifact deletion after insights extraction. No pending decisions remain.

## Pending User Decisions

- DEC-1: Given Loop 1 already showed flags-only 30 TPS is likely unreachable (~24.3 safe / ~26.5 accuracy-risk), what defines **loop success**?
  - Claude Position: success = the best-achievable config + a defensible profiler-grounded verdict on *why* the ceiling exists, even if the 30-TPS target is not reached flags-only.
  - Codex Position: keep the TPS metric definition hard; do not soften the metric.
  - Tradeoff Summary: the TPS *metric* stays fixed (AC-1); success is judged on best-achievable + ceiling proof, not strict pass/fail.
  - Decision Status: **RESOLVED — "Best + ceiling proof"**: loop success = best-achievable config + defensible profiler-grounded ceiling verdict (the 30-TPS target need not be reached flags-only). The metric definition itself is unchanged.

- DEC-2: Are the two SLO thresholds hard pass/fail, or report-only?
  - Claude Position: had proposed P99 TTFT as a hard disqualifier.
  - Codex Position: had proposed both as hard official gates.
  - Tradeoff Summary: owner overrode both: rank/select purely on client TPS; report P99 TTFT for information.
  - Decision Status: **RESOLVED — P99 TTFT is REPORT-ONLY**: client TPS is the ranking/selection metric; `P99 TTFT < 22 s` is recorded per candidate but does **not** disqualify a candidate (AC-1 updated accordingly).

- DEC-3: Should the DSA matrix be pruned when the profile shows attention/indexer is a small fraction of the decode step?
  - Claude Position: had proposed pruning below a ~10% exposed-time threshold.
  - Codex Position: pruning must never skip the launch/failure-taxonomy attempt.
  - Tradeoff Summary: owner chose the most exhaustive option.
  - Decision Status: **RESOLVED — NO PRUNING**: every launchable cell is fully gate-benchmarked AND profiled regardless of profiled attention share (AC-4.2 updated). The profile attributes each cell's delta; it never gates whether the cell runs.

- DEC-4: Which diagnostic tooling is allowed for profile-only runs, and how are artifacts handled?
  - Claude Position: copied profile-only bench command acceptable; torch profiler primary, NVTX/Nsight optional.
  - Codex Position: recommended exactly this diagnostics-vs-gate separation.
  - Tradeoff Summary: owner allowed all three tools, set the torch profiler as the non-blocking floor, and added an artifact-hygiene rule.
  - Decision Status: **RESOLVED**: copied `bench_serving --profile*` command, `--enable-layerwise-nvtx-marker`, and Nsight Systems are all allowed on profile-only runs (gate run stays unmodified). The **torch profiler is the required floor** — do not block if NVTX/Nsight fail. Extract profiler insights into a per-candidate markdown file under a dedicated directory (`development/loop2/profiling/`), then **delete all other (raw) profiling artifacts** to save disk (AC-3.4).

## Implementation Notes

### Code Style Requirements
- Any harness/parser/analysis scripts and their comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Phase", "Step", or similar workflow markers. These belong in this plan document only.
- Use descriptive, domain-appropriate naming (e.g. `client_tps`, `decode_loop_share`, `dsa_matrix_cell`) in any scripts.
- This task is flags-only: no performance-affecting edits to SGLang source, kernels, tests, the workload/dataset, or `development/benchmark.sh`. Permitted writes are confined to `development/loop2/` artifacts (harness/parser/analysis scripts, result files, profiling-insights markdown, tables, and this plan/report).
- Profiler artifact hygiene (DEC-4): raw profiling traces are transient. After extracting insights into a per-candidate markdown file under `development/loop2/profiling/`, delete the raw trace artifacts (trace JSON/`.gz`, Nsight `.nsys-rep`, etc.) to save disk. Do not commit large binary traces.

--- Original Design Draft Start ---

Task: Hillclimb GLM-5.1-FP8 on a fixed workload to meet the (rebased) client SLO using only
SGLang CLI flags and `SGLANG_*` env vars. We are testing out-of-box performance — no code changes
that affect SGLang performance.

Workload and Target: development/loop2/CLIENT_SLOS.md  (REBASED — read it first)
Benchmark Script: development/benchmark.sh   (fixed; do not modify — all gate numbers come from this
unchanged command). Workload is identical to loop 1: generated-shared-prefix, 4096 ISL (2253-token
shared system prompt + 1843-token question) / 512 OSL, max-concurrency 64, ~55% prefix-cache hit,
320 prompts, fixed seed.

Out-of-scope: code changes that affect SGLang performance; EP / MoE all-to-all backends (deepep,
--moe-a2a-backend); alternate MoE runner backends; --enable-torch-compile; NGRAM speculative;
pd-multiplexing; Blackwell-only (trtllm) / AMD-only (aiter) DSA kernels on H200.

Relevant Skills: .claude/skills/sglang-sota-performance

=== REBASED SLO (the change from loop 1 — read carefully) ===
The OFFICIAL per-user-speed metric is the client's verbatim TPS formula, NOT median ITL:
  TPS = total_output_tokens / (total_latency − TTFT)  (decode tokens ÷ decode wall-time)
      = Σ output_tokens / Σ (latency − ttft)  ≈  1000 / mean_tpot_ms
Target: TPS ≥ 30 per user, AND P99 TTFT < 22 s.
- median ITL / "1000/ITL" is a speculation-inflated cross-check ONLY (EAGLE bursts deflate it
  ~2.3×); never use it as the official verdict.
- Page size 64 is NOT a requirement (no preference for 64).
- FP8 KV cache is fully on the table (use freely if it helps).

=== PROFILING + BOTTLENECK ANALYSIS (first-class requirement for loop 2) ===
Drive this hill-climb with the /sglang-sota-performance workflow, using its profiling/analysis
portions (torch profiler) — NOT just black-box benchmarking:
- Profile the server BETWEEN candidate runs. After each fresh-server benchmark candidate, capture a
  decode-phase torch-profiler trace at concurrency 64 (via /sglang-sota-performance, and/or the
  generate-profile / llm-torch-profiler-analysis skills) and run a bottleneck analysis on it.
- Each bottleneck analysis must produce: the kernel time breakdown by category (MoE GEMMs vs
  MLA/DSA attention + DSA indexer vs all-reduce/comms vs sampling/draft-model/EAGLE-verify vs
  other), the top-N kernels by total time, overlap opportunities (idle/exposed gaps), and
  fuse-pattern candidates. Let the profile DECIDE the next knob — do not blind-sweep.
- Record per candidate, alongside its benchmark row: profile path, top-3 kernels by time, dominant
  bottleneck category (% of decode step), and any overlap/fusion/scheduling headroom observed.
- Central question profiling must answer: is the ~24–27 TPS decode ceiling hard MoE-GEMM compute
  (→ confirms expert parallelism is required, out of scope here) or is there any flags-only
  overlap / fusion / scheduling / attention-kernel headroom left? Conclude this explicitly with
  profiler evidence (not just benchmark deltas).

=== DSA / ATTENTION-BACKEND SWEEP — loop 2 MUST close these gaps ===
Loop 1 only spot-checked DSA sub-kernels (top-level attention_backend stayed `dsa`, correct for
this MLA+DSA model). It covered (bf16, all ≈ neutral ~24 TPS): prefill ∈ {flashmla_sparse(default),
fa3, flashmla_auto} and decode ∈ {fa3(default), flashmla_sparse}; the FP8 path forced
flashmla_kv/flashmla_kv and REGRESSED (21.96 TPS). Gaps loop 1 left open:
- `--dsa-decode-backend flashmla_auto` was never tried (decode only ∈ {fa3, flashmla_sparse}).
- `flashmla_kv` decode/prefill under **bf16** was deliberately skipped on theory (needs/quantizes
  for FP8) — loop 2 must actually ATTEMPT it and record the launch result / log reason, not skip.
- No full prefill×decode CROSS-PRODUCT was run — only one-off single-knob changes.

Loop 2 requirement — close the full matrix over the Hopper-valid DSA kernels
{flashmla_sparse, flashmla_kv, flashmla_auto, fa3} for `--dsa-prefill-backend` × `--dsa-decode-backend`
under bf16 (and note which combos SGLang rejects at launch, with the exact source/log reason):
- Run the untested decode backends first (decode = flashmla_auto; decode = flashmla_kv-under-bf16),
  then fill the remaining prefill×decode combinations.
- Profile each backend candidate (per the PROFILING section) and attribute any delta to a specific
  kernel change — this is what makes the sweep evidence-driven rather than brute force.
- Profiling MAY prune the matrix: if the bottleneck analysis shows MLA/DSA attention + indexer is a
  small fraction of the decode step, document that as the justification for not exhausting every
  remaining combo (cite the profile), rather than skipping silently. If attention IS a meaningful
  slice, exhaust the matrix.
- Also (only if a profile shows attention/indexer is material) probe the DSA-relevant env/flags
  (e.g. `--dsa-topk-backend`, `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`) as profile-directed
  follow-ups; threshold changes are accuracy-risk and must be flagged.

=== OTHER UNTESTED AXES — loop 2 should close or profile-justify skipping ===
- Speculative TREE (loop 1 only tried chains, eagle_topk=1): sweep `--speculative-eagle-topk` > 1
  (e.g. 2, 4) with matched `--speculative-num-draft-tokens`/`--speculative-num-steps`, watching
  accept_length vs verify cost. Loop 1's prior is that bigger verify batches cost more than they buy
  at conc 64 — profiling the EAGLE-verify share must confirm/deny before declaring this exhausted.
- Any axis the per-candidate bottleneck analysis flags as a meaningful fraction of the decode step
  becomes a required follow-up; an axis the profile shows is negligible may be skipped WITH the
  profiler evidence cited (no silent skips).
- Net rule for loop 2: every remaining knob is either measured, or skipped with explicit
  profiler-grounded justification — "covered all remaining gaps" must be backed by profiles, not assertions.

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
(Raise --max-running-requests to ≥ 64 — the speculative default of 48 caps admission below the
workload concurrency of 64.)

Prior knowledge from loop 1 (development/loop1/ — same model, same workload; use to avoid
re-treading dead knobs, but re-confirm with profiling):
- Best flags-only configs found: `combo` = cookbook EAGLE + `--chunked-prefill-size 4096`
  + `--schedule-policy lpm` (bf16) ≈ 24 TPS; `combo + IndexCache`
  (`--json-model-override-args '{"index_topk_pattern":"FFSF…SSS"}'`, ACCURACY-RISK) ≈ 26.5 TPS.
  30 TPS was NOT reached flags-only; P99 TTFT met (~11–12 s).
- Suspected binding bottleneck: MoE-decode compute at concurrency 64 — loop 2 should CONFIRM and
  QUANTIFY this with profiling (this is the main reason profiling was added).
- Dead / negative knobs at this concurrency (don't expect gains; profiling should explain why):
  DP-attention (regresses; per-rank batch collapses + DP-attn↔TP-MoE comms), FP8-KV standalone and
  combined (regresses; forces slower flashmla_kv decode, not capacity-bound), lighter EAGLE
  (accept_length collapses), --max-running-requests 80/96 (inert; workload caps conc at 64),
  --mem-fraction-static 0.9 / --cuda-graph-max-bs (inert; not capacity-bound), bf16 DSA backend
  swaps (neutral; decode pinned to fa3-class cost). DSA pins effective page size to 64.

Relevant and Useful Sources:
- docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx
- https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.1
- docs/basic_usage/deepseek_v32.md
- docs_new/docs/advanced_features/
- docs_new/docs/advanced_features/hyperparameter_tuning
- https://sgl-project-sglang-93.mintlify.app/optimization/performance-tuning

Notes:
- The official per-user metric is the TPS formula in CLIENT_SLOS.md (NOT median ITL) — this is the
  explicit SLO rebase from loop 1. Bake it into the plan as the official acceptance metric so the
  acceptance criteria are correct from the start.
- Profile between runs (see PROFILING above) — this is a first-class requirement for loop 2, not a
  nice-to-have.
- Assume FP8 KV cache is on the table.

--- Original Design Draft End ---
