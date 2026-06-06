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
