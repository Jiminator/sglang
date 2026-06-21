# Ask Codex Input

## Question

You are doing a PRE-SWEEP METHODOLOGY REVIEW for a locked DS-vs-DSA serving benchmark in the SGLang repo
(branch dev/double-sparsity-standalone). This is a read-only analysis task: produce a concrete methodology
spec + gap list. Do NOT modify code. Repo root is the working dir.

CONTEXT (already owner-decided; do not relitigate):
- Goal: one locked, honest end-to-end comparison of table-free Double Sparsity (DS) vs native DSA on
  GLM-5.1-FP8, per development/SLOS.md (PRIMARY workload only: 4096 ISL / 512 OSL, conc 16/32/64, ~55%
  prefix hit; 30 TPS decode floor = output_tokens/(latency-TTFT); P99 TTFT < 22 s). Both sides radix-ON.
- DEC-2: publish BOTH a production-envelope comparison (DS mem 0.8 / DSA mem 0.85) AND a same-memory
  sensitivity comparison (both 0.8).
- DEC-3: judge per-request MEDIAN decode-TPS (comparator's enforced metric); aggregate total-tokens/s is
  REPORTED, not gated.
- DEC-4: run TWO trials per concurrency at the SAME per-conc seed — these are REPEATED run-to-run-stability
  measurements, NOT independent samples; report min/median/max. The comparator's --ac11 mode currently
  hard-requires >=3 trials; lower that floor to 2 as an in-scope tooling tweak.
- DEC-6: DS is judged against the ABSOLUTE SLO (30 TPS / P99 TTFT < 22 s) regardless of whether native DSA
  passes; DS/DSA ratios are reported as competitive position only. A DSA miss does NOT invalidate the op-point.
- Hard constraints: ONE TP=8 server at a time (two TP=8 servers do NOT fit on one 8xH200 node); never set
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments for serving.

FILES TO READ AND GROUND YOUR ANSWER IN (cite file:line):
- development/benchmark_compare.py (especially the --ac11 path: per-request decode-TPS p50 floor ratio
  AC11_TPS_FLOOR_RATIO=0.95, P99 TTFT ceil ratio AC11_TTFT_CEIL_RATIO=1.10, the absolute 30/22 gates,
  radix-state match enforcement, mem_fraction asymmetry allowance, the >=3-trial floor, and what it does
  with dense_fallback_total / sparse-selection counters — observe vs refuse).
- development/benchmark.sh (DS driver) and development/benchmark_baseline.sh (DSA driver): how SEEDS are set
  per concurrency and whether the same seed is reused across trials; trial/duration knobs.
- development/SLOS.md (the client bars).
- development/serve_double_sparsity.sh and development/serve_native_nsa.sh (the locked op-point flags;
  mem 0.8 DS / 0.85 DSA; radix fixture gate).

PRODUCE A STRUCTURED SPEC with these exact sections:
1. RUN_ORDER: the enforceable one-server run order that controls fresh-node thermal/clock drift given only
   one server fits at a time. Specify exactly how to interleave DSA/DS boots by trial so the comparison is
   paired (not block-scheduled), what to log per boot (boot order, /server_info snapshot, thermal/clock),
   and the rule for when a block-scheduled order must be LABELED unpaired.
2. PER_TRIAL_CAPTURE: the exact per-trial fields that must be recorded for (a) prefix-reuse / cached-token
   distribution (to show production-representative ~55% reuse, not just GSP shape), and (b) the AC-5 no-op
   refusal — dense_fallback_total==0 AND a positive sparse-selection proof (selected_tokens_mean <
   total_tokens_mean or equivalent). Name the actual counter fields if they exist in the bench JSONL/sidecar
   or server metrics; if a field is missing, say what must be added.
3. RECALL_COMPARABILITY: the procedure to compare recall@2048 of the regenerated served-fp8 op-point against
   the frozen loop9/runs/20260610_m0/recall_baseline.json with matched length-set + per-length sample-count
   equality; and the rule for when to instead define a fresh served-fp8 baseline.
4. SAME_MEMORY_DESIGN: how to structure the same-memory (both 0.8) comparison alongside the production-
   envelope (0.8/0.85) one so the comparator accepts each matched op-point and the two are not conflated.
5. TRIAL_FLOOR_CHANGE: the EXACT minimal edit to benchmark_compare.py to lower the --ac11 trial-count floor
   from 3 to 2 (cite the line/check), and confirm nothing else in --ac11 implicitly assumes >=3 trials.
6. GAPS: any remaining methodology gap that would let a reviewer challenge the verdict (e.g., warmup,
   discarding the first trial, TTFT measurement window, achieved-concurrency vs requested, seed-family
   matching across DS/DSA, what to do if DS is admission-capped below nominal concurrency).

Be concrete and cite file:line. This spec will directly drive the sweep implementation.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 5400s
- Timestamp: 2026-06-16_12-07-07
- Tool: codex
