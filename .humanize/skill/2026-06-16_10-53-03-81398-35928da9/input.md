# Ask Codex Input

## Question

You are doing a FIRST-PASS planning critique of a design draft for an SGLang engineering loop.
Do NOT write code. Produce a critique in the exact output format specified at the end.

## Repository context
- Repo: SGLang (high-performance LLM serving engine). Branch: `dev/double-sparsity-standalone`.
- This is a "Double Sparsity" (DS) sparse-attention research/eng project reproducing a 2-year-old
  paper (arXiv:2408.07092) on GLM-5.1-FP8 (MLA model) and comparing it to the model's native
  trained sparse indexer (DSA). Educational; DS is not expected to beat DSA.
- DS code: `python/sglang/srt/layers/attention/double_sparsity/` (calibrate.py, channel_mask.py,
  selector.py, selection_kernel.py, cuda_graph.py, config.py, validator.py).
- Prior loop "loop 11" landed M0–M3 (deleted the per-token TokenLabelTable, made table-free
  absorbed-latent scoring the served default, authorized radix-on under owner decision DEC-12) but
  was TERMINATED before its M4 verdict milestone (task8 per-step tax guard + task9 the locked
  AC-11 sweep). loop 11's full ledger is `development/loop11/results.md` and `queue.md`.
- The new draft (loop 11b) is at `development/loop11b/draft.md` — READ IT IN FULL. It also
  references `development/SLOS.md`, `development/serve_double_sparsity.sh`,
  `development/serve_native_nsa.sh`, `development/benchmark.sh`, `development/benchmark_baseline.sh`,
  `development/benchmark_compare.py`, and `development/serve_double_sparsity_radix_fixture.json`.
  READ those too as needed.

## What loop 11b is
A "finish + validate + productionize" loop, deliberately narrow. Three goals:
1. Re-establish the serving op-point on a NEW 8×H200 node (the loop-11 machine was released; the
   GLM-5.1 channel mask under `/models/` is GONE). This means: repoint the model path, REGENERATE
   the channel mask via calibrate.py, re-validate or re-mint the radix-on authorization fixture
   (its fingerprint pins the mask SHA-256), and re-confirm capacity/AC-7 reproduce on new hardware.
2. Close loop-11's M4: task8 (per-step tax guard at bs64, AC-4) + task9 (locked AC-11 sweep, 3
   trials × 600s, conc 16/32/64) → the loop's actual AC-2/AC-3/AC-4 HARD verdicts vs native DSA.
3. Deliver the headline end-to-end DS-vs-DSA comparison per SLOS.md (30 TPS decode floor, P99 TTFT
   < 22s) and clean up the DS production UX (stale model/mask defaults, dev-only knob sprawl, a
   serve-script throughput warning that contradicts loop-11's own ladder).

## Key facts to weigh
- Hardware now: 8×H200 (~144 GB/GPU), TP=8 — same CLASS as loop 11 but a DIFFERENT physical node.
- Model GLM-5.1-FP8 is present at the exact snapshot path the committed radix fixture names.
- The committed fixture pins channel_mask_sha256=340b6c0b… and selector_mode=table_free; if the
  regenerated mask is not byte-identical, radix-on is refused (fail-closed) and the DEC-12
  production-reuse edge probes must be re-run to re-mint the fixture.
- DS radix-on is value-equivalent to radix-off ONLY at production-representative reuse (~55%, the
  SLOS workload): recall within ±0.5pp, cross-rank selection identity, no dense fallback, clean
  boundary/eviction edge probes. At near-full reuse (≥98%) it deviates +1.57pp — owner-declared
  out-of-contract value-affecting (DEC-12). NOT bit-identical (v_h-driven, upstream of DS).
- DSA's own end-to-end output is ~77% nondeterministic run-to-run, so a DSA-on-vs-off recall
  parity check was inconclusive; the loop measures PERFORMANCE SLOs (TPS/TTFT), where DSA
  nondeterminism is a non-issue.
- The repo CONTRADICTS itself on the headline question: serve_double_sparsity.sh warns DS misses
  the SLO (loop8: decode-TPS 23/17/17 ≪ 30); loop-11's directional ladder shows DS p50 33–39 (above
  30). Nobody ran the locked sweep to settle it.

## Your job
Critique this plan draft hard, as a senior performance/serving engineer. Be specific and
code-grounded where you can (cite file paths). Focus on: what could make the verdict WRONG or
unreproducible; what op-point/measurement-discipline pitfalls are likely on fresh hardware; whether
the mask-regen → fixture-fingerprint chain is correctly reasoned; whether the AC bars and the
DS-vs-DSA comparison are fair and well-defined; what's missing for a defensible close-out; and
whether the UX-pass scope risks breaking userspace.

## Required output format (use these exact headers)
CORE_RISKS:
- (highest-risk assumptions and failure modes)
MISSING_REQUIREMENTS:
- (likely omitted requirements or edge cases)
TECHNICAL_GAPS:
- (feasibility or architecture gaps)
ALTERNATIVE_DIRECTIONS:
- (viable alternatives with tradeoffs)
QUESTIONS_FOR_USER:
- (questions that need explicit human decisions)
CANDIDATE_CRITERIA:
- (candidate acceptance criteria suggestions, AC-style)

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-16_10-53-03
- Tool: codex
