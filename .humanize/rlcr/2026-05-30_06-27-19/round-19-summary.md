# Round 19 Summary — bench_serving fail-closed (AC-5 measurement linchpin); AC-5 closure still open

## Mainline objective (round contract)
AC-5 closure at the full-context Option-B operating point with MEASURED client-workload evidence (Codex
R18 STALLED: bounded-context is characterization only; AC-5 must be measured at full-context). The gating
prerequisite was the `bench_serving` window-mode empty-latency bug — without valid client arrays there is
no AC-5 proof regardless.

## What landed (commit `7f896b454`)
**`bench_serving` fail-closed on empty-latency streaming runs** (Codex R18 blocking issue 1/2).
- **Root cause:** `async_request_sglang_generate` initialises `output_len` to the *requested* count (512)
  and set `output.success=True` + `output.output_len=output_len` on **any HTTP 200**, even when the stream
  yielded no token (`ttft` never set, `generated_text=''`). So R18's empty generations were recorded as
  full 512-token completions → impossible aggregate throughput (24,599 tok/s vs the ~485 tok/s closed-batch
  reality). The same server generated correctly on direct `/generate` and under the closed-batch client —
  it was a harness-accounting bug, not a server defect.
- **Fix (fail closed at two layers):** (1) the request-func marks `success=False` + records the actual
  `output_len` (0) when a 200 produces no decoded token; (2) `calculate_metrics` **raises** when a streaming
  run reports `completed>0` but captured zero per-request latency (no ITLs and all `ttft==0`), refusing to
  emit fabricated TTFT/ITL/throughput. `--disable-stream` is legitimately exempt.
- **Tests:** 3 new regressions in `test/registered/unit/development/test_bench_serving_timing.py`
  (degenerate streaming → RuntimeError; valid streaming → passes with p99_ttft>0; `--disable-stream`
  empty-ITL → allowed); **12 pass** in the file.

## Honest status — AC-5 closure still open (NOT a completion)
This round fixed the prerequisite (the harness can no longer masquerade an empty-generation run as a valid
AC-5 result) but did **not** land the full AC-5 closure. Remaining (budget + genuine difficulty):
1. **Live root-cause of the empty-generation stream.** The server streaming format is standard
   (`data: {"text": cumulative, "meta_info": {"completion_tokens"}}`) and the bench parser matches it, so
   the empty stream is a runtime/window-driver/abort interaction at this config that needs a small live
   reproduction (fixed-count non-window streaming bench) to localize and fix so the bench produces real arrays.
2. **Full-context AC-5 client run** (DS int8/mem-0.7/radix-on/TP=8, no context cap), conc 16/32/64, exact
   per-request arrays + measured P99 TTFT/TPS + attribution + fail-closed verifier — blocked on (1).
3. **The exact full-context blocked top-k** (Codex's main code ask) for conc-16 ≥30 at full context. This is
   research-grade: under CUDA-graph capture the topk score-buffer width is fixed and `torch.topk` cannot
   skip, so an exact no-context-cap speedup needs a within-block K=2048 top-k kernel (the stubbed
   `DSGraphState.scratch_partial_*` path was never implemented; a torch reshape-topk still processes the full
   width). The R18 bounded-context op-point reaches conc-16 30.3 cheaply but Codex rejects it as a context cap.

## Files Changed
- `python/sglang/bench_serving.py` — fail-closed on empty-latency streaming (request-func + calculate_metrics).
- `test/registered/unit/development/test_bench_serving_timing.py` — 3 fail-closed regressions.
- `.humanize/bitlesson.md` — new lesson `BL-20260531-bench-empty-stream-failclosed`; goal-tracker (R19 row +
  task6 note); round-19 contract/summary (gitignored loop state).

## Validation
- `pytest test/registered/unit/development/test_bench_serving_timing.py` → **12 passed** (9 existing + 3 new).
- Guard logic unit-checked: R18-degenerate (completed>0, empty ITL, ttft all 0, streaming) → RAISE; valid
  streaming → pass; `--disable-stream` empty-ITL → pass.
- `git diff --check` clean; commit `7f896b454` pushed to `jimmy`. No server booted this round (CPU-only fix).

## Remaining Items
- AC-5 closure (items 1-3 above). **Gated AC-10** — after AC-5 met + AC-3..AC-9 verified. Cross-node smoke
  (future-gated) and DSA conc-64 TPS ~29.4 (queued) unchanged. No ABI-lock change; DS-fair AC-12 unchanged.

## Goal Tracker Update Request
### Requested Changes:
- Record R19 Plan Evolution: the `bench_serving` fail-closed fix (the AC-5 measurement prerequisite) landed +
  tested; AC-5 closure remains open on (1) live streaming root-cause, (2) full-context client run, (3) the
  full-context blocked top-k.
- **Owner decision needed (bounded-context vs research-kernel):** Codex R18 rejected the R18 bounded-context
  op-point (`--context-length 8192`, conc-16 30.3) as outside full-context Option-B and requires the exact
  full-context blocked top-k. That kernel is research-grade (within-block K=2048 top-k under CUDA-graph, with
  adversarial/zero-alloc regression coverage) for a borderline conc-16 ~30.3 gain that the bounded-context
  op-point already demonstrates, while conc-32/64 stay structurally <30 regardless. Please confirm whether to
  (a) invest the rounds in the full-context blocked-topk kernel, or (b) accept the bounded-context op-point as
  the conc-16-strict client-SLO deployment (with 64K servability as the separate full-context point, AC-8/R16).
### Justification:
The fail-closed fix is a real, tested production change addressing Codex's blocking item and is the
prerequisite to any valid AC-5 client measurement. The remaining AC-5 closure hinges on a hardware/harness
root-cause and a research-grade kernel whose payoff (full-context conc-16 ~30.3) equals the already-demonstrated
bounded-context result — an owner steer on (a) vs (b) avoids spending multiple rounds on a kernel that does not
change the structurally-unattainable conc-32/64 outcome.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260531-bench-empty-stream-failclosed
Notes: New lesson — a benchmark that backs an SLO/AC must FAIL CLOSED when "completed" requests carry no
per-request latency/text, because `async_request_sglang_generate` recorded HTTP-200-empty-stream as a full
`max_new_tokens` completion (output_len initialised to the requested count; success=True on any 200) →
fabricated throughput, and `calculate_metrics`'s percentile arrays silently stayed empty. Fix at both layers
(request-func marks empty-stream failed; calculate_metrics raises on completed>0 + zero per-request latency on
a streaming run; `--disable-stream` exempt). Applied existing lessons: BL-20260530-durable-tracked-acceptance-evidence
(fail-closed verifier / refuse to publish unusable metrics — extended here from the verifier to the producer),
BL-20260530-cold-flood-not-steady-state-slo (methodology context). No production decode-path code changed this
round; the R17 score-fix stands and the residual full-context top-k remains the open lever.
