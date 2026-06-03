# Round 19 Contract

## Mainline Objective (exactly one)
**AC-5 closure at the full-context Option-B operating point with MEASURED client-workload evidence.**
Per Codex R18 (STALLED): the bounded-context (`--context-length 8192`) result is bounded-workload
characterization only, not AC-5 completion; AC-5 must be measured at the fixed full-context Option-B point
(DS int8, `mem_fraction_static=0.7`, radix-on, TP=8, NO context cap — the point AC-8 validated). The
prerequisite linchpin: the `bench_serving` window-mode harness returned empty per-request latency
(`ttfts`/`itls`/`generated_texts` empty, impossible aggregate throughput) in R18, so **no valid AC-5 client
arrays can be produced until that is fixed and fails closed.** This round: (1) root-cause + fix the bench
harness fail-closed; (2) run the full AC-5 client workload at the full-context point with measured per-request
arrays + attribution + a fail-closed verifier; (3) profile the component breakdown (incl. the DSA FlashMLA+MoE
floor); (4) honestly state the full-context conc-16 result and the residual exact-blocked-top-k lever.

## Target AC(s)
- **AC-5 (task6)** — the done-criterion (owner: conc-16 strict + characterize 32/64). `coding` / hardware-run.

## Truly Blocking This Objective
- **The `bench_serving` window-mode empty-latency bug.** AC-5 is defined on measured P99 TTFT + exact
  per-request arrays; the harness must (a) fail closed when `completed>0` but `ttfts`/`itls`/`generated_texts`
  are missing, and (b) actually produce valid arrays so the client run is usable. This is the gating fix.

## Queued / Explicitly Out of Scope This Round
- **AC-10** — gated until AC-5 is met under the owner criterion + AC-3..AC-9 verified.
- **Cross-node wrapper smoke** — future-gated. **DSA-default conc-64 TPS ~29.4** — queued pre-existing limit
  (this round it is the *evidence* for the structural conc-64 ceiling, still not a DS defect).
- **The bounded-context (ctx8192) result** — kept as a separate characterization (`ac5_conc16_strict/`), NOT
  the AC-5 pass condition (per Codex R18).

## Concrete Success Criteria
1. **Bench harness fix (production code):** root-cause the R18 empty-latency failure (reproduce small), patch
   `python/sglang/bench_serving.py` and/or `development/benchmark.sh` so a run with `completed>0` but missing
   `ttfts`/`itls`/`generated_texts`/`output_lens` **fails closed** (refuses to publish), and so a valid run
   produces real per-request arrays. Add/adjust a regression if tractable.
2. **Measured full-context AC-5 client run:** DS int8 / mem-0.7 / radix-on / TP=8 / **full context** (no cap),
   conc 16/32/64, steady-state methodology (per the cold-flood lesson), radix-on proven from `.meta.json`.
   Publish exact per-request arrays + `client_slo_report` with **measured** P99 TTFT AND per-req TPS at all
   three conc + admission/decode attribution + a fail-closed verifier (reuse the `ac5_metrics_tool` pattern).
   No inferred TTFT.
3. **Component breakdown:** DS selection/top-k vs DSA FlashMLA+MoE floor vs token-label write vs scheduler —
   measured (DSA closed-batch floor at the same batch), so the next bottleneck is measured not guessed.
4. **Honest verdict:** the full-context conc-16 result (decode-bound at ~27 TPS/req without the topk kernel;
   conc-16 ≥30 at full context requires the exact blocked-topk — characterized with a concrete design + the
   within-block-K=2048 difficulty; the R18 bounded-context demonstrates 30.3 is reachable). conc-32/64 the
   structural decode-batch ceiling (DS ≤ DSA; conc-64 unattainable even for DSA). A Goal Tracker Update Request
   surfaces the bounded-context-vs-research-kernel tradeoff for the owner/Codex.
5. GPUs freed; commit + push to `jimmy`; goal-tracker + round-19-summary + BitLesson Delta updated.

## Applicable BitLessons (confirm per-task via bitlesson-selector)
- `BL-20260530-durable-tracked-acceptance-evidence` (fail-closed verifier; exact arrays; recompute the
  consumer's exact field) — directly drives the bench fail-closed fix + the AC-5 verifier.
- `BL-20260530-clean-latency-attribution` (row reconciliation; measured-vs-inferred; tail-to-tail) +
  `BL-20260530-cold-flood-not-steady-state-slo` (steady-state methodology) — the client run + attribution.
- `BL-20260530-admission-restore-tps-tradeoff` (per-req TPS = 1/decode_step_time; batch→TPS ceiling).
- `BL-20260531-ds-selection-fullwidth-overscan` (the topk residual is capture-width-bound; the blocked-topk
  design + difficulty).
- `BL-20260528-dsv32-ds-serving-boot-chain` + `BL-20260529-ds-radix-flip-config-bound-artifact` +
  `BL-20260530-remote-server-launch` (boot DS int8/mem-0.7/radix-on full-context + DSA floor; background boot,
  `ps | grep "[s]glang.launch_server"`, `pkill || true`, no foreground `sleep`).
- `BL-20260527-torch-topk-aliasing-corrupts-input` (if any topk-path code is touched).

## Out-of-bounds reminders
No ABI-lock / FlashMLA-assert / `top_k` change (AC-10 only). Compact path stays flag-gated, fp16 default. No
plan-process tokens in code/comments. Do not change the DS-fair AC-12 gate. Must not exit by lying / editing
loop state / cancel.
