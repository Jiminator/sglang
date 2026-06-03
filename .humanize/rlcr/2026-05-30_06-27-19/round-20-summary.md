# Round 20 Summary — AC-5 full-context MEASURED client evidence (conc-16 meets strict TTFT)

## Mainline objective (round contract)
Produce NEW full-context measured AC-5 client-workload evidence (Codex R19: no more restating the
bounded-vs-kernel tradeoff without new evidence). Root-cause the streaming empty-array, run the full
client workload at the **full-context Option-B point** (no `--context-length` cap), publish measured
arrays + attribution + a fail-closed verifier.

## What landed (commit `96bc789cc`)
1. **Streaming root cause — ctx8192-specific.** The R18 empty-latency arrays were the **bounded-context
   server only**: at full-context (DS int8/mem-0.7/radix-on/`context_len=163840`/TP=8) `bench_serving`
   produces real per-request arrays in both fixed-count and window mode (probes: completed=16/64, all
   ttfts real, thousands of ITL tokens). The R19 fail-closed guard passed on the full run (no empty-latency
   rows) — it stays the durable safety net.
2. **Measured full-context AC-5 client run** (steady-state warmup 120 s / window 300 s, conc 16/32/64,
   GSP 4096 ISL / 512 OSL, radix-on proven, `--enable-request-time-stats-logging`):

   | conc | achieved | **P99 TTFT** | `<22 s`? | per-req TPS p50 | `≥30`? |
   |---:|---:|---:|:--:|---:|:--:|
   | 16 | 16.00 | **13.13 s** | ✅ | 24.9 | ❌ |
   | 32 | 31.99 | 25.33 s | ❌ | 19.5 | ❌ |
   | 64 | 47.03 | 77.90 s | ❌ | 17.3 | ❌ |

   Exact per-request arrays + **fail-closed verifier** (`ac5_fullctx_metrics_tool.py --verify` PASS;
   recomputes P99 TTFT/TPS/achieved + asserts no empty-latency rows, output_len==512, errors empty,
   len==completed, 64-hex source SHA). No inferred TTFT.
3. **Measured admission-wait attribution** (`queue_duration` from ReqTimeStats, per-conc by print-time gap,
   n=256/320/315): queue p99 **10.5 / 22.6 / 74.0 s** → P99 TTFT is **admission-queue-dominated** at every
   conc (residual ~2.6/2.7/3.9 s = prefill). KV pool fits the concurrent set (64×4608≈295K<396K) → the
   queue is throughput/decode contention, not KV admission.

## Result (DEC-3 directional)
- **conc-16 MEETS the strict tail-latency SLO at full context (P99 TTFT 13.13 s < 22 s)** with admission
  restored (achieved 16.00) — measured, verifier-checked, with attribution.
- **Per-req TPS misses 30 at every conc** (24.9/19.5/17.3). conc-16's full-context gap (24.9 < 30) is the
  residual DS-selection `torch.topk` over-scan (the R18 bounded-context op-point reached conc-16 closed-batch
  30.3; the exact full-context fix needs the blocked-topk kernel). conc-32/64 = the structural decode-batch
  ceiling (DS ≤ DSA; even DSA is 37.0/29.4, so conc-64 ≥30 is unattainable for either).
- vs R6 full-context cold-flood (17.6/11.5/9.3 TPS): steady-state TPS markedly improved (24.9/19.5/17.3) via
  the R17 decode score-fix + steady-state methodology; TTFT collapsed vs Loop-5 (57.7/132.9/292.0 s).

## Files Changed
- `runs/20260530_dsv32_loop6/ac5_fullctx/` (NEW): `ac5_fullctx_report.md`, `ac5_fullctx_arrays.json` +
  `ac5_fullctx_metrics_tool.py` (exact arrays + fail-closed verifier), `ac5_fullctx_attribution.txt`
  (measured queue_duration per conc + decode components), `get_server_info_fullctx.json` + `meta_c16.json`
  (operating-point + radix-on `.meta.json` proof).
- `.humanize/bitlesson.md` — `BL-20260531-bench-empty-stream-failclosed` extended with the R20 ctx8192-specific
  root-cause; goal-tracker (R20 row + task6 note); round-20 contract/summary (gitignored loop state).
- (No production code change this round; R17 score-fix + R19 bench fail-closed fix stand.)

## Validation
- `ac5_fullctx_metrics_tool.py --verify` → PASS (recomputes the 3 conc rows; fail-closed sanity incl. the
  R18 empty-latency class). Operating point proven: full context, int8, mem 0.7, radix-on, stats-on, TP=8.
- Streaming probes at full-context (fixed-count + window mode) both produced real arrays. `git diff --check`
  clean; commit `96bc789cc` pushed to `jimmy`. GPUs freed (all 8 at 0 MiB; no live server).

## Remaining Items
- **conc-16 per-req TPS at full context (24.9 < 30):** the residual DS-selection topk over-scan. Either the
  exact full-context blocked-topk kernel (research-grade) or an explicit owner bounded-context rescope.
- **conc-32/64:** characterized structural ceiling (DS ≤ DSA; conc-64 unattainable even for DSA).
- **Gated AC-10** — after AC-5 met + AC-3..AC-9 verified. Cross-node smoke (future-gated), DSA conc-64 TPS
  ~29.4 (queued) unchanged. No ABI-lock change; DS-fair AC-12 gate unchanged.

## Goal Tracker Update Request
### Requested Changes:
- Record R20 Plan Evolution: full-context MEASURED AC-5 evidence landed — **conc-16 meets the strict TTFT
  axis (13.13 s < 22 s)** with attribution + fail-closed verifier; per-req TPS measured (24.9/19.5/17.3),
  admission-queue-dominated.
- **Owner decision (now backed by measured full-context data):** conc-16 strict TPS (≥30) at FULL context
  needs the research-grade blocked-topk kernel; the bounded-context op-point already reaches conc-16 30.3
  (closed-batch) and is the natural deployment for the 4608-token client workload (64K servability = separate
  full-context point, AC-8/R16). Please confirm (a) invest the rounds in the full-context blocked-topk kernel,
  or (b) accept the bounded-context op-point as the conc-16 client-SLO deployment. conc-32/64 ≥30 is
  structurally unattainable (DS ≤ DSA) regardless.
### Justification:
This round delivered the measured full-context evidence Codex required (conc-16 strict TTFT met; TPS +
attribution measured, not inferred), resolving the R18/R19 measurement blockers. The only remaining AC-5
gap is the conc-16 TPS axis at full context, whose two resolution paths (research kernel vs bounded-context
deployment) reach the same conc-16 outcome — an owner steer prevents spending multiple rounds on a kernel
that does not change the structurally-unattainable conc-32/64 result.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260531-bench-empty-stream-failclosed
Notes: Extended with the R20 root-cause — the runtime empty-stream cause was localized to the **ctx8192
bounded-context server only**; at full-context `bench_serving` produces valid per-request arrays (fixed-count
+ window mode), so the empty-array was a `--context-length 8192` × GSP-shared-prefix × radix streaming
interaction, not a general window-mode bug. The R19 fail-closed guard is the durable safety net. Applied
existing lessons: BL-20260530-clean-latency-attribution (per-conc queue_duration bucketed by print-time gaps,
measured-vs-inferred, tail-to-tail), BL-20260530-cold-flood-not-steady-state-slo (steady-state warmup120/
window300), BL-20260530-durable-tracked-acceptance-evidence (exact arrays + fail-closed verifier),
BL-20260531-ds-selection-fullwidth-overscan (the conc-16 full-context TPS residual = topk over-scan). No
production code changed this round.
