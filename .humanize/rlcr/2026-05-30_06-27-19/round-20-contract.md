# Round 20 Contract

## Mainline Objective (exactly one)
**Produce NEW full-context measured AC-5 client-workload evidence.** Boot DS int8 / `mem_fraction_static=0.7`
/ radix-on / TP=8 at the **full-context Option-B operating point (no `--context-length` cap)**, root-cause +
fix the live streaming empty-array failure (so `bench_serving` emits real `ttfts`/`itls`/`generated_texts`/
`output_lens`), then run the full AC-5 client workload (conc 16/32/64, 4096 ISL / 512 OSL) and publish
**measured** per-request arrays + P99 TTFT/TPS at all three conc + admission/decode attribution + a fail-closed
verifier + radix-on `.meta.json` proof + the DSA FlashMLA+MoE component floor. This is the concrete forward
progress Codex R19 requires (no more restating the bounded-context-vs-kernel tradeoff without new evidence).

## Target AC(s)
- **AC-5 (task6)** — the done-criterion (owner: conc-16 strict + characterize 32/64). `coding` / hardware-run.

## Truly Blocking This Objective
- **The live streaming empty-array root cause.** R19 made the harness fail closed, but the bench still must
  PRODUCE valid arrays. Until the runtime cause (window-driver / abort / dispatch interaction; the server
  format is standard) is reproduced small and fixed, no measured AC-5 client arrays exist. This is the
  gating fix and the first task.

## Queued / Explicitly Out of Scope This Round
- **The exact full-context blocked top-k kernel** — research-grade; the full-context conc-16 ≥30 lever. This
  round produces the MEASURED full-context conc-16 number (expected ~27 TPS/req, < 30); whether to invest in
  the kernel vs accept the bounded-context op-point stays an owner decision (surfaced, not re-litigated). The
  bounded-context result is characterization only (Codex R18).
- **AC-10** — gated until AC-5 verified. **Cross-node smoke** — future-gated. **DSA conc-64 TPS ~29.4** —
  queued pre-existing limit (this round = the structural-ceiling evidence).

## Concrete Success Criteria
1. **Live root-cause + fix of the streaming empty-array bug:** reproduce small (fixed-count streaming bench
   on the booted full-context server), identify why the stream yields no tokens, fix it so a valid run emits
   real per-request arrays. (If the fix is server-side or harness-side, land it with a regression where
   tractable.)
2. **Measured full-context AC-5 client run:** conc 16/32/64, radix-on proven from `.meta.json`, steady-state
   methodology (cold-flood lesson), exact per-request arrays + a fail-closed verifier (reuse `ac5_metrics_tool`
   pattern) that checks completion counts, output_len==512, TTFT/TPS, ITL source, errors, radix proof. No
   inferred TTFT — measured at all three conc.
3. **Attribution + component breakdown:** admission-wait vs decode at each conc + the DSA FlashMLA+MoE floor.
4. **Honest verdict:** the measured full-context numbers vs the strict SLO (conc-16 decode ~27 < 30 expected;
   conc-32/64 the structural ceiling, DS ≤ DSA). Record as directional (DEC-3) with the conc-16 strict gap
   attributed to the residual full-context top-k (kernel) — and surface the owner rescope question (bounded
   vs kernel) ONCE in the summary's Goal Tracker Update Request, not as the round's only output.
5. GPUs freed; commit + push to `jimmy`; goal-tracker + round-20-summary + BitLesson Delta updated.

## Applicable BitLessons (confirm per-task via bitlesson-selector)
- `BL-20260531-bench-empty-stream-failclosed` (the R19 fail-closed fix; the empty-array failure mode + the
  closed-batch cross-check).
- `BL-20260530-durable-tracked-acceptance-evidence` + `BL-20260530-clean-latency-attribution` +
  `BL-20260530-cold-flood-not-steady-state-slo` (exact arrays + fail-closed verifier; clean per-conc
  attribution + steady-state methodology).
- `BL-20260530-admission-restore-tps-tradeoff` + `BL-20260531-ds-selection-fullwidth-overscan` (per-req TPS =
  1/decode_step_time; the full-context topk residual).
- `BL-20260528-dsv32-ds-serving-boot-chain` + `BL-20260529-ds-radix-flip-config-bound-artifact` +
  `BL-20260530-remote-server-launch` (boot DS int8/mem-0.7/radix-on full-context + DSA floor; background boot,
  `ps | grep "[s]glang.launch_server"`, `pkill || true`, no foreground `sleep`).

## Out-of-bounds reminders
No ABI-lock / FlashMLA-assert / `top_k` change (AC-10 only). No `--context-length` cap for the AC-5 pass
(full-context only; bounded-context stays characterization). No plan-process tokens in code/comments. Do not
change the DS-fair AC-12 gate. Must not exit by lying / editing loop state / cancel.
