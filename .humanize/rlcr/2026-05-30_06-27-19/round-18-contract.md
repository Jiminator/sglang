# Round 18 Contract

## OWNER DECISION (resolved this round, R12-style)
Asked the loop owner the AC-5 done-criterion given the structural finding (≥30 TPS/req at conc-32/64 is
unattainable for DS — even DSA, the faster path DS cannot exceed, is 29.4 at conc-64). **Owner chose
"Conc-16 strict + characterize 32/64"** (and confirmed it as the recommended option): make the strict
gate **conc-16** (≥30 TPS/req AND P99 TTFT <22 s), and **characterize conc-32/64** as the structural
decode-batch ceiling (DS ≤ DSA; conc-64 DSA = 29.4) — a documented hardware/model limit, not a DS or
footprint defect. AC-5 = met once conc-16 strict-passes + 32/64 are characterized. Recorded as a Plan
Evolution entry in `goal-tracker.md`.

## Mainline Objective (exactly one)
**Make DS strict-pass the client SLO at conc-16 and characterize conc-32/64.** Implement the residual
`torch.topk` over-scan fix in `retrieve_topk_graph_safe` (an **exact** seq-aware path per Codex's R17
plan — no context cap, no `top_k` change, preserves AC-8) so conc-16 closed-batch reaches **≥30 TPS/req**;
add regression coverage (monolithic vs new top-k on adversarial fixtures + a CUDA-graph zero-alloc test);
re-measure closed-batch + the **component breakdown** (DSA FlashMLA+MoE floor Codex flagged missing); then
run the full client workload (NUM_PROMPTS=320, conc 16/32/64, 4096/512, radix-on, TP=8) with exact arrays
+ a fail-closed verifier + admission/decode attribution, asserting **strict conc-16** and **characterizing
conc-32/64** against the DSA ceiling.

## Target AC(s)
- **AC-5 (task6)** — the done-criterion. `coding` / hardware-run / owner claude.

## Truly Blocking This Objective
- **Whether the strict all-conc pass is the loop's done-criterion or AC-5 is graded directional (DEC-3).**
  This is the gating decision: the strict ≥30 at conc-32/64 is structurally unattainable for DS (and
  conc-64 even for DSA), so without an owner ruling the loop cannot converge. Resolve it before sinking
  a round into the conc-16-only residual top-k kernel.

## Conditionally In Scope (only if the owner chooses "continue chasing strict")
- The residual `torch.topk` over-scan fix in `retrieve_topk_graph_safe` (an **exact** seq-aware blocked
  top-k per Codex's R17 plan), to push conc-16 from 27.1 → ~30. Does NOT help conc-32/64 (structural).

## Queued / Explicitly Out of Scope This Round
- **AC-10** — gated until AC-5 is resolved + AC-3..AC-9 verified.
- **Cross-node wrapper smoke** — future-gated (single-node round).
- **DSA-default conc-64 TPS ~29.4** — the queued pre-existing DSA/H200 limit; this round makes it the
  *evidence* for the structural conc-64 unattainability, but it stays a queued tension, not a DS bug.
- More footprint reduction (KV pool fits) and more AC-7/AC-8 evidence (both verified).

## Concrete Success Criteria
1. A full AC-5 client-workload run at the R17-patched DS int8/mem-0.7/radix-on point: `client_slo_report`
   with the **absolute** P99 TTFT + per-req TPS vs strict `<22.0 / ≥30` at conc 16/32/64, radix-on proven
   from `.meta.json`, exact per-request arrays + a **fail-closed verifier** (reuse the `ac5_metrics_tool`
   pattern), and the admission-wait-vs-decode attribution — all durable/tracked.
2. The **component breakdown** Codex requested: the DSA closed-batch decode floor (FlashMLA+MoE) measured
   at the same batch, so the DS step is decomposed into selection (now tight) + shared floor — no
   unmeasured residual guessing.
3. A rigorous **structural characterization**: DS per-req TPS vs DSA vs the strict 30 at each conc, the
   decode-batch→TPS curve, and the explicit statement that `≥30 at conc-32/64` is unattainable on this
   hardware (conc-64 even for DSA), with attribution — per DEC-3 + the Lower Bound.
4. The **AC-5 done-criterion decision** obtained from the owner (AskUserQuestion) + recorded as a Goal
   Tracker Plan-Evolution entry / Update Request, and the round's remaining work driven by that answer.
5. GPUs freed; commit + push to `jimmy`; goal-tracker + round-18-summary + BitLesson Delta updated.

## Applicable BitLessons (confirm per-task via bitlesson-selector)
- `BL-20260530-admission-restore-tps-tradeoff` — per-req TPS = 1/decode_step_time; high concurrency
  (large decode batch) inherently lowers per-req TPS; measure at the real operating point.
- `BL-20260530-cold-flood-not-steady-state-slo` — pick `num_prompts`/warmup so the epoch is steady-state;
  don't read cold-flood TTFT as the SLO.
- `BL-20260530-durable-tracked-acceptance-evidence` + `BL-20260530-clean-latency-attribution` — exact
  arrays + fail-closed verifier; clean per-conc admission-vs-decode attribution with row reconciliation.
- `BL-20260531-ds-selection-fullwidth-overscan` — the R17 score-fix is in; the topk over-scan is the
  conc-16-only residual (only if "continue chasing").
- `BL-20260528-dsv32-ds-serving-boot-chain` + `BL-20260529-ds-radix-flip-config-bound-artifact` +
  `BL-20260530-remote-server-launch` — boot DS int8/mem-0.7/radix-on (+ DSA-default for the floor);
  background boot, `ps | grep "[s]glang.launch_server"`, `pkill || true`, no foreground `sleep`.

## Out-of-bounds reminders
No ABI-lock / FlashMLA-assert / `top_k` changes (AC-10 only). No new serve/bench scaffolding — reuse
Loop-5 scripts. No plan-process tokens in code/comments. Do not change the DS-fair AC-12 gate. Must not
exit by lying / editing loop state / cancel.
