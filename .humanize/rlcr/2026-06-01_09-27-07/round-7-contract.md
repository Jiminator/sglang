# Round 7 Contract

## Mainline Objective
Produce the **binding AC-3 graph-mode non-regression matrix** for the landed
Tier-2.B hybrid scorer: re-measure DS-default, DS-hybrid (graph-safe), and DSA at
the production CUDA-graph op-point in one session, with **N≥50 at 16K** (to bind
the marginally-material R6 25% hybrid uplift), a **durable dense-DS within-budget
(≤2048) parity artifact** (labeled dense-DS, not "dense-equivalent"), and an
**MMLU re-anchor (DSA vs DS-hybrid) at mem 0.7 with the ≤1.0pp gate**. This turns
the R6 preliminary production recall into binding AC-3 non-regression evidence.

## Target ACs (1–2)
- **AC-3** (primary): Tier-2.B non-regression — within-budget parity, MMLU ≤1.0pp
  of a re-anchored op-point DSA baseline, dense-DS 100%, binding 16K uplift.
- **AC-2** (secondary): the production-path (graph-mode) DS-vs-DSA recall uplift
  with N≥50 binding CIs firms up the AC-2 final claim.

## Blocking Side Issues In Scope
- None open. The R6 graph-safe scorer port is accepted; the scorer is landed and
  bit-identical to eager. This round only measures it.

## Queued Side Issues Out Of Scope (justified)
- **anchor_mode graph-safe port** (Codex gap #1 sub-item): a kernel task
  (post-topK force-include with graph-state scratch + replay-equality tests) —
  separable from the scorer measurement; anchor stays eager-only this round
  (task #15 / its own round). The Tier-2.B winner is the hybrid scorer, not anchor.
- **graph-vs-eager perf delta** (AC-6, conc-1/16 TTFT/decode-TPS/mem): the AC-6
  consolidation round; needs the perf harness.
- **AC-1 task4 alloc-detector + dense/default oracle-stride artifact**: a separate
  contained AC-1 closure round (Codex gap #2).
- **AC-4 lifted-budget** (task13–17), **AC-6 consolidation + final decision
  record** (task19–20): sequenced after AC-3 measurement.
- **Stale-comment / plan-marker / R5 evidence-label cleanup**: pre-merge hygiene;
  bundle the R5 DSA op-point label + materiality-wording fix if cheap, else queued.

## Round Success Criteria
- **N≥50 graph-mode 16K served recall**, same session/op-point, for DS-default,
  DS-hybrid (graph-safe), and DSA, with exact Clopper–Pearson CIs and the
  directional materiality rule (hybrid material only if its point exceeds the
  DS-default baseline CI high). Keep 1024w/4K/64K rows for parity + regression
  context. The 16K hybrid uplift is recorded as binding (material or, honestly,
  not — either closes AC-2/AC-3 if characterized + non-regressing).
- **Dense-DS within-budget parity artifact**: a within-budget (≤2048-token) NIAH
  run, explicitly labeled dense-DS, showing DS-hybrid stays 100% (dense within
  the budget), alongside DSA/default.
- **MMLU re-anchor**: DSA and DS-hybrid (graph-safe) at the Loop-7 single-node
  mem0.7 op-point (5-shot, a stated sample size); report accuracy + delta; the
  scorer passes only if DS-hybrid is within ≤1.0pp of the re-anchored DSA.
- Consolidated AC-3 non-regression finding + JSON artifacts; DS unit tests still
  pass; committed + pushed; goal-tracker mutable section + round-7-summary updated.
