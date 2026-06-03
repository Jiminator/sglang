# Round 6 Contract

## Mainline Objective
Port the **Tier-2.B scorer into the graph-safe Triton decode selector (AC-3
"landed path")**: bring `scorer_norm ∈ {cosine, hybrid}` (with
`scorer_norm_hybrid_threshold`) and `head_agg ∈ {max, mean}` into
`_logical_score_kernel` / `retrieve_topk_graph_safe`, matching the eager
`_compute_logical_token_scores` math, with **eager-vs-graph selection-equality
evidence on GPU**, and relax the startup guard + `_force_eager_select` so these
variants run on the production CUDA-graph path instead of requiring
`--disable-cuda-graph`. The winning scorer (R5: material 16K uplift) is currently
eager-only — this is the production-viability keystone that unblocks AC-3
non-regression-at-production-path, AC-6 perf, and the final decision record.

## Target ACs (1–2)
- **AC-3** (primary): Tier-2.B variant on a **landed graph-safe path** (the plan's
  "landed graph-safe path OR explicit disposition" — this round lands it).
- **AC-6** (secondary): removes the `--disable-cuda-graph` requirement for the
  winning scorer, a prerequisite for the conc-1/16 perf guardrails.

## Blocking Side Issues In Scope (the objective itself)
- **Numerical eager-vs-graph equality for cosine/int8.** Triton fp32 reductions
  vs torch may differ at the last bits, so bit-exact top-K selection is not
  guaranteed at boundary ties. In scope: match the eager math exactly (cosine =
  unit-normalized dot, scale-ignored; hybrid = per-request `seq_len > threshold`
  switch; head_agg mean), reuse the R23 deterministic tie-break, and
  **characterize** equality on GPU — bit-exact where scores are well-separated,
  with any residual fp-tie divergence documented as a tolerance (not hidden).

## Queued Side Issues Out Of Scope (justified)
- **Anchor-mode graph-safe port**: anchor force-include is a post-topK per-row op
  (exploratory, not the Tier-2.B winner); stays eager / a follow-on. Non-default
  `anchor_mode` continues to require `--disable-cuda-graph`.
- **AC-3 binding measurement matrix** (N≥50 16K, MMLU ≤1.0pp re-anchor, dense-DS,
  graph-vs-eager perf): needs the port done first → next round.
- **AC-1 task4 (CUDA alloc-detector under graph replay) + dense/default stride
  reference** (Codex gap #1): contained AC-1 closure; sequenced.
- **AC-4 lifted-budget** (task13–17), **AC-6 consolidation + final decision
  record** (task19–20): after AC-3.
- **R5 evidence-label cleanup** (Codex gap #5: DSA op-point label, materiality
  wording): cheap; bundle if time, else queued.

## Round Success Criteria
- `_logical_score_kernel` computes `scorer_norm` (cosine/hybrid) + `head_agg`
  (mean) when flagged (per-request seq_len read for hybrid), matching eager math;
  default (off/max) byte-identical to the current kernel.
- `retrieve_topk_graph_safe` (and the deepseek_v2 graph-safe call site + the DS
  cuda_graph capture path) thread the scorer flags; replay stays allocation-free
  (any new intermediate uses pre-alloc scratch or in-kernel registers — no new
  per-call alloc).
- **Eager-vs-graph selection equality demonstrated on GPU** for raw / cosine /
  hybrid / head_agg{max,mean}: bit-exact where scores separate; any boundary-tie
  fp divergence quantified + documented.
- Validator no longer rejects cosine/hybrid/head_agg under CUDA graph;
  `ds_scorer_is_default` / `_force_eager_select` updated so they no longer force
  eager (anchor still does). Default flag-off path unchanged.
- A new graph-safe scorer-variant unit/GPU test; all DS unit tests pass; live
  smoke on a graph-mode hybrid server confirms DS engages (sparsity meta) +
  recall is non-trivial (sanity vs the eager R5 numbers). Committed + pushed;
  tracker + round-6-summary updated.
