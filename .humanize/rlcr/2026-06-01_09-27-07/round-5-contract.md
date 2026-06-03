# Round 5 Contract

## Mainline Objective
Produce the **binding DS-vs-DSA same-node served-recall uplift matrix (AC-2)**:
measure the missing **DSA same-node NIAH reference** at the Loop-7 op-point
(int8-equivalent / mem 0.7, same node), assemble the **DS-default vs DS-hybrid
(Tier-2.B) vs DSA** matrix at 4K/16K/64K (N=20), state the **Clopper–Pearson
binomial CI materiality rule up front**, and record **within-budget (≤2048)
parity** + a **dense-DS (stride=1) reference**. This is the binding recall
evidence the loop's core question rests on, and supplies the DSA-same-node
artifact whose absence AC-2's negative tests reject.

## Target ACs (1–2)
- **AC-2** (primary): recall uplift measured DS-vs-DSA same node, floor =
  recorded+characterized non-regressing, materiality judged against the binomial CI.
- **AC-3** (secondary): within-budget (≤2048) recall parity + dense-DS 100%
  contribute to the Tier-2.B non-regression evidence.

## Blocking Side Issues In Scope
- None open. The R0–R3 oracle fail-open blocker was resolved in R4.

## Queued Side Issues Out Of Scope (justified)
- **Graph-safe Triton scorer port** (AC-3 "landed path"): bit-exact eager-vs-Triton
  equality for cosine/hybrid + int8 is research-grade kernel work; the recall
  evidence (this round) is the measure-first priority, the production port is a
  focused follow-on or documented disposition (the plan permits "landed path OR
  explicit disposition").
- **MMLU re-anchor + N≥50 binding 16K** (AC-3): long evals; extend opportunistically
  if the matrix lands with time to spare, else next round. Does not block the
  DS-vs-DSA matrix itself.
- **AC-4 lifted-budget**, **AC-6 perf/consolidation/final decision record**:
  sequenced after the recall matrix.
- **Analyzer artifact mislabel + budget-partial taxonomy** (Codex R4 gap #1): a
  small committed-evidence-integrity fix; bundled this round as a cheap correction
  (separate `uplift_4096_minus_2048` / `uplift_8192_minus_2048` fields + a
  `budget-partial` verdict + regenerate the JSON) — it does not drive the round.
- **Plan-marker / stale-comment cleanup**: pre-merge.

## Round Success Criteria
- **DSA same-node NIAH measured** at the Loop-7 op-point, N=20, 4K/16K/64K, with
  **served-recall vs admission-status as distinct fields** → a durable DSA
  reference artifact.
- **Consolidated DS-vs-DSA matrix** (DS-default, DS-hybrid/Tier-2.B, DSA) with,
  per cell: recall, N, and the **Clopper–Pearson 95% CI**; the materiality rule
  ("an uplift counts as material only when its delta exceeds the baseline's
  binomial CI") stated **before** any uplift claim. A legitimate negative result
  (no material uplift at a length, oracle-attributed scorer-/budget-limited) is an
  acceptable, loop-closing outcome when recorded + characterized + non-regressing.
- **Within-budget (≤2048-token) NIAH parity** recorded for DS-hybrid (target 100%)
  and a **dense-DS (stride=1) reference** alongside the default stride.
- **Analyzer artifact corrected + regenerated** (`oracle_budget_vs_scorer_r4.json`
  fields/taxonomy consistent with `m0_oracle_finding_r4.md`'s budget-partial 16K).
- Findings written (`development/loop7/`); all DS unit tests still pass;
  committed + pushed; goal-tracker mutable section + round-5-summary updated.
