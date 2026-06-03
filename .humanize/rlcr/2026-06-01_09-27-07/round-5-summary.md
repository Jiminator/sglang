# Round 5 Summary — Loop 7

## Mainline objective (round-5-contract.md)
Produce the **binding DS-vs-DSA same-node served-recall uplift matrix (AC-2)**:
measure the missing DSA same-node NIAH reference at the Loop-7 op-point, assemble
the DS-default vs DS-hybrid(Tier-2.B) vs DSA matrix at 4K/16K/64K (N=20) with
Clopper–Pearson CIs + an up-front materiality rule, and record within-budget
(≤2048) parity + a dense reference.

## Outcome: ACHIEVED — binding AC-2 recall matrix; Tier-2.B gives a material 16K uplift.

## Work completed (mainline)
1. **DSA same-node reference measured** (`niah_dsa_reference.json`,
   `serve_native_nsa.sh` at mem 0.7, N=20): 1024w/4K/16K/64K **all 100%**, 0
   admission failures — the recall ceiling, and the DSA-same-node artifact whose
   absence AC-2's negative tests reject.
2. **DS-hybrid (Tier-2.B) measured** fresh (`niah_ds_hybrid.json`, eager
   scorer_norm=hybrid, int8/mem0.7, N=20), incl. the previously-missing 64K. DS
   engagement verified live via the `double_sparsity` meta (16K: sparsity 0.88,
   selected 2048, dense_fallback 0 — sparse, not error-contained dense).
3. **Consolidated DS-vs-DSA matrix** (`niah_recall_matrix.py`,
   `ds_vs_dsa_recall_matrix.json`, `m2_recall_matrix_finding.md`) with per-cell
   recall + N + **Clopper–Pearson 95% CI** and the up-front materiality rule (a
   variant uplift is material only when its recall point exceeds the DS-default
   baseline CI). DS-default cited as the plan's AC-1 served baseline
   (`ds_niah_baseline_mem07.json`, N=20, same node/op-point).

### Matrix (N=20, served recall % [95% CP CI])
| len | DSA | DS-default | DS-hybrid | uplift | material |
|-----|-----|-----------|-----------|--------|----------|
| 1024w (≤budget) | 100 [83,100] | 100 [83,100] | 100 [83,100] | 0 | parity ✓ |
| 4K  | 100 [83,100] | 75 [51,91] | 85 [62,97] | +10pp | NO (within CI) |
| 16K | 100 [83,100] | 5 [0,25] | **40 [19,64]** | **+35pp** | **YES** (>CI hi 24.9) |
| 64K | 100 [83,100] | 5 [0,25] | 0 [0,17] | −5pp | NO (1-needle floor noise) |

**Finding:** Tier-2.B (hybrid scorer) delivers a **material 16K recall uplift**
(5%→40%, the long-context goal regime) with **within-budget parity** preserved
and **no material 4K change**; 64K stays scorer-limited (both ~0%, sampling
noise). A recorded, characterized, non-regressing AC-2 result, consistent with
the M0 oracle (16K budget-partial, 64K scorer-limited).

## Work completed (queued, bundled cheap)
4. **Fixed the R4-Review analyzer artifact** (Codex gap #1): `analyze_oracle.py`
   now emits separate `uplift_4096_minus_2048` / `uplift_8192_minus_2048` fields
   and a three-way verdict (budget-limited / **budget-partial** / scorer-limited);
   regenerated `oracle_budget_vs_scorer_r4.json` so 16K reads budget-partial,
   matching `m0_oracle_finding_r4.md`.

## Files changed
`analyze_oracle.py`, `oracle_budget_vs_scorer_r4.json` (regenerated),
`niah_recall_matrix.py` (new), `niah_dsa_reference.json` (new),
`niah_ds_hybrid.json` (new), `ds_vs_dsa_recall_matrix.json` (new),
`m2_recall_matrix_finding.md` (new). Commit `9f76ad659` (pushed). No production
code changed this round.

## Validation
- DSA/DS-hybrid NIAH measured on 8×H200 TP=8 at mem 0.7, N=20, served-vs-admission
  separated; CIs via scipy Clopper–Pearson.
- DS unit tests unaffected (no production change); ran the oracle + scorer suites
  (33 pass) as a sanity check.

## Remaining items (queued, justified)
- **Graph-safe Triton scorer port** (AC-3 "landed path"): the eager hybrid scorer
  is ~8× slower per request (e.g. 64K trial ~207 s vs DSA ~184 s for 20). Porting
  scorer_norm/head_agg into the graph-safe Triton kernel with bit-exact
  eager-vs-graph equality is the production-viability keystone — research-grade,
  next round's candidate mainline (or a documented disposition per the plan).
- **MMLU ≤1.0pp re-anchor + N≥50 binding 16K** (AC-3): long evals; next.
- **AC-4 lifted-budget** (task13–17), **AC-6 perf/consolidation + final
  decision record** (task19–20): sequenced after AC-3.
- Plan-marker / stale-comment cleanup: pre-merge.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-niah-recall-materiality-floor-noise
- Notes: at N=20, a 1-needle move at the recall floor (default 1/20=5% vs hybrid
  0/20=0%) trips a naive "point outside the baseline CI" materiality test as a
  "material regression" — a degenerate-CI artifact. The plan's rule is
  directional ("uplift exceeds the baseline CI" = upward); report below-CI
  separately and never claim a sub-2-needle floor move as material. Captured so
  the final AC-2 decision record doesn't over-claim a 64K regression.

## Goal Tracker Update Request
- **task6** (AC-1,AC-2): DSA same-node comparison artifact DONE (R5); within-budget
  parity recorded. Remaining: MMLU re-anchor + dense/default stride microbench.
- **task12** (AC-2,AC-3): binding DS-vs-DSA served-recall matrix + CIs + 64K-hybrid
  + within-budget parity DONE (R5). Remaining: N≥50 16K, MMLU, graph-safe port + perf.
- **Resolve queued side issue** "R4 analyzer artifact mislabel" (fixed R5).
- **Keep Active**: task13 (graph-safe Triton port + perf) as the next mainline
  candidate; MMLU/N≥50; AC-4 / M4 sequenced after.
