# Round 7 Summary — Loop 7

## Mainline objective (round-7-contract.md)
Produce the **binding AC-3 graph-mode non-regression matrix** for the landed
Tier-2.B hybrid scorer: N≥50 16K served recall (DS-default vs DS-hybrid vs DSA,
all under CUDA graph, same session) + a durable dense-DS within-budget parity
artifact + an MMLU ≤1.0pp re-anchor (DSA vs DS-hybrid).

## Outcome: ACHIEVED — AC-3 non-regression SATISFIED for the hybrid scorer.

## Work completed
1. **Binding graph-mode recall matrix, N=50, 95% Clopper–Pearson CI**
   (`ds_vs_dsa_recall_matrix_graph_n50.json`, `niah_{dsa,default,hybrid}_graph_n50.json`):
   - 1024w **dense-DS / within-budget** (≤2048 tok): DSA/default/hybrid all **100%**.
   - 4K: hybrid **80% == default 80%** (≤8192 ⇒ raw regime; no regression); DSA 100%.
   - 16K: default **6% [1.3,16.5] → hybrid 38% [24.7,52.8] = +32 pp, MATERIAL**
     (the R6 N=20 graph read 25% — a low draw; N=50 binds it at 38%, ≈ eager 40%).
2. **MMLU 5-shot re-anchor, N=200, same questions (deterministic seed), graph-mode**
   (`mmlu_{dsa,default,hybrid}_graph.json`): DSA **89.0%** / default 88.5% /
   hybrid **88.5%** → hybrid **−0.5 pp vs DSA (≤1.0 pp gate PASSED)**; 0 pp vs
   default (MMLU is within-budget ⇒ hybrid uses its raw regime = default).
3. **Fast MMLU runner** (`mmlu_5shot.py`): 5-shot "Answer:" prompt +
   `max_new_tokens=4` (single-letter extraction) — avoids the reasoning model's
   2048-token chains that made `run_eval` (default `max_tokens=2048`) glacial
   (~minutes/question); ~0.25 s/question.
4. **R5 evidence-label cleanup (Codex queued #2)**: DSA JSONs relabeled
   `DSA native-NSA (no double-sparsity)`; `niah_ds_baseline.py` gained `--op-point`;
   `niah_recall_matrix.py` materiality_rule reworded directional ("variant point
   exceeds the DS-default baseline CI high"); matrices regenerated.

## Validation
- All recall + MMLU measured **under CUDA graph** (the production path, per
  `BL-20260602-eager-vs-graph-recall-differs-despite-identical-scorer`), 8×H200
  TP=8, int8/mem0.7, same session. DS engagement implied by the hybrid-vs-default
  16K recall gap (38% vs 6%).
- No production code changed; ran `test_scorer_variants.py` (20 pass) as a sanity.

## AC-3 verdict
**Non-regression SATISFIED** for the hybrid scorer: material long-context (16K)
uplift (6%→38%) + MMLU within 0.5pp of re-anchored DSA + dense-DS/within-budget
parity + no 4K regression + (R3/R6) TP=8 determinism & bit-identical eager-vs-graph
selection. The long-context gap to DSA's 100% is reduced (16K 38%), not closed —
a recorded, characterized result (64K remains scorer-limited per the oracle).

## Files changed
`m4_ac3_nonregression_finding.md` (new), `mmlu_5shot.py` (new),
`ds_vs_dsa_recall_matrix_graph_n50.json` + `niah_*_graph_n50.json` +
`mmlu_*_graph.json` (new data), `niah_ds_baseline.py` (--op-point),
`niah_recall_matrix.py` (directional wording), `niah_dsa_reference.json` +
`ds_vs_dsa_recall_matrix.json` (relabeled/regenerated). Commit `9a37590ec` (pushed).

## Remaining items (queued, justified) — task #16 + others
- **AC-6 graph-vs-eager scorer perf delta** (conc-1/16 TTFT, decode-TPS/req, mem).
- **anchor_mode graph-safe port** (still eager-only).
- **AC-4 lifted-budget** (task13–17): the oracle gate justifies bounded Tier-2.A.
- **AC-1 task4 alloc-detector under graph replay + dense/default oracle-stride
  artifact** (Codex gap #2): contained AC-1 closure.
- **AC-6 consolidation + final strategic-gate supersession decision record** (task20).

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-mmlu-quality-gate-on-reasoning-model
- Notes: a reasoning model (DSv3.2) under the standard `run_eval` MMLU sampler
  (`max_tokens=2048`) generates long chains-of-thought per question → the quality
  gate runs at minutes/question and never finishes. Use a 5-shot "Answer:" prompt
  + `max_new_tokens=4` + leading-letter parse (the AC-12 method) for a fast,
  paired DS-vs-DSA accuracy gate; deterministic example seed gives identical
  questions across servers for an exact paired delta.

## Goal Tracker Update Request
- **task12** (AC-2,AC-3): binding graph-mode recall+MMLU+dense matrix DONE (R7) —
  AC-3 non-regression satisfied. Remaining = perf + anchor port (task #16).
- **Resolve queued side issue** "R5 evidence labels" (fixed R7).
- **Keep Active**: task #16 (AC-6 perf + anchor port + final decision record),
  AC-4 (task13–17), AC-1 task4.
