# Round 0 Contract

## Mainline Objective
Land the **M0 measure-first oracle diagnostic** instrumentation (AC-1) — the plan's gating first milestone that decides the A-vs-B sequencing. Round-0 begins with the safest, fully CPU-unit-testable brick: a standalone `selection_recall_oracle.py` module implementing the core diagnostic math (needle rank / all-in-topK / score-only recall@K) on the live all-reduced token-score tensor, honoring the DS selector's exact `(score DESC, position ASC)` tie-break.

## Target ACs (1–2)
- **AC-1** (primary): measure-first oracle diagnostic + separated baseline.
- **AC-1.1** (sub): dense-within-window oracle (post-topK replacement) — designed in this round, wiring follows.

## Blocking Side Issues In Scope
- None identified.

## Queued Side Issues Out Of Scope
- The NIAH/MMLU/perf **measurement** runs (task6/task18/task19) and the **analyze** step (task7) require the 8×H200 cluster + serve/bench scripts; they are out of scope for a CPU-only session and run when the loop is on the cluster node. The M0 *instrumentation code* (task1–5) is the prerequisite and is in scope now.
- Tier-2.B (task8–12), Tier-2.A (task13–17), Tier-2.C (task18) are downstream of M0 and the task7 A-vs-B decision; out of scope this round.

## Round Success Criteria
- The standalone oracle diagnostic math module exists with the AC-1 contract (needle_worst_rank, needle_all_tokens_in_topK, score-only recall@K, selected_contains_needle, the `recall@2048 == selected_contains_needle` invariant), implemented off the production hot path.
- A registered CPU unit test exercises the tie-break correctness (score-desc/position-asc), the multi-token worst-rank rule, the K∈{512,1024,2048,4096,8192} curve, and the invariant; it passes.
- Goal Tracker and this contract are initialized; the IMMUTABLE section is set for the loop.
- No production selection/decode path behavior changes (the module is additive and unused by the hot path until the flag-gated hook lands in a later increment).
