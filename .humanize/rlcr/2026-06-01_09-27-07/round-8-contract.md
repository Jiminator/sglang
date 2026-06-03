# Round 8 Contract

## Mainline Objective
**Close AC-1** (move PARTIAL → MET): produce the two durable evidence artifacts
the plan requires and Codex gap #1 specifies —
(1) `oracle_off_graph_replay_alloc.json`: the production graph-safe DS selector
with `recall_oracle=false` under CUDA-graph capture/replay yields **byte-identical
`selected_indices`/`valid_lengths` vs the eager baseline AND zero new replay
allocations** (the "zero hot-path cost" claim, demonstrated not asserted); and
(2) `oracle_stride_reference.json`: the oracle sampling-stride reference — record
the **default oracle stride vs stride=1 (dense)**, proven from the emitted records,
with recall@K per length and the default==stride1 statement.

## Target ACs (1–2)
- **AC-1** (primary): measure-first oracle diagnostic + separated baseline —
  specifically the oracle-off byte-equivalence/zero-alloc-under-replay evidence
  (task4) and the dense/stride reference (task6 remnant).

## Blocking Side Issues In Scope
- None open. The existing `test_cuda_graph_replay_zero_allocations` +
  `test_cuda_graph_100_step_replay_matches_eager` already exercise the mechanism;
  this round produces the DURABLE artifacts (+ a test that emits them) the plan
  demands as evidence.

## Queued Side Issues Out Of Scope (justified)
- **anchor_mode graph-safe port** (AC-3 variant completion): kernel work
  (post-topK force-include with graph-state scratch + replay-equality tests) —
  task #16; AC-3 non-regression for the WINNING hybrid scorer is already
  satisfied (R7), anchor is the exploratory extra variant.
- **AC-4 lifted-budget** (task13–17): the major Tier-2.A workstream; sequenced
  after AC-1 closure.
- **AC-6 perf consolidation + final strategic-gate decision record** (task19–20):
  the end milestone; needs AC-1 + AC-4 first.
- **niah_recall_matrix.py module-docstring "outside CI" wording + MMLU artifact
  metadata enrichment** (Codex queued #1/#2): cheap evidence-hygiene; bundle this
  round if quick.
- Plan-marker cleanup; learned-selector follow-on: pre-merge / out of scope.

## Round Success Criteria
- **`oracle_off_graph_replay_alloc.json`**: capture the production graph-safe DS
  selector (`recall_oracle=false`, the default), replay ≥100 steps; record the DS
  config (dtype/top_k/graph-mode/scorer), the eager-baseline vs replay
  `selected_indices`/`valid_lengths` equality booleans + content hashes, the
  allocation delta from `assert_no_alloc_in_region`, and a pass/fail verdict.
  Backed by a GPU test that emits it.
- **`oracle_stride_reference.json`**: from the R4 oracle sink (or a fresh small
  run), prove the oracle's emitted `stride` field == 1 for all records (dense
  sampling — every needle token, no subsample), state explicitly that the default
  oracle stride IS stride=1 (default==stride1), and include a dense-DS
  within-budget recall reference (≤2048 ⇒ 100%) alongside the default-stride
  beyond-budget served recall, with recall@K/length where available.
- A focused unit/GPU test for the oracle-off equivalence + zero-alloc that writes
  the artifact; all DS unit tests pass; AC-1 evidence consolidated in a short
  finding; committed + pushed; goal-tracker + round-8-summary updated.
