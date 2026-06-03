# Round 8 Summary — Loop 7

## Mainline objective (round-8-contract.md)
**Close AC-1** (PARTIAL → MET): produce the two durable evidence artifacts the
plan requires (Codex R7 gap #1) — oracle-off byte-identical + zero-alloc under
CUDA-graph replay, and the dense/default oracle-stride reference.

## Outcome: ACHIEVED — AC-1 is MET.

## Work completed
1. **`oracle_off_graph_replay_alloc.json`** (+ `oracle_off_replay_alloc.py`):
   with `recall_oracle=false` (default), the production graph-safe DS selector is
   captured under CUDA graph and replayed 120 steps —
   `replay_indices/lengths_byte_identical_to_eager: true` (eager & replay share
   `selected_indices` sha `87426fc4`), `replay_allocation_delta_bytes: 0`,
   `replay_zero_new_allocations: true`, **verdict PASS**. The "zero hot-path cost"
   claim is **demonstrated, not asserted**.
   - **CI backing**: new GPU test
     `test_double_sparsity_unit.py::test_oracle_off_replay_byte_identical_and_zero_alloc`
     (asserts `sel.config.recall_oracle is False` + byte-identical replay + zero
     alloc), beside the pre-existing `test_cuda_graph_100_step_replay_matches_eager`
     and `test_cuda_graph_replay_zero_allocations`.
2. **`oracle_stride_reference.json`** (+ `oracle_stride_reference.py`): the
   oracle's emitted `stride` field is **1 for all 14,640 R4 success records**
   (hook hardcodes `stride=1` — dense sampling of every needle token, no
   subsample) ⇒ **`default_equals_stride1: true`**, proven from records; plus the
   **dense-DS within-budget** reference (1024w ≤2048 tok ⇒ DS selects densely ⇒
   default & hybrid both 100%) next to the default-stride beyond-budget served
   recall (4K 80%, 16K default 6% / hybrid 38%) + per-length score-only recall@K.
3. **Bundled queued cleanups** (Codex queued #1/#2): `niah_recall_matrix.py`
   module docstring made directional ("exceeds the baseline CI high"); the three
   MMLU artifacts enriched with `op_point` / `graph_mode` / `example_seed` /
   `data_dir` metadata (+ the runner now emits it).

## AC-1 verdict: MET
Oracle records the required per-trial fields on the live all-reduced score tensor,
fail-closed, dedicated sink (R1–R4); **oracle-off byte-identical + zero-alloc
under graph replay — demonstrated (R8)**; separated served-vs-admission baseline
at mem 0.7 with the **stride=1 dense reference (R8)**; AC-1.1 post-topK replacement
(R1/task5). All sub-criteria have committed evidence.

## Validation
- `oracle_off_replay_alloc.py` → PASS (byte-identical + 0 alloc bytes), 8×H200.
- **326 DS unit tests pass** including the new oracle-off replay test + the
  existing 100-step replay + zero-alloc tests.

## Files changed
`oracle_off_replay_alloc.py` (new), `oracle_off_graph_replay_alloc.json` (new),
`oracle_stride_reference.py` (new), `oracle_stride_reference.json` (new),
`m5_ac1_closure_finding.md` (new), `test_double_sparsity_unit.py` (new GPU test),
`niah_recall_matrix.py` (docstring), `mmlu_5shot.py` + `mmlu_{dsa,default,hybrid}_graph.json`
(metadata). Commit `f05cb730e` (pushed). No production runtime code changed.

## Remaining items (queued, justified)
- **AC-3 anchor_mode graph-safe port** + **AC-6 graph-vs-eager perf delta** (task #16).
- **AC-4 lifted-budget** (task13–17): the oracle gate justifies bounded Tier-2.A.
- **AC-6 consolidation + final strategic-gate supersession decision record** (task20).

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: this round assembled durable AC-1 artifacts from existing mechanisms
  (the graph-replay alloc detector + the oracle stride field) — no new reusable
  engineering pitfall surfaced.

## Goal Tracker Update Request
- **task4** (AC-1): oracle-off zero-hot-path DEMONSTRATED (R8) → done.
- **task6** (AC-1,AC-2): dense/stride reference DONE (R8); DSA/MMLU already done →
  done.
- **AC-1 → MET.**
- **Keep Active**: task #16 (anchor port + AC-6 perf + decision record), AC-4
  (task13–17), task20.
