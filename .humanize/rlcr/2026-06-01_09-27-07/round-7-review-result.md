# Round 7 Review Result

Mainline Progress Verdict: ADVANCED

Claude advanced the Round-7 contract. The graph-mode AC-3 non-regression matrix
is present, internally consistent, and reproducible from the committed inputs:
rerunning `development/loop7/niah_recall_matrix.py` against
`niah_{dsa,default,hybrid}_graph_n50.json` reproduced
`ds_vs_dsa_recall_matrix_graph_n50.json` with no diff. The binding 16K result is
DS-default 3/50 = 6% [1.3,16.5] vs DS-hybrid 19/50 = 38% [24.7,52.8], so the
hybrid point exceeds the DS-default CI high and is material by the directional
rule. The MMLU artifacts also support the claimed <=1.0pp gate: DSA 178/200
(89.0%) vs hybrid 177/200 (88.5%).

Do not close Loop 7. Round 7 deliberately excluded original-plan acceptance
work: AC-1 allocation/stride closure, the graph-safe anchor-budget port, AC-4
lifted-budget decode, AC-6 perf/consolidation, and task20 remain active. These
are not optional queue cleanup.

## Implementation Review

Accepted R7 work:

1. **The graph-mode recall matrix is real.**

   Evidence: `development/loop7/ds_vs_dsa_recall_matrix_graph_n50.json` records
   DSA/default/hybrid at 1024w, 4K, and 16K, with served=50 and
   admission_fail=0 for every cell. The 1024w row is 50/50 for all configs, the
   4K row is default=hybrid=40/50, and the 16K row is default=3/50 vs
   hybrid=19/50. The op-point labels distinguish DSA native-NSA from DS int8.

2. **The matrix script now emits the correct directional materiality rule in the
   JSON output.**

   Evidence: `niah_recall_matrix.py` sets
   `hybrid_material_uplift_vs_default_CI` only when `h > base_hi`, and the
   generated graph matrix marks only the 16K row material.

3. **The MMLU re-anchor satisfies the Round-7 gate.**

   Evidence: `mmlu_dsa_graph.json`, `mmlu_default_graph.json`, and
   `mmlu_hybrid_graph.json` record N=200, with DSA 89.0%, default 88.5%, and
   hybrid 88.5%. The observed hybrid-vs-DSA delta is -0.5pp.

4. **The R5 evidence-label cleanup is mostly done.**

   Evidence: `niah_dsa_reference.json` and `niah_dsa_graph_n50.json` now label
   the DSA op point as native-NSA/no double-sparsity, and the matrix JSON
   materiality prose is directional.

## Mainline Gaps

1. **AC-1/task4 and the dense/default oracle-stride reference are still
   incomplete.**

   Evidence: the tracker still has task4 active for oracle-off CUDA graph replay
   allocation evidence, and task6 active for the dense/default stride reference.
   Repository search found no committed artifact showing oracle-off
   byte-identical `selected_indices` / `valid_lengths` plus zero replay
   allocations. The R4 aggregate records score-only recall, but it does not close
   the required side-by-side dense/default stride reference.

   Required implementation plan:
   - Add a durable AC-1 artifact, `development/loop7/oracle_off_graph_replay_alloc.json`.
   - Use the existing production graph-safe DS selector path with
     `recall_oracle=false`, capture CUDA graph replay, compare baseline vs
     oracle-disabled `selected_indices` and `valid_lengths` byte-for-byte, and
     wrap replay in `assert_no_alloc_in_region`.
   - Save the DS config, dtype, top_k, graph mode, output hashes, equality
     booleans, allocation delta, and pass/fail verdict.
   - Add the dense/default stride reference as
     `development/loop7/oracle_stride_reference.json`: run/analyze the oracle
     with the current default sampling stride and an explicit stride=1 run, record
     trial counts, observed stride values, recall@K for each length, and deltas.
     If the implementation currently hardcodes stride=1, the artifact must state
     that fact and prove default=stride1 from the emitted records.

2. **AC-3 remains incomplete for the anchor-budget variant: `anchor_mode` is
   still eager-only under CUDA graph.**

   Evidence: `ds_scorer_is_graph_safe()` returns false for any non-default
   `anchor_mode`; `validate_double_sparsity()` rejects non-default anchor modes
   under CUDA graph; and `capture_decode_step()` still raises because capturing
   would silently drop the anchor force-include. Round 7 explicitly queued this
   work in `m4_ac3_nonregression_finding.md`.

   Required implementation plan:
   - Extend `GraphState` with preallocated anchor scratch sized by
     `[max_bs, max_top_k]` and any fixed-shape candidate/merge buffers needed for
     post-topK replacement.
   - Implement graph-safe post-topK force-include for
     `anchor_mode={recency,global,strided}` after the graph-safe topK selection:
     preserve exactly `top_k`, clamp over-budget anchor counts as in R3, evict the
     lowest-ranked non-anchor selected positions, keep `-1` padding safe, and
     avoid Python loops, `.item()`, host syncs, and per-replay allocation.
   - Thread `anchor_mode` / `anchor_budget` through `retrieve_topk_graph_safe`,
     `deepseek_v2.py`, and `cuda_graph.py`.
   - Add eager-vs-graph equality tests over
     `scorer_norm={off,cosine,hybrid} x head_agg={max,mean} x
     anchor_mode={off,recency,global,strided}` plus CUDA-graph replay
     no-allocation tests.
   - Only after those tests pass should the validator stop rejecting non-default
     anchor modes under CUDA graph.

3. **AC-4/task13-task17 are still unimplemented even though the oracle gate
   justifies bounded Tier-2.A work.**

   Evidence: `DoubleSparsityConfig` has no `enable_lifted_budget_decode` or
   `lifted_budget_top_k` fields, and the validator still only has the old
   `SGLANG_DS_ALLOW_TOPK_MISMATCH` escape for `top_k != index_topk`. No compact
   remap lifted-budget path, correctness tests, or task17 disposition record
   exists.

   Required implementation plan:
   - Complete task13 first as a design/disposition document that defines the
     exact ABI: `enable_lifted_budget_decode: bool` and
     `lifted_budget_top_k: int`.
   - Add those fields to `DoubleSparsityConfig`, reject Twilight fields as today,
     and reject `top_k > index_topk` unless the new opt-in lifted-budget backend
     path is selected. Do not use `max_top_k` or
     `SGLANG_DS_ALLOW_TOPK_MISMATCH` as the mechanism.
   - Implement the opt-in decode path with `flash_mla_sparse_fwd` plus
     `dequantize_k_cache_paged`: selected physical slot ->
     `page_table_1_flattened` -> request-local compact KV index.
   - Mask/safe-replace `-1` padding before any dequant/index operation, keep
     fixed `lifted_budget_top_k` shapes with padding, and preserve the R23
     deterministic tie-break.
   - Add reference sparse-attention tolerance tests, prefix-sharing compact-remap
     tests, padding/duplicate/valid-length tests, TP=8 equality at 4096/8192,
     and graph replay allocation evidence.
   - Finish task17 with a landing/disposition record: production-ready landed
     path, or explicit hardening follow-on with recall evidence and the DSA
     default untouched.

4. **AC-6/task19 and task20 final consolidation are still missing.**

   Evidence: no Round-7 artifact records conc-1/16 TTFT, decode TPS/req, GPU
   memory, graph replay success, admission status, or Tier-1 spine
   non-regression for the chosen graph-mode hybrid path. The final
   strategic-gate supersession record is still not written; `m0_decision.md` is
   useful source text but not the final task20 artifact.

   Required implementation plan:
   - After AC-1 closure and task17 disposition exist, run the existing
     `development/benchmark.sh` / comparison tooling at the Loop-7 op-point for
     DS-default, DS-hybrid graph-safe, and DSA.
   - Record conc-1 and conc-16 TTFT, decode TPS/req, GPU memory, graph replay
     status, admission, radix/cache assumptions, and exact server configs.
   - Write the consolidated DS-vs-DSA recall/perf/non-regression report.
   - Write the final decision record that supersedes the Loop-6 strategic gate's
     Tier-2.A-primary ordering with the final M0/R4/R7 evidence and the AC-4
     disposition.

## Blocking Side Issues

None found in the Round-7 matrix artifacts themselves. The R7 recall/MMLU
evidence advances the current contract. The blockers to Loop-7 closure are the
mainline gaps above, not new runtime bugs in the R7 measurement scripts.

## Queued Side Issues

1. `development/loop7/niah_recall_matrix.py` still has a stale module docstring
   saying materiality means "outside" the baseline CI, while the executable JSON
   rule is directional upward. This is evidence hygiene before final task20, not
   a matrix correctness bug.
2. The MMLU JSON artifacts are minimal: they record label, N, hits, score, and
   elapsed time, but not op-point, graph mode, data-dir hash, or question IDs.
   The finding doc supplies the context, but task20 should either embed that
   metadata or regenerate richer MMLU artifacts.
3. Plan/workflow markers in production comments/tests remain cleanup work before
   final merge.
4. Learned/distilled selector work remains out of scope unless explicitly
   approved.

## Goal Alignment Summary

ACs: 5/6 addressed | Forgotten items: 0 | Unjustified deferrals: 5

AC status:

| AC | Status | Review result |
|----|--------|---------------|
| AC-1 | PARTIAL | Oracle diagnostic/fail-closed records are implemented, but oracle-off CUDA graph allocation evidence and dense/default stride closure remain active. |
| AC-2 | PARTIAL/ADVANCED | DS-vs-DSA recall evidence now includes the graph-mode N=50 matrix and MMLU re-anchor. Final closure still depends on AC-4 disposition, AC-6 consolidation, and task20. |
| AC-3 | PARTIAL/ADVANCED | The selected hybrid scorer's graph-mode non-regression matrix is accepted. Anchor-budget graph-safe support remains missing. |
| AC-4 | NOT MET | No lifted-budget ABI, compact remap, decode implementation, tests, or disposition record. |
| AC-5 | MET | 64K servability at mem0.7 remains verified with served/admission separated. |
| AC-6 | PARTIAL | No final conc-1/16 perf guardrails or Tier-1 non-regression report. |

The `Explicitly Deferred` table remains empty, but Claude's Round-7
"queued/out of scope" list includes original-plan acceptance work. Those items
must drive subsequent rounds, not be treated as optional cleanup.

## Goal Tracker Update Requests

I updated the mutable tracker section:

- bumped Plan Version to 9 for Round 7 Review;
- added a Round 7 Review plan-evolution row;
- moved task12 to Completed and Verified with the R7 graph-mode matrix evidence;
- corrected task6/task7 notes so MMLU and N>=50 are no longer listed as pending;
- added an active AC-3 anchor graph-safe follow-up so the eager-only anchor gap
  remains visible;
- kept task4/task6/task7, task13-task17, task19, and task20 active/pending.

Not complete: original-plan acceptance work remains active.
