**Findings**

- **P1: one task16 checklist item is not explicitly covered.** [m9_tier2a_disposition.md](/sgl-workspace/sglang/development/loop7/m9_tier2a_disposition.md:134) requires graph-captured TP=8 lifted-width selected-index / valid-length equality at 4096 and 8192. The evidence in [m10_lifted_graph_finding.md](/sgl-workspace/sglang/development/loop7/m10_lifted_graph_finding.md:33) proves graph-safe decode replay, and the TP=8 lifted determinism test is still the eager/logical path in [test_ds_scorer_tp_determinism.py](/sgl-workspace/sglang/test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py:178). The live TP=8 graph recall is strong integrated evidence, but it is not the explicit equality proof item 5 asks for.

- **P2: the disposition has stale contradictory prose.** It says production-ready at the top, but still says the sanctioned close is deferred-with-evidence and eager-required in [m9_tier2a_disposition.md](/sgl-workspace/sglang/development/loop7/m9_tier2a_disposition.md:66). Clean that before treating M9 as the final landing record.

- **Scope caveat:** the graph proof is for the fp8-KV/H200 op-point. The BF16 branch in [lifted_budget.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/lifted_budget.py:306) is not covered by the R16/R17 graph tests. That is fine if AC-4 is scoped to the documented fp8 production path; do not claim all validator-supported dtype/backend pairs are graph-proven.

**Answers**

1. **Yes for the stated fp8 production op-point, but not a perfectly clean “all task16 complete” close as written** until item 5 is either proven or explicitly waived/reframed.

2. **Decode graph-safety evidence is sufficient:** real CUDAGraph zero-alloc replay at 4096/8192 plus live `cuda graph: True`, graph-mode 95% recall, perf, and no admission failures is enough for the lifted decode path. Missing only the explicit graph-captured TP=8 selector equality artifact.

3. **No invalidating design gap found** in fixed-shape safe-slot/masked-index or q-padding. The safe slot is masked by `-1` compact indices, and the q pad scratch writes real heads then trims output in [dsa_backend.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py:2195). This relies on the existing selector invariant: valid entries are a prefix and pad tail stays `-1`.

4. **Remaining task16 item:** item 5, strictly. Everything else listed is covered for fp8: alloc-free dequant, fixed-shape compact builder, DSGraphState scratch, q-pad scratch, backend replay proof, validator relax, live graph recall/perf.
