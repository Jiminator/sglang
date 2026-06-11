CANDIDATE_ASSESSMENTS:
- **A wrapper**: Correct and deterministic on recorded fixtures; repair enforces score-desc/pos-asc, ascending emission, `-1` padding, and `valid_lengths`. Not graph-safe as written because it allocates full-width intermediates and sorts/gathers per call. Perf loses decisively: `1530.1 us` eager vs B-Triton `52.6 us` captured. Integration risk is high for production decode because exactness is bought by full-width torch repair.
- **A raw fast_topk_v2**: Fast floor at `17.7 us`, but fails the hard gate: boundary ties are nondeterministic from the atomic admission race. Disqualified regardless of speed.
- **B-Triton shipped path**: Best production balance. Correct/deterministic on adversarial fixtures and graph replay tests; integrated behind `retrieve_topk_graph_safe`; zero replay allocations with caller-owned scratch. Perf: `52.6 us` op point, `36.1 us` at 16k, `440.9 us` all-live. Defect: strict “num_finite” contract is not fully pinned because Triton treats `+inf` as selectable in [topk_kernel.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/topk_kernel.py:90), while AOT uses `isfinite`.
- **B-AOT**: Correct/deterministic on fixtures, true single launch after in-kernel `-1` init, no scratch, graph-safe. Perf wins op point (`43.2 us` vs `52.6 us`) but regresses long contexts (`71.0 us` at 16k, `629.3 us` all-live) because one block per row underfills 132 SMs at `bs=29`. Source/binding/tests are present; wheel build succeeded, but the wheel is intentionally uninstalled.

WINNER: **B-Triton**, with the decision rule: pass correctness + cross-rank determinism + graph-safety first, then choose the best captured-shape profile across op point and long contexts, not the single fastest op-point median.

AOT_DISPOSITION: **keep-uninstalled-followon**. The op-point win is not worth integrating a long-context regression into production decode; keep it as source-complete groundwork for a fused multi-block redesign.

REQUIRED_CORRECTIONS:
- Add/repair strict non-finite fixtures for `NaN` and `+inf`; either make B-Triton exclude all non-finite values or narrow the documented contract if `+inf` is impossible by construction.
- Update the reference/tests: `select_topk_sequence_order` only invalidates `-inf`, so it does not currently prove the documented `num_finite` contract.
- Append symbol/schema probe output to `sgl_kernel_build.log` or soften the benchmark record’s claim that the log verifies symbol/schema registration.

VERDICT: **needs-fixes**
