# Round 16 Review Result

Mainline Progress Verdict: ADVANCED

Round 16 advanced the task16 hardening path by landing the two standalone
graph-safe primitives that R15 asked for: `dequantize_k_cache_paged_out` and the
fixed-shape lifted compact builder. The targeted local review run passed:

```bash
pytest -q test/registered/unit/layers/attention/test_lifted_budget_decode.py::TestLiftedCompactIndexFixed test/registered/unit/layers/attention/test_lifted_budget_decode.py::TestLiftedBudgetGraphSafe::test_dequant_out_matches_allocating
# 2 passed
```

I did not find a high-confidence bug in the new fixed-layout index math. However,
the original-plan task16 is still incomplete. The R16 contract explicitly queued
the backend/cuda-graph-runner wiring, validator relaxation, live graph-mode recall,
perf, and task17 production-ready disposition to a later round. Under this review
prompt, that is unfinished original-plan work, not an acceptable close.

## Mainline Gaps

1. **task16 is only partially implemented; the production lifted decode path is
   still eager-only and still allocation-producing.**

   Evidence:
   - `python/sglang/srt/layers/attention/dsa_backend.py:1978-1987` still routes
     lifted decode to `_forward_lifted_budget` with no graph-state scratch.
   - `python/sglang/srt/layers/attention/dsa_backend.py:2109-2124` still calls
     `build_lifted_compact_kv`, the eager dynamic-`total_valid` helper, not
     `build_lifted_compact_kv_fixed`.
   - `python/sglang/srt/layers/attention/dsa_backend.py:2151-2154` still allocates
     `q_padded = q_all.new_zeros(...)` inside `_forward_flashmla_sparse` when TP
     head count needs FlashMLA padding.
   - `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py:49-79`
     has no lifted decode scratch fields for fixed page table, compact indices,
     compact KV, post-dedup counts, or q head padding.
   - `python/sglang/srt/layers/attention/double_sparsity/validator.py:120-126`
     still rejects lifted-budget decode unless `--disable-cuda-graph` is set.

   Required implementation plan:
   - Extend `DSGraphState` with lifted-budget scratch:
     `lifted_page_table [max_bs * max_top_k] int32`,
     `lifted_compact_indices [max_bs, max_top_k] int32`,
     `lifted_valid_counts [max_bs] int32`,
     `lifted_compact_kv [max_bs * max_top_k, 1, 576] bf16`, and
     `lifted_q_padded [max_bs, required_flashmla_heads, 576] bf16`.
   - Extend `allocate_graph_state` with explicit lifted parameters and allocate
     those tensors only when `enable_lifted_budget_decode` is active. Thread those
     arguments from both metadata allocation sites in `dsa_backend.py`.
   - Change `_forward_lifted_budget` to accept metadata/graph state. When lifted
     scratch is present, slice the scratch to the current `bs` and width, call
     `build_lifted_compact_kv_fixed`, and pass the scratch compact KV/indices into
     FlashMLA. Keep the current eager helper only as the non-graph fallback.
   - Remove `_forward_flashmla_sparse`'s graph-path `new_zeros` by adding an
     optional q-padding scratch path: copy real heads into preallocated
     `lifted_q_padded`, zero or overwrite the padded tail deterministically, call
     `flash_mla_sparse_fwd`, and return the real-head view.
   - Add backend-level CUDAGraph tests at 4096 and 8192 that exercise the wired
     lifted branch, including q head padding, prefix sharing, duplicate physical
     slots, pad lanes, and `valid_lengths < width`. Wrap replay in
     `assert_no_alloc_in_region` and compare to the eager/reference output.
   - Add the graph-captured TP=8 lifted-width determinism test required by the R15
     review: selected-index and valid-length equality across ranks at 4096 and
     8192 under CUDA graph capture, not only the eager/logical path.
   - Only after those tests pass, remove the validator's lifted
     `--disable-cuda-graph` rejection. Keep the default `flashmla_kv`
     `indices.shape[-1] == dsa_index_topk` assert untouched.

2. **task17 must still be redone as a production-ready disposition.**

   The current `development/loop7/m9_tier2a_disposition.md` is correctly marked
   superseded, but it is still a deferred-only record. After task16 is actually
   wired into production graph capture, write the replacement disposition with:
   graph-safety evidence, zero-alloc backend replay, graph-mode 4K served recall,
   graph-captured TP=8 determinism, perf/memory impact, validator status, and
   default-path non-regression.

3. **AC-6 task19 remains active.**

   Use the existing Loop-7 serve/benchmark tooling at the int8/mem0.7/fp8-KV/TP=8
   op-point. Record DS default, graph-safe DS hybrid, DSA, and production-hardened
   lifted DS where applicable. Capture conc-1 and conc-16 TTFT, decode TPS/req,
   GPU memory, graph replay status, admission behavior, radix/cache assumptions,
   exact server args, DS config, commit, GPU type, and artifact paths. Produce the
   consolidated recall/perf/non-regression report.

4. **AC-2 task20 remains active.**

   After task19, write the final strategic-gate supersession decision record. It
   must cite M0 regime attribution, AC-1 closure, AC-3 hybrid scorer evidence,
   AC-4 production-ready lifted disposition, AC-5 servability, and AC-6 perf
   guardrails, and explicitly state what measured evidence superseded the Loop-6
   Tier-2.A-primary ordering.

## Blocking Side Issues

None. The current runtime remains safe because the validator still blocks lifted
decode from CUDA graph capture. The problem is incomplete mainline work, not an
unsafe default production path.

## Queued Side Issues

1. Preserve or cite the R8 oracle-sink provenance before task20, or cite the
   hardcoded `stride=1` call site plus the committed aggregate explicitly.
2. Remove plan/workflow markers from production code/comments/tests before final
   cleanup/merge.
3. Learned/distilled selector work remains out of scope unless explicitly approved
   under DEC-5.

## Goal Alignment Summary

ACs: 6/6 addressed | Forgotten items: 0 | Unjustified deferrals: 1

| AC | Status | Review result |
|----|--------|---------------|
| AC-1 | MET | Prior R8 oracle/fail-closed/stride/zero-hot-path evidence remains accepted. |
| AC-2 | PARTIAL | Recall uplift evidence exists, but task20 final supersession record is still missing. |
| AC-3 | MET | Graph-safe non-learned scorer/head/anchor variants and non-regression matrix remain accepted. |
| AC-4 | PARTIAL / NOT MET | R16 advanced task16 primitives, but backend graph integration, validator relax, live graph-mode recall/perf, graph-captured TP=8, and task17 redo remain. |
| AC-5 | MET | 64K servability at mem0.7 remains verified; 128k remains out of scope. |
| AC-6 | PARTIAL | task16 backend graph/perf hardening and task19 final perf guardrails remain missing. |

## Goal Tracker Update Requests

No tracker edit was needed during this review. `goal-tracker.md` Plan Version 22
already records task16 as in progress with R16 primitives done and backend
integration/validator/live measurement remaining; task17, task19, and task20 are
still active; the Explicitly Deferred table is empty.

PENDING
