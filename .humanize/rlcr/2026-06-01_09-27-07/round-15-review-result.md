# Round 15 Review Result

Mainline Progress Verdict: STALLED

Round 15 produced a useful Tier-2.A disposition record, and I did not find a
factual implementation-claim bug in that record. However, this review prompt
explicitly overrides the plan's permissive deferred-with-evidence branch: deferred
original-plan work is incomplete and must be forced to completion. On that basis,
R15 does not close AC-4. `task16` remains unimplemented and `task17` must be redone
as a production-ready landing record after hardening lands.

## Mainline Gaps

1. **AC-4 is not complete because task16 was deferred instead of implemented.**

   Evidence:
   - `development/loop7/m9_tier2a_disposition.md` explicitly defers task16 and
     labels the lifted-budget path as eager/default-off research.
   - `python/sglang/srt/layers/attention/dsa/dequant_k_cache.py:195` still
     allocates `output = torch.empty(...)` inside `dequantize_k_cache_paged`.
   - `python/sglang/srt/layers/attention/double_sparsity/lifted_budget.py:162`
     builds `page_table_1_flattened` via boolean masking, and
     `:164-165` performs Python scalar extraction; this is the dynamic eager
     compact-builder shape, not a fixed-shape graph-safe path.
   - `python/sglang/srt/layers/attention/double_sparsity/lifted_budget.py:212`
     calls the allocating `dequantize_k_cache_paged` directly.
   - `python/sglang/srt/layers/attention/double_sparsity/validator.py:120-126`
     still rejects lifted-budget decode unless `--disable-cuda-graph` is set.
   - `python/sglang/srt/layers/attention/dsa_backend.py:2093-2124` routes lifted
     decode through `_forward_lifted_budget`, but that method delegates to the
     eager compact/dequant path above.

   Required implementation plan:
   - Add `dequantize_k_cache_paged_out(quant_k_cache, page_table_1_flattened, out,
     group_size=128)` in `dsa/dequant_k_cache.py`. It must validate dtype/shape,
     invoke the existing Triton kernel with caller-owned bf16 output, and let the
     existing allocating API become a thin wrapper around the `out=` variant.
   - Add a graph-safe lifted compact builder that writes into preallocated scratch:
     `page_table_1_flattened_scratch [max_bs * lifted_budget_top_k]`,
     `compact_indices_scratch [max_bs, lifted_budget_top_k]`,
     `valid_counts_scratch [max_bs]`, and
     `compact_kv_scratch [max_bs * lifted_budget_top_k, 1, 576]`. Invalid and
     duplicate lanes must write a safe physical slot into the dequant input and
     `-1` into `compact_indices`; no `-1` may reach dequant.
   - Preallocate q head-padding scratch in the backend/cuda-graph state and route
     the lifted branch through scratch-backed graph code when CUDA graph capture is
     enabled. Keep the eager path available for tests and non-graph runs.
   - Prove equivalence and graph safety: `dequantize_k_cache_paged_out` vs current
     dequant, fixed-shape compact builder vs eager remap on prefix-sharing,
     duplicate, pad, and `valid_lengths` cases, backend lifted decode at 4096/8192
     vs the existing reference, real `torch.cuda.CUDAGraph` capture/replay with
     `assert_no_alloc_in_region`, and graph-captured TP=8 selected-index equality
     at 4096 and 8192.
   - Only after those pass, relax the validator's `--disable-cuda-graph` rejection
     for lifted decode. Keep the default `flashmla_kv` `dsa_index_topk` assert
     untouched.
   - Re-measure graph-mode served 4K lifted recall and record perf/memory impact
     before rewriting the Tier-2.A landing disposition.

2. **task17 is not accepted as the final AC-4 close.**

   The R15 disposition is useful source evidence, but it is a deferred-only close.
   After task16 lands, write a revised production-ready disposition that records:
   graph-safe lifted decode evidence, zero-alloc replay, graph-mode recall, TP=8
   captured determinism, perf/memory results, validator status, and default-path
   non-regression.

3. **AC-6 task19 remains active.**

   Required implementation plan:
   - Use the existing Loop-7 serve/benchmark tooling, not new scaffolding.
   - At the selected int8/mem0.7/fp8-KV/TP=8 op-point, record DS default, graph-safe
     DS hybrid, DSA, and production-hardened lifted DS where applicable.
   - Capture conc-1 and conc-16 TTFT, decode TPS/req, GPU memory, graph replay
     status, admission behavior, radix/cache assumptions, exact server args, DS
     config, commit, GPU type, and artifact paths.
   - Produce the consolidated DS-vs-DSA recall/perf/non-regression report.

4. **AC-2 task20 remains active.**

   Required implementation plan:
   - After task19, write the final strategic-gate supersession decision record.
   - Cite the M0 regime attribution, AC-1 closure evidence, AC-3 graph-safe hybrid
     scorer evidence, AC-4 production-ready lifted disposition, AC-5 servability,
     and AC-6 perf guardrails.
   - The record must explicitly supersede the Loop-6 Tier-2.A-primary ordering by
     stating what changed in the measured evidence.

## Blocking Side Issues

None. The current lifted path is safely blocked from CUDA graph capture by the
validator, so the issue is incomplete mainline work rather than an unsafe live
production bug.

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
| AC-4 | PARTIAL / NOT MET | task13-task15 remain accepted; task16 is deferred and therefore incomplete under this review contract; task17 must be redone after task16. |
| AC-5 | MET | 64K servability at mem0.7 remains verified; 128k remains out of scope. |
| AC-6 | PARTIAL | task16 graph/perf hardening and task19 final perf guardrails remain missing. |

## Goal Tracker Update Requests

I updated `goal-tracker.md` directly:

- bumped Plan Version to 21 for Round 15 Review;
- added a Round 15 Review plan-evolution row rejecting the task16 deferral under
  this review prompt's stricter rule;
- reactivated task16 and task17 under Active Tasks;
- removed the task16 Explicitly Deferred entry and left the deferred table empty;
- removed task17 from Completed and Verified because the deferred-only disposition
  is not accepted as the final AC-4 close.

Rejected tracker change: Claude's R15 update marking AC-4 MET and task16
Explicitly Deferred is rejected for this review because the prompt says deferred
original-plan work is incomplete and must be completed now.

PENDING
