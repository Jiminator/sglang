---
id: ds-score-reduce-dispatch
type: knowledge
title: DS score-reduce collective dispatch boundary (why it lands on NCCL ring)
status: active
created: 2026-06-10
updated: 2026-06-10
tags: [pensieve, knowledge, double-sparsity, all-reduce, cuda-graph]
---

# DS score-reduce collective dispatch boundary

Verified 2026-06-10 during Loop-9 plan convergence (code reading + nsys evidence from
`development/profiling/runs/20260609/`). Reusable for any work touching Double Sparsity
collectives or per-layer decode communication.

## Symptom -> root cause -> location

- **Symptom:** DS-on decode pays 780 `ncclDevKernel_AllReduce_Sum_f32_RING` calls per 10-step
  window (+124,873 us vs DSA at same batch), even though the server runs with custom-all-reduce ON
  and the model's hidden-state reduces use the fast IPC path.
- **Root cause:** the DS bind site passes the RAW `torch.distributed` ProcessGroup
  (`get_attention_tp_group().device_group`) into the selector, so DS score reduces call bare
  `torch.distributed.all_reduce` and never reach the `GroupCoordinator.all_reduce` dispatch that
  owns custom-AR selection.
- **Locations:**
  - Bind site: `python/sglang/srt/models/deepseek_v2.py` (`_setup_double_sparsity_bind` ->
    `bind_runtime_data(process_group=get_attention_tp_group().device_group)`).
  - Reduce call sites (TWO — both must be covered by any change):
    `all_reduce_token_scores` helper (eager paths) AND a direct
    `torch.distributed.all_reduce(scores_view, ...)` inside the graph-safe selector path, both in
    `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`.

## Boundary facts (verified, save a re-exploration)

- SGLang custom all-reduce: IPC one-shot/two-shot kernels, ~8 MB buffer cap, world sizes
  2/4/6/8; dispatch lives behind `GroupCoordinator.all_reduce`
  (`python/sglang/srt/distributed/`), NOT behind `torch.distributed.all_reduce`.
- Custom AR is OUT-OF-PLACE; DS's graph-safe path assumes in-place mutation of its scratch score
  view — a re-route needs a captured copy-back (zero replay allocations).
- flashinfer allreduce fusion (`flashinfer_comm_fusion.py`) exposes ONLY
  allreduce+residual+RMSNorm; it cannot host a standalone SUM.
- Cross-layer batching of the score reduce is structurally infeasible: each layer's top-k
  consumes that layer's reduced scores before the next layer runs.
- KV pool is sized at `ModelRunner.init_memory_pool()` BEFORE the DS `TokenLabelTable` binds;
  freeing DS scratch does NOT automatically raise the decode-batch admission cap.
- `selection_recall_oracle.py` is a host-side NIAH recall diagnostic — it is NOT an
  implementation-equivalence checker for selected indices.

## Anti-pattern

Calling `torch.distributed` collectives directly from model/attention code bypasses every
SGLang communication optimization; route collectives through the bound `GroupCoordinator` (or an
explicit callable from it) instead of storing a raw `device_group`.
