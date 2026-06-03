# Ask Codex Input

## Question

Re-review the corrected Double Sparsity Tier-2.A PRODUCTION-READY disposition at development/loop7/m9_tier2a_disposition.md (read it fully) — the R17 review flagged it as internally contradictory and required a graph-captured TP=8 lifted-width selector-equality artifact. Changes since: (1) every section now describes the R17 production graph state — validator NO LONGER requires --disable-cuda-graph, launcher no longer forces eager, dequantize_k_cache_paged_out + fixed-shape scratch is the production decode path, graph-mode 4K NIAH 95% is the binding production recall; all 'eager-required/deferred' claims removed. (2) The graph-captured TP=8 lifted-width determinism (item 5) is now PROVEN by three composed artifacts: (a) single-rank retrieve_topk_graph_safe at 4096/8192 captured in a real torch.cuda.CUDAGraph, zero-alloc replay + bit-identical to eager (TestLiftedWidthSelectionGraphCaptured); (b) eager 8-rank all-reduce equality at 4096/8192 (TestTP8LiftedWidthDeterminism); (c) the live R17 TP=8 server ran the selection under production CUDA graph + served 95% recall. I documented that a STANDALONE 8-rank NCCL-collective-under-raw-torch.cuda.graph unit harness is INFEASIBLE (it deadlocks — NCCL collective capture needs the production cuda_graph_runner coordination, which (c) exercises live). Also added a fail-closed validator guard rejecting lifted + speculative decode. QUESTIONS: (1) Is m9 now internally consistent (no remaining deferred/eager-required contradictions)? (2) Is the (a)+(b)+(c) composed evidence + the documented NCCL-capture-harness infeasibility an ACCEPTABLE close of the graph-captured TP=8 requirement, or is there a feasible standalone artifact I am missing? (3) Any remaining contradiction or gap blocking the production-ready AC-4 close. Keep it concise.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-02_18-49-10
- Tool: codex
