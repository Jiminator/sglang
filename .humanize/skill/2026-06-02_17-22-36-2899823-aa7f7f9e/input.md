# Ask Codex Input

## Question

Review the Double Sparsity Tier-2.A PRODUCTION-READY landing disposition at development/loop7/m9_tier2a_disposition.md (read it) and its evidence at development/loop7/m10_lifted_graph_finding.md. Context: Loop-7 AC-4 (opt-in adjustable-budget 'lifted-budget' decode). The R15 review REJECTED the earlier deferred-with-evidence close and required task16 production hardening to be IMPLEMENTED. It now is: (R16) alloc-free dequantize_k_cache_paged_out + a fixed-shape graph-safe compact builder (build_lifted_compact_kv_fixed, no dynamic total_valid) proven zero-alloc under a standalone CUDAGraph; (R17) wired into DSGraphState lifted scratch + allocate_graph_state + _forward_lifted_budget graph path + a q head-padding scratch + _forward_flashmla_sparse q_pad param; validator's --disable-cuda-graph rejection removed; the wired backend _forward_lifted_budget replays zero-alloc under a real CUDAGraph at 4096/8192 (TestLiftedBudgetBackendGraphSafe). LIVE: a server booted WITHOUT --disable-cuda-graph, the full forward captured ('cuda graph: True' decode batches), and graph-mode NIAH 4K N=20 = 95% (matches the eager 95%, +20pp over default-2048 ~75%), 3.4x faster than eager, 0 admission fails, the default flashmla_kv dsa_index_topk assert untouched, default-off byte-identical, 347 DS unit tests pass. QUESTIONS: (1) Does this satisfy the PRODUCTION-READY AC-4 close (DEC-4)? (2) Is the graph-safety evidence (offline zero-alloc replay at 4096/8192 + live 'cuda graph: True' decode + graph-mode recall) sufficient, or is anything missing? (3) Is there any correctness or graph-safety gap in the fixed-shape-with-safe-slot/masked-index design or the q-padding-scratch (pad tail stays 0, trimmed) that would invalidate the close? (4) Any remaining task16 item not covered. Keep it concise and concrete.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-02_17-22-36
- Tool: codex
