# Ask Codex Input

## Question

Review the Double Sparsity Tier-2.A landing DISPOSITION at development/loop7/m9_tier2a_disposition.md (read it). Context: Loop-7 is closing AC-4 (opt-in adjustable-budget 'lifted-budget' decode). The plan allows AC-4 to close via EITHER production-ready hardening (task16) OR a 'deferred-with-evidence' disposition (task17), per DEC-4/DEC-6. I am choosing deferred-with-evidence. Supporting evidence files: development/loop7/m8_lifted_recall_finding.md (served 4K recall: lifted_budget_top_k=4096 95% vs default top_k=2048 75%, +20pp material, eager same-node N=20), development/loop7/m0_oracle_finding_r4.md (M0 oracle: 4K budget-limited, 16K budget-partial ~46% cap, 64K scorer-limited). Tier-2.B (the landed graph-safe hybrid scorer, AC-3 MET) serves the long-context goal. The lifted path is opt-in, default-off byte-identical, eager-required (validator rejects it without --disable-cuda-graph because dequantize_k_cache_paged allocates), and the default flashmla_kv dsa_index_topk assert is untouched. QUESTIONS: (1) Is deferring task16 (production graph hardening) and closing AC-4 via deferred-with-evidence JUSTIFIED given the M0 bounded-secondary evidence (4K-only recovery; 16K/64K served by Tier-2.B)? (2) Does the disposition satisfy the DEC-4/DEC-6 conditions (recall evidence recorded; DSA default untouched; research path gated out of production CUDA-graph capture)? (3) Is the deferred task16 follow-on scope (alloc-free out= dequant, graph-safe fixed-shape compact remap, q-padding scratch, zero-alloc-replay proof + graph-mode recall re-measure, perf) complete and correctly specified, or is anything missing? (4) Any gap that would make this an INVALID close. Keep it concise and concrete.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-02_16-34-28
- Tool: codex
