# Ask Codex Input

## Question

Please review development/loop7/m12_final_decision.md (the Loop-7 final strategic-gate supersession decision record, the loop-close artifact for an RLCR loop on DeepSeek-V3.2 FP8 Double-Sparsity long-context recall). Verify, against the cited committed artifacts in development/loop7/, that: (1) the M0 regime attribution (4K budget-limited / 16K budget-partial ~46% cap / 64K scorer-limited) matches oracle_stride_reference.json + m0_oracle_finding_r4.md; (2) the AC-2/AC-3 numbers (16K 6%->38% +32pp material with Clopper-Pearson CIs; MMLU DSA 89.0/default 88.5/hybrid 88.5, -0.5pp) match ds_vs_dsa_recall_matrix_graph_n50.json / m4_ac3_nonregression_finding.md; (3) the supersession logic is sound — does the evidence actually justify demoting Tier-2.A (wider budget) from primary to a bounded 4K-only lever and promoting Tier-2.B (selector) to primary for long-context? (4) the record explicitly states what changed from the Loop-6 gate ds_on_v32_decision.md and that the prior rationale was sound when written; (5) the R8 stride/oracle provenance is cited explicitly (committed oracle_stride_reference.json + the selection_kernel.py stride=1 call site, with the raw sink noted gitignored); (6) DEC-4 close-gate (the m9 production-ready Tier-2.A disposition exists) is satisfied. Report ONLY high-signal invalidating issues (factual mismatch with an artifact, an overclaim, or a missing plan-required element for task20/M4 per development/loop7/refined_plan_v1.md:165-167). If none, say so. Be concise.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-02_20-06-43
- Tool: codex
