# Round 10 Review Result

Mainline Progress Verdict: ADVANCED

Round 10 advanced AC-4 by adding the named ABI fields and the Codex-reviewed
decode-path design record, and the R9 anchor temporary-shape hardening is real.
However, task13 is not complete: the validator now accepts lifted-budget startup
states even though no lifted decode backend path is implemented or selected. That
violates the plan's "enable flag + opt-in backend path selected" requirement and
creates both a silent no-op path and a later old-backend assert path.

Do not close Loop 7. AC-4 task13 remains active, task14-task17 are still original
plan mainline work, and AC-6/task19 plus task20 remain active.

## Implementation Review

Accepted R10 work:

1. The config ABI fields exist.

   Evidence: `DoubleSparsityConfig` now has `enable_lifted_budget_decode` and
   `lifted_budget_top_k`, `_ALLOWED_FIELDS` accepts both, and
   `parse_double_sparsity_config` parses the bool/int fields
   (`python/sglang/srt/layers/attention/double_sparsity/config.py:30`,
   `:102`, `:263`). Config validation rejects missing budget with the flag,
   budget set without the flag, and `lifted_budget_top_k <= top_k`.

2. The design record exists and integrates the Codex review.

   Evidence: `development/loop7/m7_lifted_budget_design.md` records the
   physical -> `page_table_1_flattened` -> request-local compact remap, within-row
   duplicate hazard, `-1` pad masking before dequant, eager research gating, and
   required direct `flash_mla_sparse_fwd` 4K-topk smoke. The claimed ask-codex
   output exists at `.humanize/skill/2026-06-02_14-28-12-2622279-9cc1b981/output.md`
   and matches those design constraints.

3. The R9 anchor hardening is accepted.

   Evidence: `_force_include_anchor` clamps `A = min(anchor_budget, K, max_seq)`
   before allocating `[bs, A]` temporaries
   (`python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:917`).
   `test_anchor_over_budget_graph_matches_eager` covers `anchor_budget > top_k`
   and `seq_len < K` for recency/global/strided. Local run:
   `pytest -q test/registered/unit/layers/attention/test_scorer_variants.py` ->
   `28 passed`.

## Mainline Gaps

1. **task13 fail-closed validator gate is incomplete.**

   `validate_double_sparsity()` treats `enable_lifted_budget_decode=true` as
   sufficient to bypass the `top_k > index_topk` rejection
   (`validator.py:201-239`), but there is no implemented lifted backend path.
   Downstream, `DoubleSparsitySelector.max_top_k` is still `config.top_k`
   (`selector.py:86`), the DS output is passed through the existing
   logical-to-physical adapter (`deepseek_v2.py:2360-2382`), and the default
   `flashmla_kv` decode path still asserts `indices.shape[-1] == dsa_index_topk`
   (`dsa_backend.py:2148-2151`).

   I verified the validator gap with a stubbed valid mask/model config:
   `top_k=2048, enable_lifted_budget_decode=true, lifted_budget_top_k=4096`
   passed validation, which silently runs the old 2048 selector shape. Also
   `top_k=4096, enable_lifted_budget_decode=true, lifted_budget_top_k=8192`
   passed validation, which can route a 4096-wide tensor into the old
   2048-only `flashmla_kv` assert.

   Required fix: before claiming task13 done, make lifted-budget startup fail
   closed unless task14's opt-in backend path is actually implemented and
   selected. Add validator tests for both cases above. If task14 is not yet
   implemented, `enable_lifted_budget_decode=true` must raise a clear
   "recognized but not implemented/selected" error rather than booting.

2. **AC-4 task14-task17 remain unimplemented and cannot be treated as queue.**

   Required implementation plan:
   - Keep `top_k` as the base DSA budget equal to `index_topk`; use
     `lifted_budget_top_k` as the fixed padded lifted selection width.
   - Implement an explicit lifted decode branch gated by
     `enable_lifted_budget_decode`, initially requiring eager/no CUDA graph.
   - In that branch, select `lifted_budget_top_k` logical positions with the
     existing deterministic score-desc/position-asc ordering, then map logical
     positions to physical slots, mask/safe-replace pads before dequant, build a
     request-local compact dequant buffer, and pass compact-domain indices to
     `flash_mla_sparse_fwd`.
   - Assert or dedup duplicates after physical remap before attention; duplicates
     within a query row must not reach `flash_mla_sparse_fwd` as multiple valid
     entries.
   - Add task15 tests: reference sparse-attention tolerance on deterministic
     fp8/dequant cases, prefix-sharing remap, within-row duplicate handling,
     invalid padding, `valid_lengths`, TP=8 equality at 4096/8192, and the direct
     `flash_mla_sparse_fwd` 4K-topk smoke/accuracy test.
   - For task16, add an `out=`/scratch `dequantize_k_cache_paged` variant plus
     q-padding/compact-index scratch and prove zero allocation under graph replay
     before allowing CUDA graph production use.
   - For task17, write the landing disposition with served recall evidence and
     CIs. If production hardening is deferred, the record must show recall
     evidence, DSA default untouched, and the research path gated out of
     production capture.

3. **AC-6/task19 and task20 are still pending.**

   Required implementation plan:
   - After task17 exists, run the existing measurement/benchmark tooling at the
     Loop-7 op-point for DS-default, graph-safe DS-hybrid, and DSA.
   - Record conc-1 and conc-16 TTFT, decode TPS/req, GPU memory, graph replay
     status, admission, radix/cache assumptions, and exact server configs.
   - Write the consolidated DS-vs-DSA recall/perf/non-regression report.
   - Write the final decision record superseding the Loop-6 Tier-2.A-primary gate
     with the final M0/R4/R7/R8/R9/R10 evidence and the AC-4 disposition.

## Blocking Side Issues

None outside the mainline. The lifted-budget startup gate issue above is the
current blocker, but it is task13/AC-4 mainline work rather than a side issue.

## Queued Side Issues

1. Preserve/cite the R8 raw oracle-sink provenance before task20, or cite the
   hardcoded `stride=1` call site plus committed aggregate explicitly.
2. Remove plan/workflow markers and stale variant comments before final merge.
3. Learned/distilled selector work remains out of scope unless explicitly
   approved under DEC-5.

## Goal Alignment Summary

ACs: 6/6 addressed | Forgotten items: 0 | Unjustified deferrals: 6

| AC | Status | Review result |
|----|--------|---------------|
| AC-1 | MET | Prior R8 evidence remains accepted. |
| AC-2 | PARTIAL | Recall/MMLU evidence exists; task19 consolidation and task20 final record remain. |
| AC-3 | MET | R9 graph-safe anchor plus R10 anchor clamp/test are accepted. |
| AC-4 | NOT MET | ABI/design advanced, but task13 validator fail-closed behavior is incomplete and task14-task17 are missing. |
| AC-5 | MET | 64K servability at mem0.7 remains verified. |
| AC-6 | PARTIAL | No final conc-1/16 perf guardrail report yet. |

I updated `goal-tracker.md` mutable state:

- bumped Plan Version to 12 for Round 10 Review;
- corrected the Round 10 tracker row from "task13 DONE" to "task13 PARTIAL";
- kept task13 active with the required validator fix;
- added the lifted-budget startup gate as a Blocking Side Issue;
- left task14-task17, task19, and task20 active.

PENDING
