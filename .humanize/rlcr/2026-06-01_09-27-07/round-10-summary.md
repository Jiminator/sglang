# Round 10 Summary — Loop 7

## Mainline objective (round-10-contract.md)
**Land the Tier-2.A lifted-budget ABI (AC-4 / task13)**: the opt-in config fields
+ validator gating + the Codex-reviewed design/disposition record (the AC-4
foundation the decode path plugs into).

## Outcome: ACHIEVED — task13 done; the AC-4 ABI is landed and the decode-path design is recorded.

## Work completed
1. **ABI config fields.** `DoubleSparsityConfig.enable_lifted_budget_decode: bool`
   (default `False`) + `lifted_budget_top_k: int` (default `0`), in
   `_ALLOWED_FIELDS` + dataclass + `parse_double_sparsity_config`.
   `__post_init__` validation: `lifted_budget_top_k` must be `> top_k` when
   enabled; **set-without-flag and flag-without-budget both fail closed** (no
   silent no-op).
2. **Validator gate.** `top_k > index_topk` is **rejected unless
   `enable_lifted_budget_decode`** is set, and `lifted_budget_top_k` must be
   `> index_topk`; the error steers to the ABI and explicitly forbids
   `SGLANG_DS_ALLOW_TOPK_MISMATCH` / `max_top_k` / Twilight fields as the
   mechanism. **Default-off leaves the DSA `dsa_index_topk` assert + the
   equality-mismatch ablation escape unchanged.**
3. **task13 design/disposition record** (`m7_lifted_budget_design.md`),
   **reviewed via `ask-codex`** and integrated: the physical →
   `page_table_1_flattened` → **request-local compact** dequant-index remap;
   prefix-sharing is safe per-request but **within-row duplicates are not**
   (`flash_mla_sparse_fwd` would double-attend → dedup after remap); **`-1` pads
   masked before dequant** (a `-1` into `dequantize_k_cache_paged` is invalid);
   the alloc-free `out=` dequant + CUDA-graph landing **deferred to task16** per
   DEC-4/DEC-6 (eager research path first, gated off production capture); a
   **direct `flash_mla_sparse_fwd` 4K-topk smoke** required (local coverage is
   sparse-prefill top-k ≤ 512).
4. **Bundled (Codex R9 queued #1 + claim-correction #4).** Clamped
   `_force_include_anchor`'s temp shape to `A = min(anchor_budget, top_k, max_seq)`
   (bit-identical — clamped-out slots are invalid anyway; bounds a pathological
   opt-in budget) + a new **over-budget (`anchor_budget > top_k`, seq_len < K)**
   GPU eager-vs-graph test.

## Validation
- `TestLiftedBudgetABI`: config accept/reject matrix (valid lifted; reject
  lbk≤top_k, lbk-without-flag, flag-without-lbk; Twilight/`max_top_k` still
  rejected as unknown) + the validator `top_k > index_topk` gate via a
  monkeypatched `get_dsa_index_topk`.
- `test_anchor_over_budget_graph_matches_eager`: over-budget anchor bit-identical
  eager-vs-graph.
- **354 DS unit tests pass.**

## Files changed
`config.py` (ABI fields + validation + `_coerce_bool` field name),
`validator.py` (top_k>index_topk gate), `selection_kernel.py` (anchor temp clamp),
`test_scorer_variants.py` (`TestLiftedBudgetABI` + over-budget anchor test),
`m7_lifted_budget_design.md` (new). Commit `c41e5193a` (pushed).

## AC-4 status
ABI + validator gating + design/disposition **landed (task13 DONE)**; AC-4 itself
remains NOT MET pending the decode path + served recall evidence + the task17
disposition (task14–17). The plan's "landed-or-deferred-with-evidence" branch is
the planned closure given Tier-2.A is bounded-secondary (the long-context goal is
served by the landed Tier-2.B hybrid scorer).

## Remaining items (queued, justified)
- **task14/15: eager research lifted-budget decode path + correctness/safety
  tests** (incl. the required direct `flash_mla_sparse_fwd` 4K-topk smoke).
- **task16: production hardening** (alloc-free dequant + CUDA-graph), gated behind
  the recall win.
- **task17: Tier-2.A landing disposition record** (closes AC-4).
- **AC-6 perf consolidation (task19) + final strategic-gate decision record
  (task20)**.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: the ABI + validator gating follow the established config-borne-flag
  pattern; the one design-hygiene point (a wider-budget `top_k>index_topk` needs
  its OWN opt-in ABI, distinct from the equality-mismatch `SGLANG_DS_ALLOW_TOPK_MISMATCH`
  ablation escape) is captured in the validator message + the design doc, not a
  reusable multi-round pitfall.

## Goal Tracker Update Request
- **task13** (AC-4): lifted-budget ABI + design record DONE (R10).
- **Resolve queued**: anchor-budget temp-shape clamp (done R10); MMLU data_dir
  (done R9).
- **Keep Active**: task14–17 (AC-4 decode path/tests/disposition), AC-6/task19–20
  (perf + final decision record).
