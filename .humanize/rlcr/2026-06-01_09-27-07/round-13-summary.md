# Round 13 Summary — Loop 7

## Mainline objective (round-13-contract.md)
**task14 (completion) — wire the served opt-in *eager* lifted-budget decode branch
end-to-end and flip the availability seam.**

## Outcome: ACHIEVED — task14 DONE; the served eager branch is wired + enabled.

## What it does
When `enable_lifted_budget_decode` is set (default off, eager-only until task16),
DS decode selects up to `lifted_budget_top_k` logical positions, converts them to
physical KV slots, runs the R12 request-local compact remap, dequantizes the
selected fp8 slots via `dequantize_k_cache_paged`, and attends them with
`flash_mla_sparse_fwd` (no 2048 cap) — instead of the default `flashmla_kv` path.

## Work Completed (`coding`, Claude)
1. **Config** (`config.py`): enforce `lifted_budget_top_k % 128 == 0` (the
   `flash_mla_sparse_fwd` `topk % (2*B_TOPK)` block constraint), alongside `> top_k`.
2. **Validator** (`validator.py`): when enabled, require `top_k == index_topk` (the
   base budget stays the DSA budget; `lifted_budget_top_k` is the SEPARATE wider
   width), `lifted_budget_top_k > index_topk`, `% 128`, and **`--disable-cuda-graph`**
   (the dequant allocates internally, not graph-safe). The R11 "not implemented"
   fail-closed gate is replaced by these checks (kept as defense if a build ever
   ships the flag without the backend).
3. **Selection width** (`selector.py` + `dsa_backend.py`):
   `DoubleSparsitySelector.max_top_k` and the backend's `ds_max_top_k` (which sizes
   `ds_topk_indices_out` + `ds_graph_state`) widen to `lifted_budget_top_k` when
   enabled — one value cascades the selection/output buffers to lifted width; the
   R23 tie-break is unchanged.
4. **Decode branch** (`dsa_backend.py` + `lifted_budget.py`): `forward_decode`
   routes the lifted case (the physical FUSE_TOPK `page_table_1`) to a new
   `_forward_lifted_budget` → `build_lifted_compact_kv` (remap +
   `dequantize_k_cache_paged` for the fp8 store, gather for bf16) → the existing
   `_forward_flashmla_sparse`. Behind a default-off
   `getattr(self, "ds_lifted_budget_decode", False)` guard so the default DSA/DS
   decode is byte-identical and the `flashmla_kv` `dsa_index_topk` assert is untouched.
5. **Seam** (`selection_kernel.py`): `ds_lifted_budget_decode_available()` → `True`.

## Files Changed
- `double_sparsity/config.py` (`%128` validation).
- `double_sparsity/lifted_budget.py` (`+ build_lifted_compact_kv` decode helper).
- `double_sparsity/selection_kernel.py` (seam → True).
- `double_sparsity/selector.py` (lifted `max_top_k`).
- `double_sparsity/validator.py` (lifted gating: eager-required, `top_k==index_topk`).
- `dsa_backend.py` (`ds_lifted_budget_decode`/`ds_max_top_k` at init; `forward_decode`
  lifted route; `_forward_lifted_budget`).
- `test_scorer_variants.py` (`TestLiftedBudgetABI`: `%128` reject/accept + validator
  gating; replaced the stale R11 "not implemented" assertions).
- `test_lifted_budget_decode.py` (GPU served-helper tests at 4096/8192).
- `m7_lifted_budget_design.md` (served branch landed + updated risks).
- Commit `2ba4dafc1` (local — loop hook keeps commits local until completion).

## Validation
- `TestLiftedBudgetABI` (config `%128` + validator gating) + `test_lifted_budget_decode`
  → **24 passed** (incl. the new 4096/8192 served-helper GPU tests).
- GPU served-helper at **4096 and 8192** widths via the production
  `build_lifted_compact_kv`: prefix-sharing, `valid_lengths` < width, and an
  **interior `-1` from within-row dedup**, all matched vs a reference attention; the
  4096 case confirms a request attends 3000 > 2048 rows (no cap).
- Full DS unit suite (4 files) → **337 passed + 9 subtests**, no regressions.
  (Fixed a partial-backend stub by reading the new flag via `getattr(..., False)`.)
- Default-off path byte-identical; DSA `dsa_index_topk` assert +
  `SGLANG_DS_ALLOW_TOPK_MISMATCH` untouched; no new plan-marker leakage.

## Remaining Items (active mainline, NOT queued-out)
- **task15 (remaining)** — a **live served NIAH 4K recall-recovery sweep** (eager,
  N≥20 + CIs; the M0 oracle predicted recall@4096 ≈ 100% vs recall@2048 ≈ 44%) —
  the binding recall evidence; + TP=8 selected-index equality at the lifted
  4096/8192 width (extend the existing TP determinism harness).
- **task16** — production hardening (alloc-free `out=`/scratch dequant + CUDA-graph
  capture), gated behind the recall win; the path stays eager-required until then.
- **task17** — Tier-2.A landing disposition record.
- **task19 / task20** — AC-6 perf consolidation + final strategic-gate decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## AC-4 status
**task14 DONE** (served eager branch wired + enabled). **AC-4 NOT MET** — served 4K
recall-recovery evidence + TP=8 equality (task15), task16 hardening, and the task17
disposition remain.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: This round *applied* the R12 lesson
  `BL-20260602-flash-mla-sparse-fwd-compact-decode-contract` (the `%128` width, the
  `<0`/`>=s_kv` masking, and the request-local compact remap) directly in the config
  `%128` check, the validator gating, and the decode helper. The wiring specifics
  (widening the single `ds_max_top_k` cascades the metadata buffers; FUSE_TOPK gives
  a physical `page_table_1` at decode; `getattr`-default guards for partial test
  backends) are codebase-structural, not a reusable cross-round pitfall — no new or
  updated lesson.

## Goal Tracker
Updated directly (Plan Version 16): R13 Plan Evolution row added; task14 → **done**;
task15 → partial (decode-helper tests done; live recall + TP=8 remain). No Goal
Tracker Update Request needed.
