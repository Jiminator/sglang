# Round 13 Contract — Loop 7

## Mainline objective (EXACTLY ONE)
**task14 (completion) — wire the served opt-in *eager* lifted-budget decode branch
end-to-end and flip the availability seam.**

When `enable_lifted_budget_decode` is set (default off, eager-only until task16),
DS decode selects up to `lifted_budget_top_k` logical positions, converts them to
physical KV slots, runs the R12 request-local compact remap
(`build_compact_decode_index`), dequantizes the selected fp8 slots via
`dequantize_k_cache_paged`, and attends them with `flash_mla_sparse_fwd` (no 2048
cap) — instead of the default `flashmla_kv` path. The seam
`ds_lifted_budget_decode_available()` flips to `True` only after the branch is
wired, eager-gated, validated, and covered by tests. The default DSA/DS paths stay
byte-identical (every lifted code path behind a default-off `if lifted` guard; the
`flashmla_kv` `dsa_index_topk` assert is untouched).

## Target AC(s)
- **AC-4** — the opt-in adjustable-budget decode path. This round wires + enables the
  served eager branch (task14) and adds its served-correctness tests (task15 core).

## Blocking issues (truly block the mainline)
- **None outstanding.** The R12-discovered `lifted_budget_top_k % 128 == 0` kernel
  constraint must be enforced as part of enablement (a success criterion below), not
  a separate blocker. No live side blocker exists (the seam is `False` until flipped).

## Queued — explicitly OUT of scope this round (NOT closed/deferred)
- **task15 (remaining)** — TP=8 selected-index equality at 4096/8192 on the wired
  path (the existing TP determinism harness extended to the lifted width); a live
  served NIAH 4K recall-recovery sweep (the binding recall evidence) — both next.
- **task16** — production hardening (alloc-free `out=` dequant + CUDA-graph capture),
  gated behind the recall win; the lifted path stays eager-required until then.
- **task17** — Tier-2.A landing disposition record.
- **task19 / task20** — AC-6 perf consolidation + final strategic-gate decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## Concrete success criteria
1. **Config**: `lifted_budget_top_k % 128 == 0` is required when
   `enable_lifted_budget_decode` is set (the `flash_mla_sparse_fwd`
   `topk % (2*B_TOPK)` block constraint), with reject + accept unit tests.
2. **Validator**: when lifted is enabled it requires `top_k == index_topk`
   (the base budget stays the DSA budget; the lift is the *separate* wider width),
   `lifted_budget_top_k > index_topk`, `% 128 == 0`, **and `--disable-cuda-graph`**
   (the internally-allocating dequant is not graph-safe). The default fp8 path still
   uses `flashmla_kv`; lifted fp8 decode is allowed (routed internally), so the
   backend/dtype pairing is not broken. The R11 "recognized but not implemented"
   fail-closed gate is replaced by these checks now that the seam is `True`.
3. **Selection width**: `DoubleSparsitySelector.max_top_k` and the backend's
   `ds_max_top_k` (which sizes `ds_topk_indices_out` + `ds_graph_state`) become
   `lifted_budget_top_k` when lifted, so the selector emits a lifted-width selection
   with the existing deterministic (score-desc, position-asc) tie-break.
4. **Decode branch**: a wired eager path converts the lifted physical slots →
   `build_compact_decode_index` → `dequantize_k_cache_paged` (only
   `page_table_1_flattened`, never a `-1`) → `flash_mla_sparse_fwd`
   (`compact_indices`). Factored into a directly-testable helper in
   `lifted_budget.py` that the backend's `forward_decode` calls under the lifted guard.
5. **Seam flip**: `ds_lifted_budget_decode_available()` returns `True`.
6. **Served-correctness tests (GPU)**: the wired decode helper matches a reference
   sparse attention on deterministic fp8/dequant cases at **4096 and 8192** lifted
   widths, including prefix sharing, `valid_lengths` < width, and an **interior `-1`
   produced by within-row dedup** (not only suffix padding).
7. **Non-regression**: default-off path byte-identical; DSA `dsa_index_topk` assert +
   `SGLANG_DS_ALLOW_TOPK_MISMATCH` untouched; full DS unit suite passes; no
   plan-marker leakage in production code.
8. `m7_lifted_budget_design.md` + `goal-tracker.md` updated; commit.

## Tag routing
- task14 is a **`coding`** task → Claude executes directly.
