# Round 11 Summary — Loop 7

## Mainline objective (round-11-contract.md)
**Complete AC-4 / task13 by making the lifted-budget opt-in fail closed at
startup.** `enable_lifted_budget_decode=true` must raise a clear
*recognized-but-not-implemented/selected* error until the opt-in decode backend
path (task14) is implemented/selected — closing the two R10-review startup holes.

## Outcome: ACHIEVED — task13 DONE; the R10-review Blocking Side Issue is resolved.

## Problem (R10 review, verified in code before fixing)
`validate_double_sparsity` accepted a lifted config because the model-topk block's
`elif lifted: logger.info(...)` branch passed it through (`validator.py:235-240`),
even though no lifted decode backend exists. Two concrete holes (both confirmed):
- **silent no-op**: `top_k=2048, enable_lifted_budget_decode=true, lifted_budget_top_k=4096`
  booted the locked 2048 selector (the wider budget was never honored);
- **wide-into-old-assert**: `top_k=4096, enable_lifted_budget_decode=true, lifted_budget_top_k=8192`
  booted toward the default `flashmla_kv` `indices.shape[-1] == dsa_index_topk` (2048) assert.

## Work Completed (`coding`, Claude)
1. **Capability seam.** Added `ds_lifted_budget_decode_available()` in
   `selection_kernel.py` (returns `False` today), mirroring the existing
   `ds_scorer_is_graph_safe` idiom. This is the **single one-line seam** the
   decode-path landing (task14) flips to `True` once the path exists.
2. **Fail-closed validator gate.** `validate_double_sparsity` now raises a clear
   "recognized but not implemented/selected" `ValueError` whenever
   `enable_lifted_budget_decode` is set while
   `ds_lifted_budget_decode_available()` is `False`. Placed **right after the
   channel_mask_path check, before the capability/model-topk block**, so it is
   **hf_config-independent** — it cannot be skipped when the model config can't be
   resolved. The error names both failure modes and the remedy.
3. **Steering + defaults preserved.** The no-flag `top_k > index_topk` steering
   (toward the ABI, not `SGLANG_DS_ALLOW_TOPK_MISMATCH`) is unchanged. The
   model-topk block's lifted-shape validation (`lifted_budget_top_k > index_topk`
   + info log) is retained as the **post-backend** layer (reachable once the seam
   flips). Default-off path, the DSA `dsa_index_topk` assert, and the
   `SGLANG_DS_ALLOW_TOPK_MISMATCH` ablation are untouched.
4. **Design doc.** `m7_lifted_budget_design.md` updated: the "Landed" section now
   states the R11 fail-closed gate + the enablement seam (task14 knows the
   one-line flip and that the model-topk lifted validation activates afterward).

## Files Changed
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` —
  `+ ds_lifted_budget_decode_available()` seam (returns `False`).
- `python/sglang/srt/layers/attention/double_sparsity/validator.py` —
  `+` hf_config-independent fail-closed gate on `enable_lifted_budget_decode`.
- `test/registered/unit/layers/attention/test_scorer_variants.py` —
  flip the stale `test_validator_topk_gt_index_topk_requires_flag` "WITH flag
  passes" assertion to no-flag-steering-only; `+` two fail-closed case tests.
- `development/loop7/m7_lifted_budget_design.md` — R11 fail-closed gate + seam.
- Commit `a62ce91de` (local — see Push note).

## Validation
- `pytest test_scorer_variants.py -k LiftedBudget` → **9 passed** (incl. both new
  fail-closed cases now RAISE, and the no-flag steering still raises).
- Full DS unit suite (`test_scorer_variants.py`, `test_double_sparsity_unit.py`,
  `test_ds_scorer_tp_determinism.py`) → **322 passed + 9 subtests**, no regressions.
- Config parse-level ABI validation unchanged (the field still parses; the
  fail-closed guard lives at the server validator).

## Push note
`git push` was **blocked by the loop bash hook** ("Current commits should stay
local … use `--push-every-round`"). The commit is local-only; the loop manages it.
(For this loop the harness explicitly overrides the standing round-boundary-push
preference.)

## Remaining Items (active mainline, NOT queued-out)
- **task14 (next mainline)** — eager research lifted-budget decode path
  (`flash_mla_sparse_fwd` + request-local compact remap, dedup, `-1`/pad mask,
  R23 tie-break); flips `ds_lifted_budget_decode_available()`.
- **task15** — kernel-correctness/safety tests incl. the direct
  `flash_mla_sparse_fwd` 4K-topk smoke.
- **task16** — production hardening (alloc-free `out=` dequant + CUDA-graph),
  gated behind the recall win.
- **task17** — Tier-2.A landing disposition record (closes AC-4 via
  landed-or-deferred-with-evidence).
- **task19 / task20** — AC-6 perf consolidation + final strategic-gate
  supersession decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## AC-4 status
**task13 DONE** (opt-in ABI fields + Codex-reviewed design + fail-closed validator
gate). **AC-4 itself remains NOT MET** — the decode path is task14–17. The seam is
in place so task14 is a clean continuation.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: Selected `BL-20260602-ds-flag-must-be-config-borne-not-env` (the gate
  reads `config.enable_lifted_budget_decode`, config-borne — reaches TP workers,
  no env read) and `BL-20260528-dsv32-ds-serving-boot-chain` (fail closed at boot
  with a clear error). Both were *applied*, not extended: a one-round, single-gate
  validator fix following established patterns, not a multi-round pitfall.

## Goal Tracker
Updated directly (Plan Version 13): R11 Plan Evolution row added; task13 moved to
Completed and Verified (pending R11 review); the R10-review Blocking Side Issue
moved to Resolved (R11); Blocking Side Issues now empty; task14 flagged as the next
mainline with the seam note. No Goal Tracker Update Request needed.
