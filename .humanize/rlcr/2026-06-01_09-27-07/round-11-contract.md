# Round 11 Contract — Loop 7

## Mainline objective (EXACTLY ONE)
**Complete AC-4 / task13 by making the lifted-budget opt-in fail closed at startup.**
`enable_lifted_budget_decode=true` must raise a clear *recognized-but-not-implemented/selected*
error at `validate_double_sparsity` until task14's opt-in lifted decode backend path is
implemented and selected — eliminating the two startup holes the R10 review found:
- **silent no-op**: `top_k=2048, enable_lifted_budget_decode=true, lifted_budget_top_k=4096`
  boots the locked 2048 selector (the lifted budget is never honored);
- **wide-into-old-assert**: `top_k=4096, enable_lifted_budget_decode=true, lifted_budget_top_k=8192`
  boots toward the default `flashmla_kv` `indices.shape[-1] == dsa_index_topk` (2048) assert.

This is the exact Codex-required fix ("If task14 is not yet implemented,
`enable_lifted_budget_decode=true` must raise a clear 'recognized but not implemented/selected'
error rather than booting"). It is the round's primary success condition.

## Target AC(s)
- **AC-4 (task13)** — the opt-in lifted-budget ABI. This round closes the fail-closed validator
  requirement that R10 left incomplete. AC-4 itself stays NOT MET (decode path = task14-17).

## Blocking issues (truly block the mainline)
- The single R10 **Blocking Side Issue**: "Lifted-budget flag can pass startup without an
  implemented/selected lifted backend." This IS the mainline this round — not a side issue to
  bundle. Fixing it completes task13.

## Queued — explicitly OUT of scope this round (but NOT closed/deferred)
These remain **active mainline for subsequent rounds** (Codex: "task14-task17 ... cannot be
treated as queue" — i.e. they must not be permanently deferred; they are simply sequenced after
the fail-closed gate):
- **task14** — eager research lifted-budget decode path (`flash_mla_sparse_fwd` + request-local
  compact remap, dedup, `-1`/pad masking, R23 tie-break). *Next round's mainline.*
- **task15** — kernel-correctness/safety tests incl. the direct `flash_mla_sparse_fwd` 4K-topk smoke.
- **task16** — production hardening (alloc-free `out=` dequant + CUDA-graph), gated behind the recall win.
- **task17** — Tier-2.A landing disposition record (closes AC-4 via landed-or-deferred-with-evidence).
- **task19 / task20** — AC-6 perf consolidation + final strategic-gate supersession decision record.
- Evidence-hygiene queued items (R8 oracle-sink provenance; plan-marker cleanup) — untouched this round.

## Concrete success criteria
1. `validate_double_sparsity` **raises a clear fail-closed error** whenever
   `enable_lifted_budget_decode` is set while the backend path is unavailable, **independent of
   hf_config resolution** (fires before the capability/model-topk block, so it cannot be skipped
   when the model config can't be resolved).
2. A single clean seam — `ds_lifted_budget_decode_available()` predicate (returns `False` now) —
   mirroring the existing `ds_scorer_is_graph_safe` idiom, so **task14 flips one line** to enable.
3. Two new validator tests reproduce Codex's exact cases and now **RAISE**:
   (A) `top_k=2048 + lifted + lifted_budget_top_k=4096`; (B) `top_k=4096 + lifted + lifted_budget_top_k=8192`.
4. The stale `TestLiftedBudgetABI::test_validator_topk_gt_index_topk_requires_flag` "WITH the
   lifted flag → passes validation" assertion is **flipped to fail-closed** (it asserted the
   exact wrong behavior R10 flagged); the no-flag `top_k>index_topk` steering assertion is kept.
5. Config **parse-level** ABI validation is unchanged (the field still parses; the fail-closed
   guard lives at the server validator — the correct layer, where the backend availability is known).
6. **Default-off path byte-identical**: DSA `dsa_index_topk` assert + the
   `SGLANG_DS_ALLOW_TOPK_MISMATCH` equality-mismatch ablation are untouched.
7. Full DS unit suite passes.
8. `m7_lifted_budget_design.md` + `goal-tracker.md` updated (task13 → done, blocking side issue
   resolved); commit + push.

## Tag routing
- task13 fail-closed completion is a **`coding`** sub-task → Claude executes directly (the
  decode-path *design* part of task13 was already done via `ask-codex` in R10).
