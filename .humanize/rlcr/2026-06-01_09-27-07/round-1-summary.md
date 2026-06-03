# Round 1 Summary — Loop 7

## Mainline objective (round-1-contract.md)
Land the production-shaped AC-3 Tier-2.B selector — primarily a **length-conditional hybrid scorer** — and measure it to show it recovers 4K while keeping the 16K gain; fix the two scorer-related blocking issues.

## Outcome: ACHIEVED (measured)

The length-conditional **hybrid scorer is the best of both regimes**:

| length | raw (prod) | uniform cosine | **hybrid** | path |
|--------|-----------|----------------|-----------|------|
| 4K | 75% [.51,.91] | 25% [.09,.49] | **85% [.62,.97]** | raw (≤8K) |
| 16K | 5% [.00,.25] | 40% [.19,.64] | **40% [.19,.64]** | cosine (>8K) |

The hybrid **recovers 4K (85%, cosine's regression gone)** AND **keeps 16K (40%)** — it is the per-length max. Measured 8×H200, N=20, via the new `LOOP7_MEASUREMENT` op-point mode. `development/loop7/m1_hybrid_finding.md`.

## Work Completed
- **`config.py`**: `scorer_norm` extended to `{off, cosine, hybrid}` + `scorer_norm_hybrid_threshold` (8192); independent `head_agg` (`max|mean`) and `anchor_budget` (int) config fields; all validated. Default = byte-identical.
- **`selection_kernel.py`**: hybrid (per-request `seq_len` raw/cosine) + head-aggregation in both score paths; `_force_include_recency_anchor` (anchor force-include); `ds_scorer_is_default` guard; threaded through `retrieve_topk_via_labels`.
- **`selector.py`**: passes all config variant values to the scorer.
- **`models/deepseek_v2.py`**: routes decode to the eager logical scorer for ANY non-default scorer.
- **`cuda_graph.py` (Blocking B1)**: `capture_decode_step` FAILS FAST for a non-default scorer instead of silently raw-scoring under capture.
- **`serve_double_sparsity.sh` (Blocking B2)**: `LOOP7_MEASUREMENT=1` pins int8/mem 0.7 and logs the effective `double_sparsity_config`.

## Files Changed
`config.py`, `selection_kernel.py`, `selector.py`, `deepseek_v2.py`, `cuda_graph.py`, `serve_double_sparsity.sh`, new `test_scorer_variants.py`, `development/loop7/m1_hybrid_finding.md` + `recall_hybrid.json`. Commit `273622705`.

## Validation
- **5 new variant unit tests** + **308 existing DS tests pass** (hybrid picks raw≤thr / cosine>thr, off==raw byte-identical, head_agg max≠mean, anchor force-include, default-guard).
- Live 8×H200 hybrid NIAH 4K/16K (N=20) — numbers above.

## Remaining Items (queued, justified)
- **Oracle fail-closed + 64K oracle re-run** (task #12, Codex gap #1): M0 diagnostic hardening; does not block the served-recall hybrid measurement. High priority next round.
- **Graph-safe Triton hybrid port + full AC-3 matrix** (task #13): MMLU re-anchor, dense-DS/within-budget parity, N≥50 binding 16K, DSA same-node reference, TP=8 cross-rank determinism per flag, hybrid-threshold sweep. **AC-3 closure depends on this — not claimed yet.**
- Tier-2.A / AC-4, M4 consolidation / AC-6 perf, plan-marker cleanup: separate milestones, queued.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: applied the prior round's lesson (BL-20260602-ds-flag-must-be-config-borne-not-env) — all new variants are DS-config fields. No new reusable engineering pitfall surfaced.

## Goal Tracker Update Request

### Requested Changes
- **task8 (AC-3 scorer)** → near-complete: the landable hybrid is implemented + measured (4K 85% / 16K 40%, best of both); remaining is the graph-safe port + matrix (task #13).
- **task9 (head-aggregation)** + **task10 (anchor-budget)** → mark **implemented + unit-tested** (config-gated, default byte-identical); per-variant measurement deferred to the AC-3 matrix (task #13).
- **Resolve Blocking Side Issue "graph-capture ignores scorer_norm"** → fixed (`cuda_graph.capture_decode_step` fail-fast + deepseek_v2 routing via `ds_scorer_is_default`).
- **Resolve Blocking Side Issue "serve op-point mismatch"** → fixed (`LOOP7_MEASUREMENT=1`).
- **Add Active task #13** (graph-safe Triton hybrid port + full AC-3 matrix) for AC-3 closure; **keep Active task #12** (oracle fail-closed + 64K) for AC-1/AC-2 binding.

### Justification
The round delivered the evidence-backed AC-3 landed candidate (hybrid, best-of-both measured) and fixed both scorer Blocking Side Issues, advancing AC-3/AC-6. The remaining AC-3 items (graph-safe port, full matrix) and the oracle fail-closed (AC-1) are scoped as the next round's mainline so the loop converges on binding closure rather than partial claims.
