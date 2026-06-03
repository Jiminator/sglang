# Round 2 Summary — Loop 7

## Mainline objective (round-2-contract.md)
Make the AC-3 non-learned scorer variants **correct and production-safe** (fix the Round-1 review's correctness holes; no measurement claims).

## Outcome: ACHIEVED

All four correctness items + the blocking issue are fixed and tested.

## Work Completed
1. **Production-path graph-capture safety (Blocking)** — Round 1's guard only covered `capture_decode_step`, not the real CUDA-graph runner (where a non-default scorer would run the eager selector *inside* capture). `validate_double_sparsity` now **fails fast at server init** when `ds_scorer_is_default(config) == False` and CUDA graph is enabled. **Verified on a real boot**: with `scorer_norm=cosine` + graphs enabled it errors *before* "Load weight begin" with `"...not yet supported under CUDA graph capture... Re-run with --disable-cuda-graph"`. (The `capture_decode_step` guard is kept as defense-in-depth.)
2. **Physical-path hybrid mis-application** — `compute_token_scores` now **rejects** `scorer_norm="hybrid"` (it has no per-request `seq_len`) instead of silently degrading to cosine (the exact moderate-context regression hybrid avoids).
3. **Anchor ablation completed** — `anchor_mode {off, recency, global, strided}` config field with a single deterministic generator `_anchor_positions` (recency = most-recent, global = earliest, strided = evenly spaced over `[0, seq_len)`), budget-clamp / short-seq / dedup / ascending-order handling; replaces the recency-only impl.
4. **TP cross-rank determinism** — a real 2-rank **gloo multiprocess** test (`test_ds_scorer_tp_determinism.py`) parameterized over `scorer_norm × head_agg × anchor_mode`: each rank holds a head-shard, computes per-rank scores, all-reduces, runs the shared top-K + anchor, and asserts identical per-rank `selected_indices`/`valid_lengths`. The `DoubleSparsityTPMisconfigured`/`DoubleSparsityRebindError` fail-fast guards are kept in the matrix.
5. **Launcher knobs** — `serve_double_sparsity.sh` exposes `HEAD_AGG`, `ANCHOR_MODE`, `ANCHOR_BUDGET`, `SCORER_NORM_HYBRID_THRESHOLD` in `DS_CONFIG` (needed for the AC-3 matrix).

## Files Changed
`config.py` (anchor_mode), `selection_kernel.py` (physical-hybrid reject, anchor modes, `ds_scorer_is_default`), `selector.py`, `validator.py` (startup guard), `cuda_graph.py` (message), `serve_double_sparsity.sh` (knobs), updated `test_scorer_variants.py`, new `test_ds_scorer_tp_determinism.py`. Commit `72c704edf`.

## Validation
- **301 DS unit tests + 6 gloo TP determinism tests pass** (anchor generators all modes, recency/strided/off/dedup force-include, physical-hybrid reject, default-guard, TP cross-rank equality for 5 flag combos, startup-guard reject + eager-pass).
- **Real boot** confirms the startup guard fires before model load.

## Remaining Items (queued, justified)
- **Graph-safe Triton hybrid/cosine/head/anchor port + full AC-3 measurement matrix** (task #13): the definitive production fix (so the variants run under CUDA graph) + binding non-regression matrix (MMLU re-anchor, dense-DS/within-budget parity, N≥50 16K, DSA same-node, eager-vs-graph perf, per-variant attribution, threshold sweep). Heavy kernel + GPU work; next round. **AC-3 closure depends on this.**
- **Oracle fail-closed + 64K re-run** (task #12, AC-1): M0 diagnostic hardening; next round.
- **Tier-2.A / AC-4** (task13–17), **M4 consolidation / AC-6 perf** (task19–20): separate milestones.
- **Plan-marker code/comment cleanup**: pre-merge cleanup, queued.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: no new reusable engineering pitfall surfaced. Reinforced the prior config-borne-flag lesson (the production guard reads `selector.config`, not env). The "guard the real runner path, not just the helper" point is captured in the round summary + the validator comment.

## Goal Tracker Update Request

### Requested Changes
- **Resolve Blocking Side Issue "graph-capture fix does not cover the production CUDA-graph runner"**: fixed — startup guard in `validate_double_sparsity` rejects a non-default scorer + CUDA graph before model load (real-boot verified); `capture_decode_step` guard kept as defense-in-depth.
- **task10 (anchor-budget)** → move to **implemented + unit-tested**: full `anchor_mode {off,recency,global,strided}` ablation with deterministic generators + tests (recency/global/strided, budget>top_k, short-seq, dedup, ordering). Per-variant *measurement* deferred to the AC-3 matrix (task #13).
- **task11 (TP determinism)** → move to **implemented + tested**: parameterized gloo multiprocess test over scorer_norm × head_agg × anchor_mode with cross-rank equality + fail-fast guard coverage.
- **task8 (AC-3 scorer)**: physical-path hybrid mis-application fixed (now rejected); remaining for AC-3 closure is the graph-safe Triton port + matrix (task #13).
- **Keep Active**: task #13 (graph-safe port + matrix) and task #12 (oracle fail-closed) as the next round's mainline; AC-4 (task13–17 plan) and M4 (task19–20) queued.

### Justification
Round 2 closed the correctness/safety gaps the Round-1 review found (production graph guard, physical-hybrid, full anchor ablation, TP determinism) so the AC-3 selector variants are trustworthy and cannot be silently misused under production graph capture. The remaining AC-3 work (graph-safe port + binding matrix) and the oracle fail-closed (AC-1) are the next round's mainline, converging the loop toward binding closure.
