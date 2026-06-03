# Round 2 Contract

## Mainline Objective
Make the AC-3 non-learned scorer variants **correct and production-safe** (no measurement claims this round — fix the correctness holes Codex's Round-1 review found):
1. **Production-path graph-capture safety** — reject a non-default DS scorer when CUDA graph is enabled, at server init, covering the *real* model-forward capture path (not just `capture_decode_step`). Until the graph-safe Triton port lands, a non-default scorer MUST require `--disable-cuda-graph`, enforced with a clear startup error.
2. **Physical-path hybrid mis-application fix** — `compute_token_scores` (physical mode, no `seq_len`) must NOT silently treat `hybrid` as `cosine`; it must reject `scorer_norm="hybrid"` with a clear error.
3. **Complete the anchor ablation** — add `anchor_mode` `{off, recency, global, strided}` with a single deterministic anchor generator; keep `anchor_budget`.
4. **TP-determinism coverage** — parameterized TP-shaped test over `scorer_norm × head_agg × anchor_mode` asserting identical `selected_indices`/`valid_lengths` across ranks, preserving the fail-fast guard tests.

## Target ACs (1–2)
- **AC-3** (primary): variant correctness + TP cross-rank determinism + non-regression safety.
- **AC-6** (premise): no Tier-1 regression; non-default scorer is safe-or-rejected under production graph capture.

## Blocking Side Issues In Scope
- **The Round-1 graph-capture guard does not cover the production CUDA-graph runner** (`cuda_graph_runner._capture_graph` captures the full forward; a non-default scorer there sets `_force_eager_select` and runs the eager selector *inside* capture → allocates/host-syncs, not the claimed clean failure). This is the core of objective #1 and must be fixed at server init.

## Queued Side Issues Out Of Scope (justified)
- **Graph-safe Triton hybrid/cosine/head/anchor port + full AC-3 measurement matrix** (task #13): the definitive production fix + the binding non-regression matrix (MMLU, dense-DS, N≥50, DSA, eager-vs-graph perf). Heavy kernel + GPU work; next round. This round makes the variants *safe* (reject under graph) and *correct*; the matrix follows once they run under graph.
- **Oracle fail-closed + 64K re-run** (task #12, AC-1): M0 diagnostic hardening; does not block scorer correctness. Next round.
- **Tier-2.A / AC-4** (task13–17), **M4 consolidation / AC-6 perf** (task19–20): separate milestones; queued.
- **Plan-marker code/comment cleanup** (Loop/AC wording): cheap; done opportunistically if time, else queued for pre-merge.

## Round Success Criteria
- Server init **fails fast** (clear error) when `ds_scorer_is_default(config) == False` and CUDA graph is enabled; a test simulates the production capture path (not only `capture_decode_step`).
- `compute_token_scores` raises a clear error for `scorer_norm="hybrid"` (physical mode); unit test added.
- `anchor_mode {off, recency, global, strided}` implemented deterministically (recency = most-recent; global = earliest stable; strided = evenly spaced over `[0, seq_len)`), preserving exactly the selected count by evicting lowest-scoring non-anchor entries; unit tests for all modes, budget > top_k, short sequences, duplicate prevention, ascending ordering.
- Parameterized TP-shaped determinism test over `scorer_norm × head_agg × anchor_mode` asserts identical per-rank `selected_indices`/`valid_lengths`; `DoubleSparsityTPMisconfigured`/`DoubleSparsityRebindError` fail-fast preserved.
- Launcher exposes `HEAD_AGG`, `ANCHOR_MODE`, `ANCHOR_BUDGET`, `SCORER_NORM_HYBRID_THRESHOLD` env knobs in `DS_CONFIG`.
- All DS unit tests pass. The round summary makes no AC-3 *closure* claim (matrix/Triton-port pending).
