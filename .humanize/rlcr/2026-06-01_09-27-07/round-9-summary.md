# Round 9 Summary — Loop 7

## Mainline objective (round-9-contract.md)
**Port the anchor-budget variant to the graph-safe path (AC-3 completion)**: a
tensorized, fixed-shape, alloc-free post-topK force-include for
`anchor_mode ∈ {recency,global,strided}` that is bit-identical to the eager
`_force_include_anchor`, wired into the graph-safe path, with the guard relaxed so
anchor no longer requires `--disable-cuda-graph`.

## Outcome: ACHIEVED — AC-3 variant coverage is complete on the production path.

## Work completed
1. **Tensorized `_force_include_anchor`.** Replaced the per-row Python loop
   (`.item()`, `for b`) with a fully tensorized, fixed-shape, host-sync-free
   implementation: `effective_budget = min(anchor_budget, valid_count, seq_len)`;
   anchor positions via `_anchor_positions_tensor` (recency/global/strided +
   strided ascending set-dedup); evict the k lowest-score non-anchor selected
   (stable score-asc / position-asc tie-break via `_stable_argsort_ascending`);
   insert the first k missing anchors; re-sort. **Bit-identical to the former
   reference — fuzz 2000/2000.** Used by BOTH the eager and graph-safe paths, so
   they cannot diverge.
2. **Graph-safe integration.** `retrieve_topk_graph_safe` runs the force-include
   after the top-K; `anchor_mode`/`anchor_budget` thread through the deepseek_v2
   graph-safe call site and `capture_decode_step`.
3. **Guards relaxed.** `ds_scorer_is_graph_safe()` now returns `True` (every
   non-learned variant is graph-safe); the validator / `_force_eager_select` /
   capture guard no longer force eager for anchor; the serve script only adds
   `--disable-cuda-graph` for the recall-oracle diagnostic.

## Validation (GPU)
- **Eager-vs-graph bit-identical selection** over the full
  `scorer_norm{off,cosine,hybrid} × head_agg{max,mean} × anchor_mode{off,recency,
  global,strided}` matrix (24 combos) on **fp16 + int8**
  (`TestGraphSafeScorerEqualsEager`).
- **Real CUDA-graph capture/replay**: a hybrid+recency-anchor selection captured
  in a `torch.cuda.CUDAGraph` replays **byte-identical to eager + 0 new
  allocations** (`test_anchor_graph_safe_replay_zero_alloc`).
- **TP=8 cross-rank determinism** holds (`test_ds_scorer_tp_determinism.py`).
- Default (anchor off) byte-identical; **346 DS unit tests pass**.

## AC-3 status
All three AC-3 non-learned variants are now flag-gated + graph-safe +
non-regressing: channel-normalization (cosine/hybrid) + head-aggregation [R6] and
anchor-budget (recency/global/strided) [R9]; default byte-identical; within-budget
parity + MMLU ≤1.0pp + binding 16K uplift [R7]; TP=8 equality [R3/R9]. **AC-3
variant coverage complete on the production CUDA-graph path.**

## Files changed
`selection_kernel.py` (tensorized `_force_include_anchor` + `_anchor_positions_tensor`
+ `_stable_argsort_ascending` + graph-safe threading), `cuda_graph.py`,
`validator.py`, `deepseek_v2.py`, `serve_double_sparsity.sh`,
`test_scorer_variants.py` (24-combo matrix + anchor replay no-alloc + guard tests),
`m6_anchor_graphsafe_finding.md` (new), `mmlu_{dsa,default,hybrid}_graph.json`
(data_dir patch). Commit `e7cf1f146` (pushed).

## Remaining items (queued, justified)
- **AC-4 lifted-budget** (task13–17): the opt-in Tier-2.A adjustable-budget decode
  (the oracle gate justifies bounded Tier-2.A) — the next major workstream.
- **AC-6 perf consolidation (conc-1/16 TTFT/decode-TPS/mem) + final strategic-gate
  supersession decision record** (task19–20): the end milestone.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-tensorize-per-row-eviction-for-graph-safe
- Notes: porting a per-row Python eviction/force-include loop to a graph-safe
  tensor op (fixed shape, no `.item()`/host sync) — fuzz it bit-identical against
  the original per-row reference before swapping, and reuse the deterministic
  (score-asc, position-asc) tie-break so the stable Python `list.sort` order is
  reproduced exactly.

## Goal Tracker Update Request
- **AC-3 anchor graph-safe follow-up** → done (R9); **AC-3 variant coverage complete.**
- **Resolve queued** "MMLU data_dir" (patched R9).
- **Keep Active**: AC-4 (task13–17), AC-6/task19–20 (perf + final decision record).
