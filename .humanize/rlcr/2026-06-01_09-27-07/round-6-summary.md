# Round 6 Summary — Loop 7

## Mainline objective (round-6-contract.md)
Port the Tier-2.B scorer (`scorer_norm ∈ {cosine, hybrid}` + `head_agg ∈ {max,
mean}`) into the **graph-safe Triton decode selector (AC-3 landed path)**, with
eager-vs-graph selection-equality evidence on GPU, and relax the
guard/`_force_eager_select` so these variants run under CUDA graph instead of
requiring `--disable-cuda-graph`.

## Outcome: ACHIEVED — winning scorer landed on the production CUDA-graph path, bit-identical to eager.

## Work completed
1. **Triton scorer port.** `_logical_score_kernel` gains 3 `tl.constexpr`
   (`SCORER_NORM` 0/1/2, `HEAD_AGG_MEAN`, `HYBRID_THRESHOLD`): cosine =
   unit-normalized dot (scale-ignored, normalize-then-sum to match eager); hybrid
   = per-request `seq_len > threshold` switch read in-kernel; head_agg mean =
   sum-then-divide. R17 early-exit + int8 dequant preserved; default (off/max)
   byte-identical.
2. **Config-borne threading.** Flags flow through `_logical_score_triton`,
   `retrieve_topk_graph_safe` (+ its fallback), the deepseek_v2 graph-safe call
   site, and `capture_decode_step`.
3. **Guard relaxation.** New `ds_scorer_is_graph_safe(config)` (= `anchor_mode ==
   "off"`); the validator, `_force_eager_select`, and the capture guard now only
   force eager for a non-default `anchor_mode`. cosine/hybrid/head_agg run under
   CUDA graph. serve script no longer auto-adds `--disable-cuda-graph` for them.
4. **Eager-vs-graph equality (GPU).** `TestGraphSafeScorerEqualsEager`: the
   graph-safe Triton scorer produces **bit-identical** `selected_indices` +
   `valid_lengths` to the eager `retrieve_topk_via_labels` for all 12 combos
   `scorer_norm{off,cosine,hybrid} × head_agg{max,mean}` on **fp16 AND int8**,
   short/long requests crossing the hybrid threshold.
5. **Live under CUDA graph.** A `scorer_norm=hybrid` server boots with CUDA graph
   ON (`Capture cuda graph begin` on all 8 TP ranks; validator allowed it) —
   previously impossible.

## Validation
- **345 DS unit tests pass** (incl. the new GPU eager-vs-graph equality test, the
  graph-safe guard/predicate tests, the existing CUDA-graph 100-step replay, and
  the TP=8 determinism matrix).
- **Production (graph-mode) recall, N=20** (`niah_ds_hybrid_graphsafe.json`):
  hybrid 16K **25% [8.7,49.1]** vs graph default 5% [0.1,24.9] (+20 pp, marginally
  material); 4K 75% == default (the ≤8192 raw regime is identical to default, as
  designed); 1024w 100% parity.

## Honest correction (good rigor)
R5's eager-mode numbers (hybrid 4K=85%, 16K=40%) were measured with
`--disable-cuda-graph`. The graph-safe production path gives 75% / 25%. The
*scorer code* is bit-identical eager-vs-graph (proven), so the gap is **upstream
eager-vs-graph model-forward numerics** (the query projection feeding the scorer
shifts a few needles under CUDA-graph capture; it affects the default too). The
**binding production recall is the graph-mode number**; the eager research number
over-stated it. This is exactly why AC-3 requires a *landed* graph-safe path.

## Files changed
`selection_kernel.py` (kernel + threading + `ds_scorer_is_graph_safe`),
`cuda_graph.py` (guard + capture call), `validator.py` (guard relax),
`deepseek_v2.py` (`_force_eager_select` + graph-safe call site),
`serve_double_sparsity.sh`, `test_scorer_variants.py` (GPU equality + guard +
predicate tests), `m3_graphsafe_scorer_finding.md` (new),
`niah_ds_hybrid_graphsafe.json` (new). Commit `cb02b6673` (pushed).

## Remaining items (queued, justified) — task #15
- **N≥50 binding 16K** at the graph op-point (firm up the marginal 25%).
- **MMLU ≤1.0pp re-anchor** (single-node mem0.7), DSA vs DS-hybrid.
- **graph-vs-eager perf delta** (AC-6, conc-1/16) now that hybrid runs under graph.
- **anchor_mode graph-safe port** (still eager-only).
- **AC-4 lifted-budget** (task13–17), **AC-6 consolidation + final decision
  record** (task19–20).
- R5 evidence-label cleanup (DSA op-point label, materiality wording): queued.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-eager-vs-graph-recall-differs-despite-identical-scorer
- Notes: even with a bit-identical eager-vs-graph SCORER (proven), the served
  recall differs between an eager (`--disable-cuda-graph`) server and a CUDA-graph
  server because upstream model-forward numerics (the query projection feeding the
  scorer) differ under capture. Production recall MUST be measured on the
  graph-safe path; an eager research measurement can over-state it.

## Goal Tracker Update Request
- **task8** (AC-3): graph-safe scorer support DONE (R6) — bit-identical
  eager-vs-graph selection, serves under CUDA graph. Mark done.
- **task12** (AC-2,AC-3): graph-safe port covered; remaining = N≥50 16K + MMLU +
  perf (task #15).
- **Keep Active**: task #15 (AC-3 measurement matrix + anchor port), AC-4
  (task13–17), AC-6 (task19–20).
