# Round 9 Contract

## Mainline Objective
**Port the anchor-budget variant to the graph-safe path (AC-3 completion).**
Implement a tensorized, fixed-shape, allocation-free post-topK force-include for
`anchor_mode ∈ {recency, global, strided}` (with `anchor_budget`) that is
**bit-identical to the eager `_force_include_anchor`**, wire it into
`retrieve_topk_graph_safe` (threaded through the deepseek_v2 graph-safe call site
+ `capture_decode_step`), and relax `ds_scorer_is_graph_safe` / the validator /
the capture guard so a non-default `anchor_mode` no longer requires
`--disable-cuda-graph`. This completes AC-3's third variant (anchor-budget) on the
production CUDA-graph path, alongside the R6 scorer_norm/head_agg port.

## Target ACs (1–2)
- **AC-3** (primary): anchor-budget variant flag-gated + graph-safe + non-regressing
  (default byte-identical when off; eager-vs-graph selection equality; TP=8 + replay
  no-alloc).

## Blocking Side Issues In Scope (the objective itself)
- **Bit-identical eviction semantics.** The eager path evicts the k lowest-score
  non-anchor selected positions (stable tie-break = position-ascending among equal
  scores), inserts the first k missing anchors (position-ascending), and re-sorts —
  per request with `effective_budget = min(anchor_budget, valid_count, seq_len)`
  and strided's set-dedup. The graph-safe version must reproduce this exactly,
  tensorized (no Python per-row loop, no `.item()`, no per-call alloc). In scope:
  match these semantics or document any precise residual.

## Queued Side Issues Out Of Scope (justified)
- **AC-4 lifted-budget** (task13–17): the major Tier-2.A workstream — next.
- **AC-6 perf consolidation + final strategic-gate decision record** (task19–20):
  end milestone; needs AC-3 + AC-4.
- **MMLU `data_dir` field missing from the committed JSONs** (Codex queued #2): a
  cheap evidence patch — bundle this round.
- **Stride-sink provenance note** (Codex queued #1), plan-marker cleanup: pre-merge.

## Round Success Criteria
- A graph-safe tensorized anchor force-include (in `selection_kernel.py`):
  fixed-shape, no host syncs, no per-call allocation; integrated into
  `retrieve_topk_graph_safe` after the top-K, using pre-allocated `DSGraphState`
  anchor scratch.
- **Eager-vs-graph selection equality on GPU**: the graph-safe force-include is
  **bit-identical** to `_force_include_anchor` over
  `scorer_norm{off,cosine,hybrid} × head_agg{max,mean} × anchor_mode{off,recency,
  global,strided}` for short + long requests (incl. `anchor_budget > top_k` and
  `budget ≥ seq_len` over-budget cases from R3); + CUDA-graph replay no-allocation.
- `ds_scorer_is_graph_safe` returns true for non-default `anchor_mode`; the
  validator + `_force_eager_select` + capture guard no longer force eager for
  anchor; `anchor_mode`/`anchor_budget` threaded through the graph-safe call sites.
  Default (anchor off) byte-identical.
- `anchor_mode`/`anchor_budget` graph-safe scratch added to `allocate_graph_state`
  (alloc-free under replay). All DS unit tests pass + a new graph-safe anchor
  equality test. Bundled: MMLU JSON `data_dir` patch. Committed + pushed;
  goal-tracker + round-9-summary updated.
