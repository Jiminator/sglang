# Round 17 Contract — Loop 7

## Mainline objective (EXACTLY ONE)
**task16 (part 2) — wire the R16 graph-safe primitives into the production CUDA-graph
path and prove the backend lifted decode replays zero-alloc, then relax the validator
and confirm a live CUDA-graph boot.**

When `enable_lifted_budget_decode` is set, the lifted decode must run under CUDA
graph (not `--disable-cuda-graph`). Wire the fixed-shape builder + alloc-free dequant
(R16) into the backend so `_forward_lifted_budget` is alloc-free under capture, then
remove the validator's eager-only requirement.

## Target AC(s)
- **AC-4** (lifted-budget production graph hardening) + **AC-6** (graph-safe/perf path).

## Blocking issues (truly block the mainline)
- **None.** The lifted path is opt-in/default-off; all changes are behind the lifted
  guard + new optional params, so the default DSA/DS decode stays byte-identical.

## Queued — explicitly OUT of scope this round (NOT closed/deferred)
- **task17 redo** (production-ready disposition) — after the live graph-mode evidence
  exists. Next round.
- **AC-6/task19** (full perf consolidation conc-1/16: TTFT, decode TPS/req, mem,
  graph-replay, admission) — next; a quick graph-mode recall + a memory note this round
  is enough to confirm the landing, not the full consolidated report.
- **task20** (final decision record).
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## Concrete success criteria
1. **`DSGraphState` + `allocate_graph_state`**: add lifted scratch — `lifted_page_table
   [max_bs*max_top_k] int32`, `lifted_compact_indices [max_bs, max_top_k] int32`,
   `lifted_valid_counts [max_bs] int32`, `lifted_compact_kv [max_bs*max_top_k, 1, 576]
   bf16`, `lifted_q_padded [max_bs, required_flashmla_heads, 576] bf16` — allocated
   **only when `enable_lifted_budget_decode`** is active; threaded from both metadata
   allocation sites in `dsa_backend.py`.
2. **`_forward_lifted_budget`**: when the lifted scratch is present (graph path), slice
   it to the current `bs`/`width`, call `build_lifted_compact_kv_fixed` (fixed-shape) +
   the `out=` dequant into scratch, and pass scratch compact KV/indices to FlashMLA via
   a q-padding-scratch path. Keep the eager `build_lifted_compact_kv` as the non-graph
   fallback.
3. **`_forward_flashmla_sparse`**: add an optional preallocated q-padding scratch param
   (the lifted graph path provides it; existing callers unaffected → byte-identical).
4. **Offline backend CUDAGraph proof (GPU)**: capture the wired `_forward_lifted_budget`
   (with a constructed backend + lifted `DSGraphState`) in a real `torch.cuda.CUDAGraph`,
   replay, assert **zero new allocations** at 4096 and 8192 (incl. q head padding,
   prefix sharing, duplicate slots, pad, `valid_lengths < width`), matching the eager
   reference. Plus a single-rank graph-capture of the lifted-width selection replaying
   zero-alloc.
5. **Validator relax**: remove the lifted `--disable-cuda-graph` rejection (now graph-safe);
   the default `flashmla_kv` `indices.shape[-1] == dsa_index_topk` assert stays untouched;
   update the serve script so `LIFTED_BUDGET=1` no longer forces `--disable-cuda-graph`.
6. **Live CUDA-graph confirmation (capstone)**: boot a DS-lifted-4096 server **WITHOUT
   `--disable-cuda-graph`** (bounded capture batch to keep the scratch memory sane),
   confirm it captures + serves, and re-measure the **graph-mode** 4K served recall
   (the eager 95% is not the graph number). Record it + the scratch memory cost. If the
   live capture hits an environment limit, record the blocker honestly; the wiring +
   offline zero-alloc proof + validator relax remain the binding deliverable.
7. **Non-regression**: default-off path byte-identical; full DS unit suite passes; no new
   plan-marker leakage.
8. `m7`/`goal-tracker.md` updated; commit.

## Tag routing
- task16 is a **`coding`** task → Claude executes directly.
