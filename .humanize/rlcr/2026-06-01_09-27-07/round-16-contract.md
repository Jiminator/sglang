# Round 16 Contract — Loop 7

## Mainline objective (EXACTLY ONE)
**task16 (part 1) — land the graph-safe lifted-budget decode *primitives* and prove
them zero-alloc under real CUDA-graph capture/replay + eager-equivalent.**

The R15 review (STALLED) overrides the deferred-with-evidence branch and requires
task16 production hardening to be implemented. The load-bearing graph-safety
problem is the **dynamic shape**: the current `build_lifted_compact_kv` produces a
`total_valid`-length compact buffer that varies per decode step (uncapturable), and
`dequantize_k_cache_paged` allocates internally. This round lands the two primitives
that fix that and proves them on GPU:

1. **`dequantize_k_cache_paged_out(quant_k_cache, page_table_1_flattened, out, group_size=128)`**
   — alloc-free dequant writing into a caller-owned bf16 `out`; the existing
   allocating `dequantize_k_cache_paged` becomes a thin wrapper around it.
2. **A fixed-shape, fully-tensorized graph-safe compact builder** that writes into
   preallocated scratch (`page_table_1_flattened_scratch [max_bs*lifted_budget_top_k]`,
   `compact_indices_scratch [max_bs, lifted_budget_top_k]`, `valid_counts_scratch
   [max_bs]`) with **no `.item()` / no dynamic boolean-mask shapes**: invalid and
   within-row-duplicate lanes write a **safe in-bounds physical slot** into the
   dequant input (never `-1`) and `-1` into `compact_indices` (so
   `flash_mla_sparse_fwd` masks them). It attends exactly the same valid slots per
   request as the eager `build_compact_decode_index` (same attention result), just
   in a fixed `max_bs*lifted_budget_top_k` layout.

## Target AC(s)
- **AC-4** (task16 production hardening) + **AC-6** (the graph-safe/perf path).

## Blocking issues (truly block the mainline)
- **None.** New standalone functions + a new test; no existing runtime path changes
  this round (the backend still uses the eager builder; the validator still requires
  `--disable-cuda-graph`), so default/eager paths stay byte-identical.

## Queued — explicitly OUT of scope this round (NOT closed/deferred; task16-part-2)
- **Backend / cuda-graph-runner wiring**: add the lifted scratch to `DSGraphState` /
  `allocate_graph_state`, route `_forward_lifted_budget` through the scratch-backed
  fixed path (with a preallocated q head-padding scratch) under capture, keeping the
  eager path for non-graph runs. **Next round.**
- **Validator relax** (`--disable-cuda-graph` rejection) — only after the live server
  captures + replays the lifted decode. **Next round.**
- **Live graph-mode 4K recall re-measure + perf/memory** (the eager 95% is not the
  graph number) and the **task17 production-ready disposition rewrite**. **Next round.**
- **task19 (AC-6 perf consolidation), task20 (AC-2 final decision record)** — after AC-4.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## Concrete success criteria
1. `dequantize_k_cache_paged_out` exists; `dequantize_k_cache_paged` wraps it;
   **byte-identical** output to the pre-change function on a deterministic fp8 case
   (equivalence test).
2. The fixed-shape graph-safe compact builder exists, is fully tensorized (no host
   sync), writes into caller scratch, and is **attention-equivalent** to the eager
   `build_compact_decode_index` (same per-request valid-slot set after dedup) on
   prefix-sharing, duplicate-slot, pad, and `valid_lengths < width` cases at widths
   4096 and 8192.
3. **Standalone CUDA-graph zero-alloc proof (GPU)**: capture the fixed-shape lifted
   decode primitive chain (fixed builder → `dequantize_k_cache_paged_out` into a
   scratch compact buffer → `flash_mla_sparse_fwd`) in a real `torch.cuda.CUDAGraph`,
   replay it, and assert **zero new allocations** (`assert_no_alloc_in_region`) at
   4096 and 8192, with the replay output matching the eager `_forward_lifted_budget`
   reference within tolerance.
4. **Non-regression**: default-off path byte-identical; the eager lifted path
   unchanged; the default `flashmla_kv` `dsa_index_topk` assert untouched; full DS
   unit suite passes; no new plan-marker leakage in production code.
5. `m7`/`m9` updated to reflect the graph-safe primitives landed + the remaining
   part-2 scope; `goal-tracker.md` updated (task16 in progress: primitives done,
   integration/validator/live remain); commit.

## Tag routing
- task16 is a **`coding`** task → Claude executes directly.
