# Round 16 Summary — Loop 7

## Mainline objective (round-16-contract.md)
**task16 (part 1) — land the graph-safe lifted-budget decode primitives and prove
them zero-alloc under real CUDA-graph capture/replay + eager-equivalent.**

(The R15 review STALLED the loop and overrode the deferred-with-evidence close,
requiring task16 production hardening to be implemented. This round lands the
technical core; the backend/cuda-graph-runner plumbing + live re-measure is part 2.)

## Outcome: ACHIEVED — the graph-safe primitives are landed and PROVEN zero-alloc on GPU.

## The blocker, and the fix
The load-bearing graph-safety problem was the **dynamic shape**:
`build_lifted_compact_kv` produced a `total_valid`-length compact buffer that varies
per decode step (uncapturable), and `dequantize_k_cache_paged` allocates internally.

## Work Completed (`coding`, Claude)
1. **`dequantize_k_cache_paged_out(quant, page_table_1_flattened, out, group_size=128)`**
   (`dsa/dequant_k_cache.py`): alloc-free dequant writing into a caller-owned bf16
   scratch (no internal `torch.empty`); the existing allocating
   `dequantize_k_cache_paged` is now a **thin wrapper** around it (byte-identical).
2. **`build_lifted_compact_index_fixed` / `build_lifted_compact_kv_fixed`**
   (`double_sparsity/lifted_budget.py`): a **fixed-shape, fully-tensorized** graph-safe
   compact builder (no `.item()`, no dynamic boolean-mask shapes). It keeps a fixed
   `[bs*lifted_width]` layout — every lane gets a compact row at ordinal
   `b*width+lane`, so the compact buffer is always `[bs*width, 1, 576]`. Masked /
   within-row-duplicate lanes write a **safe in-bounds physical slot** into the
   dequant input (never `-1`) and `-1` into the compact index (so
   `flash_mla_sparse_fwd` masks them). A request attends **exactly the same valid
   (post-dedup) slots as the eager builder — identical attention** — in a capturable
   fixed shape. `build_lifted_compact_kv_fixed` runs it + the alloc-free `out=` dequant
   into preallocated scratch.

## Files Changed
- `dsa/dequant_k_cache.py` (`+ dequantize_k_cache_paged_out`; existing API wraps it).
- `double_sparsity/lifted_budget.py` (`+ build_lifted_compact_index_fixed` / `_kv_fixed`).
- `test_lifted_budget_decode.py` (CPU fixed-layout/dedup; dequant `out=` equivalence;
  GPU CUDA-graph zero-alloc capture/replay at 4096/8192).
- `development/loop7/m7_lifted_budget_design.md` (task16 primitives DONE + part-2 scope),
  `m9_tier2a_disposition.md` (superseded banner — the deferral is overridden).
- Commit `714cf62b2` (local — loop hook keeps commits local until completion).

## Validation
- `TestLiftedCompactIndexFixed` (CPU): the fixed `b*width+lane` layout, safe-slot for
  masked lanes, `-1` compact index, within-row dedup keep-first — **1 passed**.
- `TestLiftedBudgetGraphSafe` (GPU, H200): `dequantize_k_cache_paged_out` **byte-identical**
  to the allocating dequant; and a **real `torch.cuda.CUDAGraph` capture** of
  (fixed builder → dequant `out=` into scratch → `flash_mla_sparse_fwd`) **replays
  ZERO-alloc** (`assert_no_alloc_in_region`) at **4096 and 8192** and matches the eager
  reference — **3 passed**.
- Full DS unit suite (4 files) → **345 passed + 9 subtests** (was 341; +4 R16), no
  regression. No existing runtime path changed (the backend still uses the eager
  builder; the validator still requires `--disable-cuda-graph`); default byte-identical.

## Remaining Items (active mainline — task16 part 2, next round)
- Wire the fixed scratch (incl. a **q head-padding** scratch) into `DSGraphState` /
  `allocate_graph_state`; route `_forward_lifted_budget` through the scratch-backed
  fixed path under capture (keep the eager path for non-graph runs).
- **Relax the validator `--disable-cuda-graph`** requirement for the lifted path.
- **Live** boot with CUDA graph + **graph-mode 4K recall re-measure** (the eager 95%
  is not the graph number) + perf/memory; graph-captured TP=8 determinism.
- **task17 redo**: the production-ready landing disposition after part 2 lands.
- Then **AC-6/task19** (perf consolidation) + **AC-2/task20** (final decision record).
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## AC-4 status
task16 part 1 (graph-safe primitives + zero-alloc proof) **done**; the hardest
technical risk is implemented and proven on GPU. **AC-4 stays NOT MET** until part 2
(backend integration + validator relax + live graph-mode re-measure) + the task17
production-ready disposition.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260602-flash-mla-sparse-fwd-compact-decode-contract
- Notes: added the **graph-safety corollary** — the dynamic-`total_valid` compaction
  is uncapturable; use a FIXED `[bs*width]` layout (masked/dup lanes → safe in-bounds
  slot in the dequant input + `-1` compact index) instead of compacting, paired with
  an alloc-free `dequantize_k_cache_paged_out`, and PROVE it with a real CUDAGraph
  capture + `assert_no_alloc_in_region(replay)`. A reusable graph-safety technique for
  any DS dynamic-length gather/dequant path.

## Goal Tracker
Updated directly (Plan Version 22): R16 Plan Evolution row; task16 → in progress
(primitives done R16; part-2 integration/validator/live remain); task17 redo pending.
No Goal Tracker Update Request needed.
