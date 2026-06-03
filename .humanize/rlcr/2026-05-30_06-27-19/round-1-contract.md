# Round 1 Contract

## Mainline Objective (exactly one)
Implement **AC-3 — the int8-symmetric compact `TokenLabelTable` path** (the lever AC-2 selected), flag-gated with **fp16 as the default**, **CUDA-graph-safe**, threaded through every site that touches signatures: config → table allocation/byte-accounting → quantize-on-write → dequant-at-scoring (torch reference paths **and** both Triton kernels **and** the allocation-free `retrieve_topk_graph_safe` scratch path) → bind wiring. Plus the CPU-testable AC-3/AC-6 unit tests.

Design (low special-casing): keep `signatures` int8 `[L,T,H,D]` + a static `scales` fp16 `[L,T,H]` (one symmetric scale per 16-dim vector). Because `score = scale[t,h]·Σ_d(q_proj[h,d]·int8_sig[t,h,d])`, dequant is a single per-head multiply by `scale[t,h]` after the integer dot, before the max-over-heads. fp16 default carries **no** scales tensor and pays zero overhead (`HAS_SCALE`/`None` gating).

## Target ACs (1–2)
- **AC-3** (footprint reduction; `coding` → Claude) — primary.
- **AC-6** (opt-in / DSA-default) — only its CPU-checkable surface this round: config default = fp16, and a DSA-default (no DS flag) path allocates **no** `TokenLabelTable`. The hardware opt-in proof is AC-6's later round.

## Blocking Side Issues in Scope
1. **AC-2 sequencing drift (Codex R0 review).** Revise `runs/20260530_dsv32_loop6/footprint_feasibility.md` and the tracker so the binding path is unambiguous: **AC-3 implements int8 same-`label_dim` next**; the low-`f` fp16 sweep is at most optional instrumentation logged during the AC-4 sweep — it must **not** gate, replace, or precede the AC-3 compact-table implementation. Page-level/two-stage remains escalation-only on int8 insufficiency after hardware evidence. This blocks safe AC-3 execution and must land first.
2. **Anchor B label cleanup (queued→folded in).** In the same artifact, change Anchor B `mem_fraction_static≈0.77-0.8` → `≈0.70` so the next hardware round targets the right window.

## Queued Side Issues Out of Scope
- **AC-3 hardware evidence** — real-mask NIAH non-regression (Loop-5 mask) and the compact-vs-fp16 decode-scoring microbench against the 33.9→30 TPS margin: both need a live server, so they pair with the AC-4/AC-5 hardware round (must not delay this code round).
- **CUDA-graph capture correctness on real GPU** — exercised on hardware in AC-4 (this box runs CPU unit tests; Triton paths are guarded).
- AC-4/AC-5/AC-7/AC-8/AC-9 and the gated AC-10 — later rounds. **Do not** touch the FlashMLA `indices.shape[-1]==dsa_index_topk` decode assert anywhere (AC-3.3 ABI lock).

## Round Success Criteria
1. AC-2 artifact + tracker reconciled (int8 is the unambiguous binding AC-3 path; Anchor B label fixed); the rejected Plan-Evolution entry corrected.
2. `DoubleSparsityConfig` gains an explicit allowed field for the compact dtype, default fp16; unknown-field rejection preserved.
3. `TokenLabelTable` supports compact mode (int8 signatures + static fp16 `[L,T,H]` scales); `bytes_per_rank()`/`estimate_hbm_bytes()` count **both** and report the ~0.5625× reduction; fp16 path byte-identical to today.
4. `token_label_write` quantizes symmetric per-`(slot,head)` int8 + scale (compact), fp16 path unchanged; no host sync / dynamic allocation introduced into captured paths; zero-handling correct.
5. Dequant-at-scoring implemented in: `compute_token_scores` torch ref, `_compute_logical_token_scores` torch ref, `_compute_token_scores_kernel`, `_logical_score_kernel`, and `retrieve_topk_graph_safe` (scale scratch wired, static-shaped). No Python dtype dispatch inside captured paths.
6. Bind wiring (`finalize_double_sparsity_bind`) allocates the compact table + scales from the parsed config; DSA-default boot allocates nothing.
7. CPU unit tests pass: config parse/default; byte-count reduction; **synthetic selection-equivalence `top-k overlap@2048 ≥ 0.99`** (int8 vs fp16 via the torch reference path); DSA-default no-table. Existing DS unit suite stays green.
8. Commit + push to `jimmy`; `round-1-summary.md` with BitLesson Delta.

## Out-of-Scope Guards
- Code-only round (no hardware artifact). Acceptable **iff** the next round validates on hardware (AC-4). The ABI lock (AC-3.3) and "fp16 default until hardware-validated" both hold.
- No new fixture/scaffolding; reuse the existing unit-test harness.
