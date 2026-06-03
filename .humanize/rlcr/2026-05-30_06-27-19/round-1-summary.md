# Round 1 Summary — AC-3 int8-symmetric compact TokenLabelTable

## Mainline objective (round contract)
Implement **AC-3** — the int8-symmetric compact `TokenLabelTable` path (the lever the feasibility budget selected), flag-gated with **fp16 default**, **CUDA-graph-safe**, threaded through every site that touches signatures. Plus the CPU-testable AC-3/AC-6 unit tests. Blocking prerequisite (Codex R0 review): revise the AC-2 artifact so int8 is the unambiguous binding path.

## Blocking fix landed first (Codex R0 review)
Revised `runs/20260530_dsv32_loop6/footprint_feasibility.md`: the Binding Lever Decision now reads unambiguously — **build int8 same-`label_dim` for the footprint reduction**, not conditional on any prior fp16 experiment. The fp16 lower-`mem_fraction` window is demoted to *optional comparison instrumentation* logged only during the AC-4 compact-table sweep; it must not gate/replace/precede the int8 build. Anchor B label corrected `≈0.77-0.8`→`≈0.70` (matches the recorded "0.7 OOMs during generation"). Tracker reconciled (blocking issue marked resolved; Plan Evolution updated).

## What was implemented (commit `84d3410b9`)
Design: store int8 signatures `[L,T,H,D]` + a static fp16 `scales [L,T,H]` (one symmetric scale per vector). Because `score = scale[t,h]·Σ_d(q_proj[h,d]·int8_sig[t,h,d])`, dequant is a single per-head multiply by `scale[t,h]` **after** the integer dot, **before** the cross-head max — so fp16 stays zero-overhead via a `HAS_SCALE` compile-time branch.

- **config.py** — explicit allowed field `signature_dtype` (`"fp16"` default | `"int8"`); unknown-field rejection preserved.
- **token_label_table.py** — compact mode allocates int8 signatures + static fp16 `scales`; `bytes_per_rank`/`estimate_hbm_bytes` count both (measured **0.5625×**); `is_compact`; fp16 path byte-identical (no scales).
- **token_label_write.py** — symmetric per-`(slot,head)` int8 quantize-on-write (`scale = max(|label|)/127`) + scale; fp16 path unchanged; zero-vector safe (no div-by-zero); no host sync.
- **selection_kernel.py** — dequant-at-scoring in the torch refs (`compute_token_scores`, `_compute_logical_token_scores`), **both** Triton kernels (`_compute_token_scores_kernel`, `_logical_score_kernel` — added `scale_ptr` + `HAS_SCALE: tl.constexpr`), and the allocation-free `retrieve_topk_graph_safe` scratch pipeline; `token_scales` threaded through.
- **selector.py / cuda_graph.py / deepseek_v2.py / dsa_backend.py** — pass `token_scales`/`scales`; bind maps `signature_dtype`→torch dtype; DSA-default (fp16) allocates no scales. The FlashMLA `indices.shape[-1]==dsa_index_topk` assert is untouched (AC-3.3 ABI lock).

## Files changed
8 production files (config, token_label_table, token_label_write, selection_kernel, selector, cuda_graph, dsa_backend, models/deepseek_v2), 1 test file (+230 lines), and the AC-2 artifact revision. Loop state (goal-tracker, round-1 contract/summary) is in `.humanize/rlcr/` (gitignored).

## Validation — 272 DS unit tests pass (260 existing + 12 new), GPU enabled, no regression
New `TestCompactInt8Signatures`:
- config: fp16 default, int8 opt-in, invalid dtype rejected, unknown-field still rejected.
- table: fp16 → no scales; int8 → static fp16 `[L,T,H]` scales; **byte ratio exactly 0.5625×**.
- quantize-on-write round-trip (dequant within one quant step) + zero-vector safety.
- **selection-equivalence: int8-vs-fp16 top-k overlap@2048 ≥ 0.99** (torch path — the binding gate).
- **GPU (H200, ran this round):** int8 Triton kernels match the torch reference (max score err **9.5e-7**; graph-safe logical overlap **1.0**); CUDA-graph **capture+replay** of the int8 path is **allocation-free** (`assert_no_alloc_in_region`) and **replay == eager**.

The RLCR dev box exposes 2 H200s, so the Triton-kernel int8 scale logic and CUDA-graph-safety — normally AC-4 concerns — were validated on real hardware this round, de-risking the serving round.

## Remaining items (gated on this round)
- **AC-3 hardware evidence (paired with the AC-4/AC-5 hardware round, needs a live served model):** real-mask NIAH non-regression on the Loop-5 mask vs the fp16 Loop-5 DS baseline, and the compact-vs-fp16 decode-scoring microbench against the 33.9→30 TPS/req margin. These AC-3.1 positive tests require serving, so per the plan ("a code-only round is acceptable if the next round validates on hardware") they belong to the next round — **not forgotten, explicitly deferred to AC-4** (tracked as `task4: partial`).
- **AC-4 (next mainline):** boot DS with `signature_dtype=int8` on TP=8, sweep `mem_fraction_static` 0.6→0.8 with full NVML/torch-residual HBM accounting + sustained long `/generate` no-OOM. Then AC-5 client-SLO benchmark with admission-vs-prefill attribution, AC-6 opt-in/DSA-default hardware proof, AC-7/AC-8/AC-9, then gated AC-10.

## Note for review
AC-3 code + unit/GPU evidence is complete this round; the two AC-3.1 **hardware** positive tests are intentionally paired with the AC-4 serving round. The optional `code-simplifier` agent is not installed; I did a manual simplification pass instead (consolidated three near-duplicate test alloc-helpers into one).

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: AC-3 was a planned feature implemented and validated in a single round (CPU torch-path tests + real-H200 Triton/CUDA-graph validation, 272 tests green) — not a defect solved across multiple rounds, so no new lesson is warranted. Applied existing lessons as design constraints: BL-20260527-torch-topk-aliasing-corrupts-input (left the graph-safe topk pipeline untouched; only inserted the scale multiply before topk), BL-20260528-ds-radix-capture-cuda-graph-safe / the CUDA-graph host-sync rule (scale is pre-allocated, static-shaped, device-only — capture/replay verified allocation-free), and BL-20260528-ds-private-server-args-attrs-crash-ipc (scales live inside the already `_`-prefixed TokenLabelTable, so the IPC filter still covers them).
