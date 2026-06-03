# Ask Codex Input

## Question

You are doing a REASONABILITY REVIEW of Claude's candidate implementation plan (v1) for "Loop 7" — DeepSeek-V3.2 FP8 double-sparsity (DS) long-context recall R&D in SGLang. You already did a first-pass critique of the draft; this is the plan that resulted. Challenge it hard but fairly. Use the verified code facts below.

## Verified code facts (re-confirmed by reading the files since your first pass)
- DECODE always uses `flashmla_kv` with the hard `assert indices.shape[-1] == self.dsa_index_topk` (dsa_backend.py ~2150). `dsa_index_topk`=2048 for V3.2.
- `flash_mla_sparse_fwd` (`_forward_flashmla_sparse` ~2064) has NO such assert (accepts variable topk), but for fp8 KV it needs `dequantize_k_cache_paged` and is currently wired only for PREFILL.
- DS selector path: selector.py::retrieve_topk -> selection_kernel.py::retrieve_topk_via_labels -> compute_token_scores (query·channels w/ channel_weights) -> all_reduce_token_scores (TP) -> select_topk_sequence_order (torch.topk+sort, R23 deterministic tie-break in _topk_by_score_then_pos). max_top_k=config.top_k. NO kernel change to improve scoring within 2048.
- config.py EXPLICITLY rejects `selection_mode`/`top_p`/`min_top_k`/`max_top_k` (Twilight ABI deferred to Loop 11). So "pull top-p forward" needs a config-ABI change, not just a scorer swap.
- metrics.py::record_selection exists -> telemetry has a home. channel_mask.py::startup_sanity_probe is a unit NIAH-min probe. test/manual/test_double_sparsity_v32.py is the server NIAH harness (emits ac12_niah_<len>_*.json). serve_double_sparsity.sh builds DS_CONFIG with top_k/page_size/channel_mask_path/device_buffer_size/signature_dtype.
- KEY EVIDENCE (runs/20260528_dsv32_mvp/ac12_analysis.md): at 4K words DS selects ~50% of tokens yet recalls only 75%; 16K ~12.5% selected -> 5%; 64K unservable at mem 0.6. i.e. even selecting HALF the tokens, DS misses 25% of needles -> strong evidence the gap is scorer-limited, not purely budget-limited.
- Strategic gate (ds_on_v32_decision.md) names Tier-2.A (adjustable-topk kernel) as the "selected direction" yet its own rationale says "the gap is selection quality ... not raw sparse attention budget alone" and a better selector "may close recall without widening top_k."

## Candidate Plan v1 (summary)

GOAL: Empirically diagnose then close-or-characterize the DS long-context recall gap on DSv3.2 FP8, by (1) instrumenting the DS selector to measure WHY the needle is missed (budget-limited vs scorer-limited), (2) running the cheaper selection-quality uplift (Tier-2.B) first, (3) gating the heavy adjustable-budget decode path (Tier-2.A) behind evidence that budget is the wall, (4) optionally extending servability to 128k (Tier-2.C). No regression to the Tier-1 spine or the DSA default path.

MILESTONES:
- M0 MEASURE-FIRST (the pivot): add flag-gated selector telemetry — per-request needle-rank, selected_contains_needle, valid_lengths, recall@K for K in {512,1024,2048,4096,8192}, score distribution — and reproduce baseline 75/5/0 on 8xH200 TP=8. This evidence decides A-vs-B. Off the hot path by default (no perf regression when disabled).
- M1 Tier-2.B selection-quality uplift (no kernel ABI change): flag-gated improved/query-aware/learned scorer within 2048, channel-mask default; optionally hybrid anchors (recency/global/strided) within 2048. Measure recall delta + MMLU + within-budget parity non-regression.
- M2 (CONDITIONAL on M0 showing needle-rank often in (2048, 8192]): Tier-2.A opt-in adjustable-budget decode. Prototype via flash_mla_sparse_fwd + dequantize_k_cache_paged as an opt-in DS decode path with a FIXED configured max_top_k (4096/8192) and padded entries (NOT dynamic shapes); default DSA flashmla_kv assert untouched; top_k>2048 fails fast unless opt-in path selected; CUDA-graph-safe + zero-alloc replay; kernel-correctness vs reference within tolerance; R23 tie-break preserved.
- M3 (secondary, separable) Tier-2.C 128k servability: KV-budget/admission so 128k /generate serves at lifted op-point (mem_fraction_static=0.7) OR document new ceiling. Kept separate from recall measurement.
- M4 Consolidation: DS-vs-DSA recall report same node; Tier-1 spine + AC-5 directional non-regression; decision record on budget-vs-scorer.

ACCEPTANCE CRITERIA (each will get positive/negative tests):
- AC-1 Selector instrumentation + baseline reproduction (needle-rank histograms + recall@K curves; reproduce 75/5/0; telemetry off-hot-path by default).
- AC-2 Recall uplift measured (delta vs 75/5/0, DS-vs-DSA same node). Floor=recorded+characterized; stretch=strict target. Gate hardness = pending decision.
- AC-3 (Tier-2.B) flag-gated selector, channel-mask default, NIAH non-regression (not bitwise), MMLU non-regression, dense-DS within-budget recall parity preserved.
- AC-4 (Tier-2.A, if pursued) opt-in adjustable-budget decode: CUDA-graph-safe, zero-alloc replay, fixed configured max_top_k+padding, default DSA assert untouched, top_k>2048 fails fast w/o opt-in, fp16/DSA default unchanged, R23 tie-break preserved, kernel-correctness vs reference within tolerance.
- AC-5 (Tier-2.C, if in scope) 128k /generate serves (no HTTP 400) at lifted op-point OR new ceiling documented; servability separated from recall.
- AC-6 No Tier-1 regression: Loop-6 admission/TTFT spine + directional AC-5 conc-16 hold; DSA/fp16 defaults unchanged; perf guardrails (TTFT, decode TPS, mem at conc-1/16) recorded and within agreed budget.

PENDING DECISIONS:
- DEC-1 Which leads: measure-first -> B -> A-if-evidence (Claude+evidence) vs A-primary (draft/gate). 
- DEC-2 Recall gate hardness: strict target vs DEC-3-style recorded directional.
- DEC-3 128k servability (Tier-2.C) in this loop or its own.
- DEC-4 Production-ready code vs measured-evidence-sufficient.
- DEC-5 Learned artifacts allowed (DSA-teacher distillation/new calibration)? (owner standing pref: theory-over-pragmatism).
- DEC-6 Slower opt-in research decode acceptable, or must preserve Loop-6 throughput?

## Your task
Output EXACTLY these sections:
AGREE: points accepted as reasonable.
DISAGREE: points you consider unreasonable, and why (be specific — e.g. is making Tier-2.B "learned" too heavy for one loop? is M0 telemetry over/under-specified? is the A-vs-B reframe right given the draft makes A primary? does the flash_mla_sparse_fwd prototype have a fatal flaw I'm missing?).
REQUIRED_CHANGES: must-fix items before this plan converges.
OPTIONAL_IMPROVEMENTS: non-blocking improvements.
UNRESOLVED: opposite opinions that need a human (owner) decision.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-01_01-31-07
- Tool: codex
