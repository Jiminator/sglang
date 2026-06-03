# Ask Codex Input

## Question

You are a senior LLM-inference systems reviewer doing a FIRST-PASS planning critique of a design draft, BEFORE a detailed plan is written. Repository: SGLang (high-performance LLM serving). The work is "Loop 7" of an ongoing DeepSeek-V3.2 (DSv3.2) FP8 double-sparsity (DS) R&D effort on 8xH200 (TP=8).

## Repository context (verified by reading the code)

DSv3.2 supports two attention selection regimes that share the SAME sparse decode kernel:
- DSA = the model's native, *trained* indexer (places a needle inside a 2048-token selection budget reliably).
- DS  = "double sparsity": an *offline channel-mask* selector (calibrated channel-importance projection), NOT trained per-query.

Both feed a top-k set of KV indices into a FlashMLA decode kernel. The DS recall problem: on NIAH (needle-in-a-haystack), DS recalls 4K=75% / 16K=5% / 64K=0%, while DSA recalls 100% at every length using the SAME 2048 budget and SAME decode kernel. Dense DS (seq<=2048) recalls 100% and DS MMLU == DSA MMLU (89.00%), so decode math is sound — the gap is *selection quality* + *budget cap*.

### Verified code facts (I read these files)

1. Decode kernel cap (the 2048 wall). `python/sglang/srt/layers/attention/dsa_backend.py`, `_forward_flashmla_kv` (~line 2113) calls `flash_mla_with_kvcache(...)` and asserts:
   `assert indices.shape[-1] == self.dsa_index_topk  # requirement of FlashMLA decode kernel`
   `dsa_index_topk` comes from `get_dsa_index_topk(hf_config)` = 2048 for V3.2. The env `SGLANG_DS_ALLOW_TOPK_MISMATCH=1` does NOT bypass this. This is the kernel-locked budget cap. The DECODE path always uses `flashmla_kv` (`self.dsa_decode_impl == "flashmla_kv"`).

2. An ALTERNATIVE sparse decode primitive already exists. `_forward_flashmla_sparse` (~line 2064) calls `flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v)` and has NO `indices.shape[-1] == dsa_index_topk` assert — it accepts variable-`topk` indices shaped `(s_q, h_kv=1, topk)`. HOWEVER: for fp8 KV it is used with a `dequantize_k_cache_paged` step (see comment "flashmla_kv vs flashmla_sparse + dequantize_k_cache_paged"), and it is currently wired only for PREFILL (EXTEND), not decode. So routing a *larger-budget opt-in DS decode* through `flash_mla_sparse_fwd` + dequantize is a plausible cheaper path than authoring a brand-new CUDA kernel — at the cost of a dequantize pass and unproven decode-path perf/CUDA-graph behavior.

3. DS selector (Tier-2.B surface). `double_sparsity/selector.py::DoubleSparsitySelector.retrieve_topk` -> `selection_kernel.py::retrieve_topk_via_labels` -> `compute_token_scores` (query projected onto offline-selected channels with channel_weights) -> `all_reduce_token_scores` (TP all-reduce) -> `select_topk_sequence_order` (`torch.topk` + `torch.sort`, deterministic tie-break in `_topk_by_score_then_pos`, "R23 deterministic tie-break"). `max_top_k = config.top_k`. Labels are written by `token_label_write.py` (channel selection of projected K_nope, optional int8 quant). All of this stays within the locked 2048 ABI — a learned/query-aware scorer or a top-p/nucleus selection rule slots in here with NO kernel change.

4. The selector produces `[bs, max_top_k]` int32 indices; if `max_top_k > 2048`, the `flashmla_kv` assert fires. So budget size and decode kernel are COUPLED: Tier-2.B (better selection within 2048) needs no kernel; Tier-2.A (budget > 2048) needs a decode path that consumes more indices.

## The draft to critique

--- DRAFT START ---
# Loop 7 — DS Long-Context Recall R&D (Tier-2 / AC-10), high-priority carryover from Loop 6

## Objective
Close the DS long-context recall gap on DSv3.2 FP8. DS recalls 4K=75% / 16K=5% / 64K=0% (NIAH) vs DSA 100% at every length using the same 2048 budget + same decode kernel. Make DS competitive on recall. Loop 6 fixed admission/TTFT (Tier-1 spine), not selection quality.

## Recall root cause (established Loop 6)
DS decode is sound (dense DS recalls 100%, DS MMLU == DSA MMLU). Two compounding limits:
1. Selection budget kernel-locked at index_topk=2048 (the flashmla_kv assert above). DS cannot spend >2048 tokens on a hard/long prompt without a new decode kernel.
2. DS offline channel-mask selector is inferior to V3.2's trained DSA indexer at the same 2048 budget. Selection-quality gap, not budget-size per se — but they interact.

## Scope IN — two R&D directions + one engineering item
- Tier-2.A PRIMARY: adjustable-`top_k` sparse decode kernel. A flashmla_kv-style decode kernel (mirroring native NSA/DSA sparse-matmul decode) exposing adjustable top_k by relaxing the `indices.shape[-1]==dsa_index_topk` cap — as a NEW, opt-in DS decode path, NOT by weakening the assert on the default DSA path. Lets DS spend a larger budget (e.g. 4096/8192) on long prompts. The only lever when the 2048 cap itself is the wall. Heavy: CUDA-graph-safe sparse-attention decode with its own fixed-shape ABI.
- Tier-2.B SECONDARY: learned / query-aware DS selector that places the needle inside the existing 2048 budget better than the offline channel-mask projection — NO kernel change (stays in locked ABI), cheaper to try first as a recall-uplift probe. Candidates: lightweight learned scorer, or pulling top-p/nucleus selection (Twilight, roadmap Loop 11) forward — top-p can spend more of the 2048 adaptively.
- Tier-2.C secondary engineering: 128k servability. Extend Loop-6 64K servability to 128k context (KV-budget/admission to SERVE 128k). Servability is separate from recall; both needed for the 128k deliverable.

## Scope OUT
- Re-litigating the Tier-1 spine (DS int8 / mem_fraction_static=0.7 / radix-on / TP=8 op-point + directional AC-5 stand as the Loop-7 baseline; do not regress).
- The strict all-concurrency client SLO (P99 TTFT<22s AND >=30 TPS/req at every conc) — separate downstream op-point/DSA-side question (DS per-request decode TPS is structurally <= DSA; conc-64 >=30 unattainable even for DSA at 29.4). In scope only if owner explicitly merges.
- GLM-5.1 / nvfp4 / multi-node / knob-compat (their own roadmap loops).

## Draft acceptance criteria (gen-plan will formalize positive/negative tests)
1. Recall uplift, measured: NIAH 4K/16K/64K recall delta vs DS baseline 75/5/0, real hardware, DS-vs-DSA same node. gen-plan sets binding uplift gate (e.g. 16K materially > 5%); a recorded+characterized result is the floor (DEC-3-style), strict recall target is the stretch.
2. (If Tier-2.A) new decode kernel CUDA-graph-safe and opt-in: bit-exact selection contract (R23 deterministic tie-break carries), zero-alloc under graph replay, default DSA path's dsa_index_topk assert untouched, fp16/DSA default unchanged.
3. (If Tier-2.B) new selector flag-gated with offline channel-mask as default; selection equivalence is NIAH non-regression, not bitwise (selector granularity changed).
4. (If Tier-2.C) 128k /generate serves (no HTTP 400) at the lifted op-point, or the new ceiling is documented.
5. No Tier-1 regression: Loop-6 admission/TTFT spine + directional AC-5 conc-16 still hold at the chosen op-point.

## Hardware/op-point
Single node 8xH200 (TP=8), V3.2 FP8, page_size 64, fp8 KV, flashmla_kv prefill+decode, overlap-schedule + piecewise-cuda-graph disabled, radix-on via config fixture, DS int8 compact table at mem_fraction_static=0.7. Reuse Loop-5/6 serve/bench scripts + NIAH harness — no new serve/bench scaffolding.

## Pending decisions (resolve in gen-plan discussion)
- Which direction leads? Tier-2.B (cheap, no-kernel, first) vs Tier-2.A (only lever if 2048 is the wall). Recommend B-as-probe then A-if-needed; A is higher-ceiling/higher-cost.
- Recall gate hardness: strict recall target vs DEC-3-style recorded directional uplift.
- Does 128k servability (Tier-2.C) belong here or its own loop?
- Theory-over-pragmatism (standing owner preference): prefer theoretically correct adjustable-budget/learned-selector design over a cheap hack even at higher engineering cost.
--- DRAFT END ---

## Your task

Critique assumptions, find missing requirements, and propose stronger plan directions. Be concrete and SGLang-specific; use the verified code facts. Output EXACTLY these labeled sections:

CORE_RISKS: highest-risk assumptions and failure modes (e.g. is the root-cause attribution right? is Tier-2.A really required, or does a better selector within 2048 close 16K/64K? what could make a 4096/8192 budget NOT improve recall?)
MISSING_REQUIREMENTS: likely omitted requirements / edge cases (e.g. how is "competitive recall" defined and measured; NIAH harness determinism; whether the offline channel-mask must be re-calibrated; TP all-reduce correctness at larger budget; CUDA-graph capture with variable budget; perf/latency guardrails so a recall win doesn't tank TTFT/TPS).
TECHNICAL_GAPS: feasibility/architecture gaps (especially: is flash_mla_sparse_fwd + dequantize_k_cache_paged a viable opt-in decode path vs a from-scratch kernel; fp8 vs bf16 KV implications; CUDA-graph fixed-shape ABI with a configurable budget; how a top-p/nucleus rule keeps a CUDA-graph-safe fixed output shape).
ALTERNATIVE_DIRECTIONS: viable alternatives with tradeoffs (ordering of A vs B; B-first as a falsification probe of the budget hypothesis; reuse flash_mla_sparse_fwd vs new kernel; whether to scope Tier-2.C out).
QUESTIONS_FOR_USER: questions needing explicit human decisions.
CANDIDATE_CRITERIA: candidate acceptance criteria suggestions (with concrete, measurable, deterministic checks where possible).

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-01_01-26-18
- Tool: codex
