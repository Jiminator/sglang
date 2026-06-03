# Ask Codex Input

## Question

You are a senior systems reviewer doing a FIRST-PASS planning critique of a design draft, BEFORE an implementation plan is written. Be rigorous and specific. Do not write the plan; critique the draft and propose stronger directions.

## Repository context (verified)
- Project: **SGLang**, a high-performance LLM serving framework. Working dir `/sgl-workspace/sglang`, branch `dev/double-sparsity-standalone`.
- Feature under work: **Double Sparsity (DS)** — an offline channel-mask-based sparse KV selection path, being made shippable on **DeepSeek-V3.2 (FP8)**, which ALSO ships a native *trained* sparse indexer called **DSA** (DeepSeek Sparse Attention).
- Verified key files:
  - `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py` — defines `TokenLabelTable` (fields: signatures `[L,T,H_local,label_dim]` fp16, written bool `[L,T]`), `allocate_token_label_table(...)`, `bytes_per_rank()`, `estimate_hbm_bytes(...)`, `validate_table_covers_kv_pool(...)`. Worst-case footprint ≈ `num_layers_local * max_tokens * num_heads_local * label_dim * dtype_bytes`. For V3.2 TP=8: 60 layers * 262144 slots * 16 heads * 16 label_dim * 2 bytes ≈ 8 GB/rank (fp16). The table is sized by the PHYSICAL KV slot address space `max_tokens = kv_pool.size + page_size`, not request rows.
  - `python/sglang/srt/layers/attention/double_sparsity/token_label_write.py` — `token_label_write(...)` writes `signatures[layer, cache_loc, :, :] = gather(k_nope, channel_selection_layer)`; `invalidate_token_label_slots(...)`. Quantize-on-write would slot here.
  - `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — `compute_token_scores(queries, token_signatures, ...)` returns `token_scores[bs, max_tokens]` fp32 (max-over-heads of channel-masked dot of query-projection vs signature). Triton kernel `_compute_token_scores_kernel` reads `sig_ptr` as fp16/fp32. Apply-scales for int8 dequant would slot here.
  - `python/sglang/srt/layers/attention/dsa_backend.py` — DSA + DS decode backend. At line ~2148 the FlashMLA decode path asserts `indices.shape[-1] == self.dsa_index_topk` ("requirement of FlashMLA decode kernel"). `dsa_index_topk` comes from `get_dsa_index_topk(hf_config)` (V3.2 native budget = 2048). `ds_max_top_k` defaults to 2048. This is the hard cap blocking DS selection beyond 2048 on the SHARED decode kernel.
  - `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — registered CI unit tests (large).
  - `test/manual/test_double_sparsity_v32.py` — manual NIAH/quality + within-budget gate harness (hits a live server).
  - `development/benchmark.sh`, `development/benchmark_baseline.sh`, `development/benchmark_compare.py`, `development/serve_double_sparsity.sh` — serve/bench/compare tooling at the fixed "Option B" operating point (TP=8, fp8 KV, page 64, flashmla_kv prefill+decode, overlap-schedule + piecewise-cuda-graph DISABLED, radix on). `mem_fraction_static` is the lever the loop moves.
- Client SLO (development/CLIENT_SLOS.md): DeepSeek-V3.2 FP8; **30 TPS/req with P99 TTFT < 22 s**; workload 4096 ISL / 512 OSL / max-conc 64 / min-conc 16 / ~55% cache hit; page 64; TP + cuda graphs + radix.
- Loop-5 measured state: DS per-request gen ≈ 34/33.9/33.9 TPS at conc 16/32/64 (PASSES 30 TPS). But P99 TTFT = 57.7 / 132.9 / 292.0 s (FAILS < 22 s). Root cause stated: DS must run at `mem_fraction_static=0.6` (DSA runs 0.85) because of the ~8 GB/rank TokenLabelTable on top of ~84 GB/rank V3.2 FP8 weights; small KV pool admits only 14.5/24.6/35.7 of nominal 16/32/64 concurrency → queueing → TTFT explosion. Raising mem past 0.6 currently OOMs DS during generation.
- Loop-5 NIAH recall: DS 75% / 5% / 0% at 4K/16K/64K vs DSA 100% (DS capped at 2048 budget + inferior offline selector).
- Hardware: 2-node 8xH200 cluster. Loop-5 channel mask already on disk at `/models/dsv32-fp8-channel-mask.safetensors`.

## The draft to critique
The loop's single done-criterion: DS serves the client workload (4096 ISL / 512 OSL / conc 16-64 / ~55% cache) at **absolute P99 TTFT < 22 s AND >= 30 TPS/req** on real hardware, measured as an absolute pass/fail vs the client SLO (NOT a DS-vs-DSA ratio).

The spine: shrink the per-rank TokenLabelTable footprint (int8-symmetric signatures with per-layer/slot/head scales applied at scoring, OR narrower label_dim, OR tighter slot model) so DS can boot at a higher `mem_fraction_static` (target ~0.8) without generation-time OOM, restoring admission, so TTFT drops below 22 s.

Tiers:
- Tier 1 (engineering wins, pay off regardless): (1) TokenLabelTable footprint reduction; (2) mem_fraction lift + no-OOM validation; (3) direct client-SLO validation at NUM_PROMPTS=320, conc 16/32/64; (4) AC-11 directional DS+DSA re-sweep at the lifted operating point; (5) 64K servability (HTTP 200 not 400); (6) AC-12 within-budget gate asserted from real `usage.prompt_tokens` instead of a word-count proxy.
- Tier 2 (GATED on strategic decision DEC-1): DS long-context recall R&D — either a flashmla_kv decode-kernel variant accepting `top_k > index_topk`, and/or a query-aware/learned DS selector, measured by NIAH 4K/16K/64K recall delta vs DS 75/5/0.

Strategic gate DEC-1 (decide FIRST): Is DS worth pursuing past the engineering wins on a model that already ships a trained native sparse indexer (DSA)? DS is capped at the native index_topk=2048 by the shared decode kernel AND uses an inferior offline selector, so it cannot match DSA long-context recall at the shared budget. DS value is clearer on models WITHOUT a trained indexer. A closed gate (cap at Tier 1) is a legitimate outcome.

Pending decisions in the draft: DEC-1 (pursue Tier 2?), DEC-2 ("shippable" = DS meets SLO itself vs DS available as opt-in while DSA is default), DEC-3 (confirm TTFT target is absolute P99<22s at full NUM_PROMPTS=320), DEC-4 (footprint approach + target mem_fraction 0.7/0.8 + OOM-safety bar), DEC-5 (deployment topology single-node TP=8 vs multi-node).

Constraints: Reuse Loop-5 mask + serve/bench scripts; no new scaffolding. Keep fp16 as default behind a flag until compact path has unit + hardware evidence. Must NOT change the DECIDED DS-fair AC-12 gate definition. Implementation code/comments must NOT contain plan-process markers (AC-, DEC-, Tier, Option B, Round N).

## Your required output format (use these exact section headers)
CORE_RISKS: highest-risk assumptions and potential failure modes (be concrete about WHY the int8/label_dim/admission approach could fail to reach P99<22s).
MISSING_REQUIREMENTS: likely omitted requirements, edge cases, or unstated invariants (e.g. quantization numerics, calibration of per-slot scales at write time, CUDA-graph capture safety of a quantized scoring path, mask reuse vs regeneration, prefill-compute vs admission-queue split, page-table interplay).
TECHNICAL_GAPS: feasibility/architecture gaps — is the footprint→admission→SLO causal chain sound? Could TTFT be prefill-bound at conc 64 (4096 ISL) even after admission is fixed? Is int8 quant of signatures CUDA-graph-safe given per-slot scales? Is the FlashMLA top_k>2048 relaxation realistic in one loop?
ALTERNATIVE_DIRECTIONS: viable alternatives with tradeoffs (e.g. paged/page-level label table instead of per-token, offloading the table, recompute-on-demand labels, a smaller label_dim sweep first, chunked prefill to fix prefill-bound TTFT, accepting DEC-2 opt-in framing).
QUESTIONS_FOR_USER: questions needing explicit human decisions (beyond DEC-1..DEC-5 if any new ones surface).
CANDIDATE_CRITERIA: candidate acceptance-criteria refinements (sharper positive/negative tests, especially the admission-wait vs prefill-compute breakdown for the SLO claim, and the selection-equivalence tolerance for the int8 table).

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 900s
- Timestamp: 2026-05-30_02-30-05
- Tool: codex
