# Ask Codex Input

## Question

You are acting as an independent expert reviewer/author for "Loop 6" of a Double Sparsity (DS) engineering effort on DeepSeek-V3.2 FP8, served by SGLang on a single node of 8xH200 (TP=8), at a fixed "Option B" operating point (fp8 KV, page_size=64, flashmla_kv prefill+decode, overlap-schedule + piecewise-cuda-graph disabled, radix-on via a config-bound fixture). Produce TWO markdown documents (specs below). These are analysis/decision deliverables — do NOT modify any code. Return both documents in full, each fenced by the exact delimiters I give, so I can split them.

=== SHARED BACKGROUND (all numbers are REAL, taken from source + Loop-5 hardware logs) ===

Why Loop 6 exists: DS already beats the client's per-request throughput SLO (>=30 TPS/req: Loop-5 measured p50 34.0/33.9/33.9 tok/s at conc 16/32/64) but MISSES the tail-latency SLO (P99 TTFT < 22 s: Loop-5 measured 57.7/132.9/292.0 s at conc 16/32/64). The TTFT miss is NOT a speed problem — it is an ADMISSION/QUEUE problem. DS reserves a per-rank GPU "TokenLabelTable" on top of the ~84 GB/rank V3.2 FP8 weights, which forces a small mem_fraction_static (0.6) and hence a small KV pool, so DS admits only 14.5/24.6/35.7 of nominal conc 16/32/64 (DSA admits ~16/32/64). Requests queue -> P99 TTFT explodes. Raising mem_fraction_static past 0.6 currently OOMs DS during generation. The Loop-6 spine: shrink the TokenLabelTable footprint so DS boots at a higher mem_fraction_static WITHOUT generation-time OOM, restoring admission so TTFT falls toward < 22 s.

TokenLabelTable footprint formula (from python/sglang/srt/layers/attention/double_sparsity/token_label_table.py):
  table_bytes_per_rank = num_layers_local * max_tokens * num_heads_local * label_dim * elem_size
  For V3.2 @ TP=8: num_layers_local=61, num_heads_local=128/8=16, label_dim=16, elem_size=2 (fp16).
  max_tokens = max_total_num_tokens + page_size(64).  The signatures tensor is [L, max_tokens, H_local, label_dim] fp16; there is also a `written` bool[L, max_tokens] (negligible). The table is allocated AFTER weights + KV pool, from the runtime headroom — it is NOT part of the mem_fraction_static "static" budget. So raising mem_fraction_static grows the KV pool (max_total_num_tokens), which grows the table PROPORTIONALLY, which eats runtime headroom. This is a memory fixed point, not a flat "save N GB".

REAL Loop-5 hardware anchors (8xH200, 139.80 GiB/GPU; "Load weight begin avail" ~= 138 GB/rank):
  - Anchor A — mem_fraction_static=0.6, SERVES:
      token_label_table: 1.55 GB/rank  (T=53120)
      max_total_num_tokens = 53056
      "Memory pool end. avail mem = 37.78 GB"
      Admits 14.5/24.6/35.7 of nominal conc 16/32/64; DS P99 TTFT 57.7/132.9/292.0 s; DS p50 TPS 34.0/33.9/33.9 (DS/DSA TPS ratio 0.726/0.900/1.146).
  - Anchor B — mem_fraction_static ~= 0.77-0.8, BOOTS BUT GENERATION-TIME OOM:
      token_label_table: 11.52 GB/rank  (T=396160)
      max_total_num_tokens = 396096
      runtime headroom ~= 12.29 GB -> during 4096-ISL x concurrency traffic a 248 MiB alloc fails with ~52 MiB free (generation-time OOM).
  - Anchor C — mem_fraction_static=0.897, BOOT-TIME OOM (table alloc):
      KV Cache allocated: 1,072,000 tokens, 47.99 GB; "Memory pool end. avail mem = 7.20 GB"; DS then tries to allocate the table and dies: "TokenLabelTable allocation failed: CUDA out of memory. Tried to allocate 31.18 GiB ... 7.20 GiB free." (Check: 61*1072064*16*16*2 = 31.18 GiB. OK)
  - DSA baseline (no table) runs mem_fraction_static=0.85, KV pool ~910K-1072K tokens, admits conc 16/32/64 fully, P99 TTFT 0.73/1.37/2.04 s.

DS recall / kernel-lock facts (for the decision doc):
  - DS top_k is KERNEL-LOCKED to the model's native DSA index_topk = 2048. The shared flashmla_kv DECODE kernel asserts `indices.shape[-1] == self.dsa_index_topk` (python/sglang/srt/layers/attention/dsa_backend.py, _forward_flashmla_kv, ~line 2148) during CUDA-graph capture; even SGLANG_DS_ALLOW_TOPK_MISMATCH=1 does not bypass it. So you CANNOT widen DS top_k on V3.2 without a NEW decode kernel.
  - DS NIAH recall at top_k=2048: 4K=75%, 16K=5%, 64K=0% (64K also unservable at mem-0.6: a 69970-token prompt > pool 53056 -> HTTP 400). DSA recalls 100% at EVERY length using the SAME 2048 budget + SAME kernel. Dense DS (seq <= 2048) recall = 100%. => The DS<->DSA recall gap is purely SELECTION QUALITY (V3.2's trained DSA indexer places the needle inside its 2048; DS's OFFLINE channel-mask selector does not), NOT budget size, and a larger top_k is not an available lever.

int8 compaction model (for the budget): replace fp16 signatures (2 bytes) with symmetric int8 (1 byte) at the SAME label_dim, plus one per-(layer, slot, head) scale (fp16) applied at scoring. Per-element: int8 sig = 0.5x the fp16 sig. Scale overhead per (L,slot,H) is one fp16 = 2 bytes per 16-dim int8 vector = 2/(16*1) = 0.125x the int8-sig bytes. So int8+scale = 0.5*(1+0.125) = 0.5625x the fp16 table -> net win ~= 1.78x. (Confirm/critique this and fold in the larger-pool feedback.)
Structural escalation lever (page-level / two-stage): store one signature per PAGE (64 tokens) instead of per token -> ~64x fewer rows on the page axis (the token_label_table.py docstring cites "~480 MB for the page-level table" vs ~8 GB token-level worst case). This changes selector granularity (select pages then refine) so it is held to NIAH NON-REGRESSION, not bitwise equivalence.

Plan-fixed constraints you must respect:
  - The feasibility budget is AUTHORITATIVE and BINDING on the implementation. Evaluate int8-same-label_dim FIRST on paper; if int8 is predicted INSUFFICIENT to restore nominal conc-64 admission WITH generation headroom, the implementation builds the page-level/two-stage lever DIRECTLY (no throwaway int8 build). int8 is implemented only if predicted sufficient. Narrowing label_dim is explicitly NOT a choice.
  - The real success target is ENOUGH admitted KV-pool capacity / concurrency WITH HBM headroom to move TTFT toward < 22 s — NOT "mem_fraction_static = 0.8" as a number in itself. 0.8 is the nominal validation target; 0.7 acceptable as a conservative first step; the true knob is "minimum lever that admits conc 64 with generation headroom".
  - Risk to flag (do NOT hand-wave): even after admission is fixed, at conc 64 the bottleneck may shift from admission-queue to PREFILL COMPUTE (4096 ISL x 64). The budget is a MEMORY/ADMISSION budget, not a TTFT guarantee; the downstream client-SLO benchmark must attribute admission-wait vs prefill-compute. Say this explicitly.
  - The TokenLabelTable work must be justified as the MINIMUM REVERSIBLE, opt-in DS fix (DSA's trained indexer already wins recall, and DSA stays the production default); the compact path is flag-gated with fp16 as the default.

=== DELIVERABLE 1 — strategic decision doc ===
Wrap EXACTLY between the lines `<<<BEGIN ds_on_v32_decision.md>>>` and `<<<END ds_on_v32_decision.md>>>`.
Write `ds_on_v32_decision.md`: a strategic-gate decision that records the resolution to PURSUE DS long-context-recall R&D on V3.2, but STRICTLY AFTER the full engineering spine (footprint reduction -> mem lift -> client-SLO -> hardening) has landed. The chosen recall-R&D direction is a CUSTOM sparse-matmul DECODE kernel that mirrors the native NSA/DSA kernel but exposes an ADJUSTABLE top_k (relaxing the `indices.shape[-1] == dsa_index_topk` hard cap), with a learned/query-aware DS selector as the SECONDARY alternative. The doc must contain: (1) the decision statement; (2) the rationale built on the kernel-lock + selection-quality evidence above (index_topk=2048 lock; DS recall 75/5/0 vs DSA 100 at the same budget+kernel; dense=100% proves DS decode is sound; the gap is selection quality vs the trained DSA indexer); (3) why a top_k>2048 relaxation needs a custom kernel (not a config tweak) and why a learned selector is the alternative; (4) the explicit recall-R&D SEQUENCING / consequence: it is gated behind this doc AND behind a landed engineering spine, must not block or regress the spine, and is legitimately deferrable to its own loop; (5) a short note that on a model WITHOUT a trained sparse indexer DS's value proposition is stronger (relevant to deferred GLM-5.1 / 128k work). Keep it tight and engineering-grade.

=== DELIVERABLE 2 — feasibility budget ===
Wrap EXACTLY between `<<<BEGIN footprint_feasibility.md>>>` and `<<<END footprint_feasibility.md>>>`.
Write `footprint_feasibility.md`: an HBM-fixed-point footprint feasibility budget. It MUST:
  (a) State the table formula and the 3 real anchors (A/B/C) above as the empirical basis, and derive an approximate per-token KV cost + the (mem_fraction_static -> max_total_num_tokens -> table bytes -> runtime headroom) relationship from them. Be explicit that the table competes with generation headroom, not with the static budget.
  (b) Compute the ADMISSION TARGET: from Anchor A (max_total 53056 admits 35.7 of conc 64), derive the per-admitted-request effective token footprint and the minimum max_total_num_tokens needed to admit nominal conc 64, plus a headroom target (e.g. to admit 64 with margin). Show the arithmetic.
  (c) For EACH lever — (i) no-code baseline: fp16 table + just raising mem_fraction_static; (ii) int8-same-label_dim (primary); (iii) page-level/two-stage (structural escalation) — give: freed-HBM math, scale/overhead, the mem_fraction_static needed to hit the admission-target pool, the resulting table bytes at that pool, the predicted runtime/generation headroom (cross-checked against the Anchor-B generation-OOM at ~12.29 GB headroom and Anchor-A serving at ~36 GB free-after-table), and the predicted achieved concurrency at conc 64.
  (d) Make the BINDING lever decision: state whether int8 is PREDICTED SUFFICIENT to restore nominal conc-64 admission WITH generation headroom. If yes -> select int8 and say the implementation builds int8. If no -> select page-level/two-stage directly. Justify against the negative tests: the budget must NOT omit scale-storage overhead and must NOT ignore the larger-pool feedback. Critically and honestly assess whether the no-code fp16-at-lower-mem_fraction baseline already suffices (the true "minimum lever") given Anchor A serves at 0.6 and Loop-5 found 0.7 gen-OOMs — i.e. is there an fp16 mem_fraction window between 0.6 and 0.7 that admits conc 64 with headroom, and if so, recommend the hardware mem-fraction sweep test that cheapest path FIRST before committing to int8. (The int8 lever is the chosen primary compaction lever; the fp16-f-bump is the zero-cost baseline that the budget must rule in or out.)
  (e) Name the PRIMARY selection-equivalence metric and its numeric fail threshold that the int8 implementation will be held to: top-k overlap@2048 >= 0.99 vs the fp16 baseline on a synthetic shape (state it as the binding number); list secondary diagnostics (selected-token recall, score-error distribution) as recorded-only.
  (f) Give the one-line MINIMUM-REVERSIBLE-OPT-IN justification for the table work.
  (g) Add an explicit caveat that this is a PREDICTED budget; the mem-fraction sweep + full HBM accounting (NVML/torch reserved+allocated residual) + no-OOM long-generate is the hardware confirmation, and that a conc-64 TTFT miss may be prefill-compute-bound (needs admission-vs-prefill attribution) rather than admission-bound.

Be rigorous and quantitative; show arithmetic; do not pad. If any of my numbers look inconsistent, say so and reason from the anchors.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 5400s
- Timestamp: 2026-05-30_06-38-15
- Tool: codex
