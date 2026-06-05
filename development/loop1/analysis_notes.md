# GLM-5.1-FP8 hill-climb — bottleneck analysis & sweep plan

## Independent analyst (Codex) consultation — round 0

Inputs: spec-ON baseline vs spec-OFF (both fresh server, 320 completed, 0 errors).

### Bottleneck
At concurrency 64 the model is **compute-bound** (256-expert MoE, batch 64). EAGLE's
4-draft-token verify costs more than the accept_length≈3.1 buys, so speculation
**raises sustained `mean_tpot`** (42.16 vs 37.74 ms) while **lowering median ITL**
(18.05 vs 35.63) via ~3-token bursts that deflate the median. => the official
median-ITL metric is gameable; `mean_tpot` must always be reported alongside.

### Why spec-OFF has WORSE p99 TTFT (29.6 s vs 13.3 s)
Spec-ON finishes each request in ~165 verify iterations vs ~512 plain-decode
iterations, so running requests free their slots sooner → queued cold 4096-token
prefills are admitted earlier → lower p99 TTFT. Speculation helps slot turnover
even though it is slower per accepted token.

### Honest ceiling
Reaching `mean_tpot < 33 ms` (≥30 tok/s sustained) needs ~22% below spec-ON or
~13% below honest spec-OFF. Flags-only on BF16 KV + TP8 + no-EP likely cannot get
there; **DP attention** is the one clean knob that might. If DP + lighter EAGLE
still lands >35–36 ms, the ceiling is MoE/decode compute and further scheduler-flag
tuning is futile — to be stated in the gap report.

### Corrections incorporated
- On BF16 KV, KEEP `fa3` decode. Do NOT force `flashmla_kv` under bf16 (on-the-fly
  cache quant → meaningless). `flashmla_auto` is a prefill-selection aid, not decode.
- DSA/FlashMLA on this path is effectively page_size 64; alternates may be overridden
  (still probe one for the page-size-flexibility evidence requirement).
- `mem-fraction 0.9` is not a speed knob here (capacity already ample at 0.85).

## Sweep plan (on the gate-passing spec-ON base)
Run each knob independently, keep what lowers `mean_tpot` while holding
median_itl ≤ 33.3 and p99_ttft < 22 s, then combine winners and confirm.

| id | change | hypothesis |
|----|--------|-----------|
| dp_attn | `--enable-dp-attention --dp-size 8` | biggest decode-scaling lever at conc 64 |
| spec_decmode | `--speculative-attention-mode decode` | cut EAGLE verify overhead (default=prefill) |
| eagle_light | steps2 topk1 draft3 | recover sustained mean, keep median pass |
| eagle_xlight | steps1 topk1 draft2 | likely best sustained mean among spec configs |
| lpm | `--schedule-policy lpm` | exploit 55% shared prefix → less cold-prefill TTFT pressure |
| chunk | `--chunked-prefill-size 4096` (DP-aware) | protect decode tails |
| page32 | `--page-size 32` | page-size flexibility evidence (≥2 sizes) |
| fp8kv* | `--kv-cache-dtype fp8_e4m3` (DSA→flashmla_kv) | ACCURACY-RISK rung 1: decode-kernel speed experiment only |

`*` accuracy-risk knob — introduced only after lower-risk knobs are exhausted, and
flagged as such in the report.

## Page-size finding (AC-4)
Source `python/sglang/srt/server_args.py:1918-1920`: on CUDA (non-HIP), the DSA
path unconditionally executes `self.page_size = 64` and logs "Setting page size to
64 for DeepSeek DSA." regardless of any `--page-size` value. Verified empirically:
`--page-size 32` launched + benchmarked successfully (tpot 41.21, gates pass) but the
server resolved page_size=64. FlashMLA also "only supports a page_size of 64"
(server_args.py:2852). => GLM-5.1 DSA on H200 supports exactly ONE effective page
size (64); the flag is accepted but always overridden. The winning config therefore
uses page 64 by hard architectural constraint, not by preference (DEC-3 intent met:
no penalty for non-64, but non-64 is simply unavailable on this DSA path).

## Sweep snapshot (gate-passing configs, run-to-run noise ~±1 ms on tpot)
baseline 42.16 | spec_decmode 42.15 | lpm 41.92 | page32(->64) 41.21 | chunk4096 41.19
=> chunked-prefill 4096 + lpm are the mild positives; combine for the winner candidate.

## Metric decision (user, supersedes plan DEC-1)
Client gave only "30 tok/s per user"; the "(or 1000/ITL)" was the user's own earlier
gloss, NOT a client mandate. Faithful per-user speed = sustained decode rate (TPOT),
not 1000/ITL (speculation-inflated). HEADLINE = 1000/median_TPOT (typical user);
report p99_TPOT as the worst-case/SLO-style guarantee (client specified P99 for TTFT,
so tail matters); show 1000/median_ITL only as "literal-but-burst-inflated".
TARGET measured at the MEDIAN: need median_TPOT <= 33.3 ms (30 tok/s).
Best gate-passing (combo): median_TPOT 40.31 ms -> 24.8 tok/s; gap ~5 tok/s (~17%).
Worst-case p99_TPOT ~68 ms -> ~14.6 tok/s.

## Why DP attention regressed at conc 64 (regime mismatch)
DP attention is a high-concurrency / KV-capacity-bound throughput optimization. At
conc 64 it lost on every axis: mean_tpot 48.63 vs 42.16, aggregate throughput 1057
vs 1266 tok/s, p99_itl 444 vs 307; only median_itl "improved" (12.81, burst artifact).
Mechanisms: (1) per-rank batch collapses to 64/8=8 reqs -> small-batch attention
under-utilizes the GPU; (2) DP-attn + TP-MoE (ep_size=1) forces all-gather before MoE
+ reduce-scatter after, every layer/step, with ~8 tokens/rank to amortize -> comm
dominates; (3) its KV-replication-avoidance win is worthless here (bf16 MLA KV tiny,
not capacity-bound; capacity even halved 300k->161k). => stay TP8 (also satisfies AC-6).

## IndexCache (accuracy-risk rung 2) — the only knob that moved the binding metric
combo + --json-model-override-args '{"index_topk_pattern":"FFSF...SSS"}' (GLM-5 doc pattern):
median_tpot 36.30 ms -> 27.5 tok/s (vs combo 24.8), mean_tpot 37.72, p99_tpot 63.25 -> 15.8 tok/s,
median_itl 17.36, p99_ttft 11387, accept 3.093 (unchanged), 320 ok / 0 err.
Works because it reuses the DSA indexer result ACROSS LAYERS -> cuts per-step DECODE
indexer compute (the binding cost). Contrast FP8-KV which only touched KV/attn bandwidth
(decode kernel) and regressed. ACCURACY-RISK: docs claim "negligible" loss but the
latency benchmark cannot verify quality -> MUST be flagged; report as best-achievable,
with combo (bf16) as the safe no-risk recommendation.
Rung 3 (raise SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD) NOT pursued: it affects
dense PREFILL attention (TTFT, which has ~10s slack), not the binding decode TPOT.

## Two reported winners
- SAFE (no accuracy risk): combo = baseline-spec + chunked-prefill 4096 + schedule-policy lpm.
  median_tpot 24.8 tok/s, gap to 30 ~5 tok/s (~17%).
- BEST-ACHIEVABLE (accuracy-risk, flagged): combo + IndexCache.
  median_tpot 27.5 tok/s, gap ~2.5 tok/s (~8%). p99_ttft 11.4s (both well under 22s).

## Client ground-truth TPS + FP8 fully tested (final)
Client's verbatim TPS definition: "(total latency − TTFT) / total tokens" (seconds/token);
TPS = inverse = total_tokens/(latency−ttft). Applied to run totals:
TPS = Σtokens / Σdecode_time ≈ 1000/mean_tpot. ("per user" was an in-house inference, dropped.)
Per-request mean-of-rates reads higher (over-weights fast short decodes); the literal total/total
is the honest figure. Final client-TPS: baseline 23.8 | combo 24.3 | combo+IndexCache 26.5
(best) | combo+FP8 21.7 | combo+IndexCache+FP8 22.9. Gap to 30: best ~3.5 (12%).
FP8 KV is FULLY PERMITTED (user) yet regresses in all 3 configs (forces flashmla_kv decode,
not capacity-bound) → rejected on merit, not accuracy. P99 TTFT met (~11–12s) throughout.

## Round 1: lower-risk ladder exhaustion (AC-7) + dual-metric reconciliation
Owner directives this round: PE-1 client ground-truth TPS = (latency−TTFT)/tokens ⇒
TPS=Σtok/Σdecode≈1000/mean_tpot is authoritative (1000/ITL retracted as a non-client gloss);
report it AND the plan scalar median_itl≤33.3 (dual-metric, not a revert). PE-2 page-size-64
requirement WAIVED. PE-3 FP8 fully allowed.

Lower-risk ladder on combo base (fresh server each, all 320/0err), client TPS:
  combo 24.2 | mrr80 24.4 | mrr96 24.1 | mem0.9+cg64 23.8 | eagle_xlight(s1/d2) 22.3 (accept 1.88)
  | dsa_decode=flashmla_sparse 23.9 | dsa_prefill=flashmla_auto 24.0 | indexcache 26.6 (acc-risk).
=> All lower-risk ≈24 TPS; none beats combo. mrr80/96 inert (workload caps conc at 64);
mem0.9 inert (not capacity-bound); lighter spec hurts (accept collapses); bf16 DSA backend
swaps neutral (decode pinned to fa3-class cost). Lower-risk EXHAUSTED -> only accuracy-risk
IndexCache moves the metric. AC-7 ordering satisfied with evidence.
Dual-metric verdict: plan scalar median_itl≤33.3 MET (combo & indexcache); client TPS 30 NOT met
(best 26.5, gap ~3.5). P99 TTFT met (~11-12s) throughout. 24 fresh-server runs total.
