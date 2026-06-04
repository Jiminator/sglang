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
