# Ask Codex Input

## Question

You are adversarially reviewing a REVISED root-cause for an SGLang Double Sparsity (DS) accuracy regression on GLM-5.1-FP8 (8xH200 TP=8, GSM8K temp 0, completion API). Challenge it; try to falsify.

SETUP: DS = table-free channel-importance token selection for MLA decode; DSA = GLM's native learned indexer (target). DS top_k=2048. config.py HARD-LOCKS scorer_norm="off" (raw channel-dot) because the absorbed-latent identity score=max_h v_h·c_kv only holds for the raw dot; the table-free rewrite (Loop 11) deleted the materialized per-head signature (TokenLabelTable) that a cosine scorer would need. Loop 7 had earlier measured a cosine scorer lifting 16K NIAH recall 5%->40%.

NEW MEASUREMENTS (all live GSM8K; dense=5-shot/200 ~716tok<top_k; sparse=24-shot/150 ~5.6ktok>top_k):
- DSA: dense 0.975, sparse 0.953.
- production DS: dense 0.620, sparse 0.000.
- FAITHFUL raw-dot reference (exact fp32 absorbed channel-dot, TF32 disabled, current decode slot force-INCLUDED so it is H3-clean; dense reports selected==seq_len): dense 0.950, sparse 0.013.
- FAITHFUL COSINE reference (same setup but direction-normalized cosine on a MATERIALIZED per-head signature: |K_label_h[t]| = ||absorbed_w_sel[h] @ c_kv[t]||, |Q_label_h| = ||w_c⊙q_{S_h}||, normalize AFTER the mask-channel gather): dense 0.940, sparse 0.940. DS genuinely active on sparse (selected 2048<5610, dense_fallback 0, impl=reference_cosine confirmed, 0 errors).
- Earlier H3 evidence: production dense keeps 715/716 (drops current slot); forced-all/anchor-b1 (include current slot) recovers dense 0.620->0.950/0.970.

GATE (user threshold: GOOD iff naive-DS=best(raw,cosine) sparse within 5pp of DSA AND >0, AND dense within 3pp):
- dense best(0.950,0.940)=0.950 vs 0.975 -> 2.5pp (PASS)
- sparse best(0.013,0.940)=0.940 vs 0.953 -> 1.3pp, >0 (PASS)
=> GATE = GOOD.

VERDICT (flipped from a prior "sparse=confounded H0/H2"): the algorithm TRANSFERS (ceiling GOOD). The production collapse = TWO regressions:
(1) DENSE: H3 current-decode-slot exclusion (the _slot_written invalidation). Cost ~33pp.
(2) SPARSE: the scorer_norm="off" raw-dot lock introduced by the Loop-11 table-free rewrite (which dropped the Loop-7 cosine scorer). Single-variable: faithful raw-dot sparse 0.013 vs faithful cosine sparse 0.940 (only the normalization differs). Cost ~92.7pp.

ADVERSARIALLY ANSWER:
1. Is the GOOD gate valid given the user threshold and the best-of(raw,cosine) rule? Any reason 0.940 sparse is not a legitimate ceiling (e.g., is the cosine arm secretly not pruning, or is dense<3pp marginal)?
2. Is "raw-dot sparse 0.013 vs cosine sparse 0.940, only the scorer normalization differs" a CLEAN single-variable attribution that the scorer_norm="off" lock is THE sparse culprit? Or could the cosine arm differ in more than the normalization (e.g. the materialized-signature path changes which slots are scored/validity)? What single additional control would make it airtight?
3. Is the cosine norm |K_label_h[t]|=||absorbed_w_sel[h]@c_kv[t]|| mathematically the correct per-head signature norm given absorbed_w_sel are the mask-channel rows of the per-head K-noPE up-projection? Any error that would make "cosine" not actually cosine?
4. Does this flip correctly retire the prior "sparse = H0/H2 (algorithm/mask doesn't transfer)" wording? Is a no-mask ablation still required to CLOSE the loop, or is it moot now that cosine reaches ~DSA (i.e. the mask is clearly adequate for cosine)?

Output: VERDICT_SOUND: yes|partly|no, then KEY_RISKS, then MUST_DO_EXPERIMENTS (ranked), then ALTERNATIVE_EXPLANATIONS. Concise, technical.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-20_23-08-07
- Tool: codex
