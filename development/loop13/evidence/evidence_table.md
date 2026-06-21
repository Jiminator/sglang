# Loop 13 — Per-arm GSM8K evidence ledger (AC-1 / AC-4), generated from evidence/meta/arms/*.json

ledger generator blob f8771c7f2f9a (head@gen 29ed825fa, worktree dirty (+uncommitted evidence/generator)) · per-arm measured_git_sha in each evidence/meta/arms/*.json (baselines @180f6dd6d, R1 ref arms @fea920c06) · model GLM-5.1-FP8 · mask sha256 5c89c516… · TP=8 page64 fp8_e4m3 KV seed42 · temp0 max_tokens512 completion API
Dense = 5-shot/200 (~716 tok < top_k 2048). Sparse = 24-shot/150 (~5.6k tok > 2048). batched=64 threads.
selected/total: DS selected vs total tokens by regime (— = native DSA / no DS meta).

| Arm | dense (b) | sparse (b) | dense (serial) | sparse (serial) | DS selected/total (dense; sparse) | note |
|---|---|---|---|---|---|---|
| dsa | 0.975 | 0.973 | 0.965 | 0.947 | — | native DSA indexer (DS off) — accuracy target |
| dsa_noradix | 0.960 | 0.940 | — | — | — | DSA + radix-cache disabled — output-neutral control |
| production_ds | 0.620 | 0.000 | 0.655 | — | dense 715/716; sparse 2048/5620 | table-free DS (scorer_norm=off,head_agg=max,bf16 reduce,radix,W=5120) — the regression |
| ref_faithful | 0.950 | 0.013 | — | — | dense 714/714; sparse 2048/5610 | faithful raw-dot ceiling: exact fp32, TF32 off, current slot incl (dense selected==seq_len) |
| ref_cosine | 0.940 | 0.940 | — | — | dense 714/714; sparse 2048/5610 | faithful COSINE ceiling: materialized per-head signature, normalize after gather |
| ds_forced_all | 0.950 | — | — | — | dense 716/716 | dense forced-all [0..seq-1] control (incl current); dense-only |
| ds_anchor_b1 | 0.970 | 0.000 | — | — | — | recency anchor budget=1 (current slot only) on production top-k |
| ds_anchor_b64 | 0.960 | 0.007 | — | — | — | recency anchor budget=64 on production top-k |

Fields not instrumented this loop (listed in each arm JSON, not faked): per-example sample IDs/order; per-step length-cap garbage counters (invalid/unwritten/duplicate/out-of-range physical slots). Gate uses the measured batched DSA comparator (0.975/0.973).

Gate (AC-5, evidence/gate_ac5.md): naive-DS=best(faithful raw-dot, cosine): dense 0.950 (2.5pp), sparse 0.940 (3.3pp) -> GOOD. Verdict: dense=H3 current-slot; sparse=raw-dot scorer_norm=off lock (reference-ceiling; production-path bisection pending).
