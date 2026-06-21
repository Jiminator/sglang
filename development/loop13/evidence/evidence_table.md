# Loop 13 — Per-arm GSM8K evidence ledger (AC-1 / AC-4), generated from evidence/meta/arms/*.json

ledger generator blob 837069691534 (head@gen 393966c02, worktree dirty (+uncommitted evidence/generator)) · per-arm measured_git_sha in each evidence/meta/arms/*.json (baselines @180f6dd6d, R1 ref arms @fea920c06) · model GLM-5.1-FP8 · mask sha256 5c89c516… · TP=8 page64 fp8_e4m3 KV seed42 · temp0 max_tokens512 completion API
Dense = 5-shot/200 (~716 tok < top_k 2048). Sparse = 24-shot/150 (~5.6k tok > 2048). batched=64 threads.
selected/total: DS selected vs total tokens by regime (— = native DSA / no DS meta).

| Arm | dense (b) | sparse (b) | dense (serial) | sparse (serial) | DS selected/total (dense; sparse) | note |
|---|---|---|---|---|---|---|
| dsa | 0.975 | 0.973 | 0.965 | 0.947 | — | native DSA indexer (DS off) — accuracy target |
| dsa_noradix | 0.960 | 0.940 | — | — | — | DSA + radix-cache disabled — output-neutral control |
| production_ds | 0.620 | 0.000 | 0.655 | — | dense 715/716; sparse 2048/5620 | table-free DS (scorer_norm=off,head_agg=max,bf16 reduce,radix,W=5120) — the regression |
| ref_faithful | 0.950 | 0.013 | — | — | dense 714/714; sparse 2048/5610 | faithful raw-dot ceiling: exact fp32, TF32 off, current slot incl (dense selected==seq_len) |
| ref_cosine | 0.940 | 0.940 | — | — | dense 714/714; sparse 2048/5610 | faithful COSINE ceiling: materialized per-head signature, normalize after gather |
| ref_cosine_noinc | 0.625 | 0.313 | — | — | — | AC-6 single-variable bisection arm (R5): cosine with reference_include_current=FALSE — the ONE variable flipped vs ref_cosine (production current-slot exclusion). head_agg=max, exact-fp32, TF32-off all unchanged. reference_cosine_select code unchanged since R1 fea920c06; serve.sh ref_cosine_noinc mode added R5. RESULT: dense 0.940->0.625 (=production 0.620) AND sparse 0.940->0.313 -> current-slot exclusion (H3) is a major culprit in BOTH regimes, not dense-only. Sparse needs BOTH cosine scorer AND current-slot inclusion (see 2x2 in ROOT_CAUSE). |
| ds_forced_all | 0.950 | — | — | — | dense 716/716 | dense forced-all [0..seq-1] control (incl current); dense-only |
| ds_anchor_b1 | 0.970 | 0.000 | — | — | — | recency anchor budget=1 (current slot only) on production top-k |
| ds_anchor_b64 | 0.960 | 0.007 | — | — | — | recency anchor budget=64 on production top-k |

Fields not instrumented this loop (listed in each arm JSON, not faked): per-example sample IDs/order; per-step length-cap garbage counters (invalid/unwritten/duplicate/out-of-range physical slots). Gate uses the measured batched DSA comparator (0.975/0.973).

Gate (AC-5, evidence/gate_ac5.md): naive-DS=best(faithful raw-dot, cosine): dense 0.950 (2.5pp), sparse 0.940 (3.3pp) -> GOOD. Verdict (AC-6 bisection, R5): the scorer x current-slot 2x2 is measured — sparse 0.94 needs BOTH the cosine scorer AND current-slot inclusion (cosine+excl=0.313, rawdot+incl=0.013, rawdot+excl=production 0.000). Current-slot exclusion (H3) is a culprit in BOTH regimes (cosine dense 0.940->0.625, sparse 0.940->0.313 when flipped). Radix+width retired on real sparse rows (AC-2.3). Untested numeric legs (fp8/bf16-reduce/head_agg) need a production-path cosine kernel = code change, out of scope (no fix).
