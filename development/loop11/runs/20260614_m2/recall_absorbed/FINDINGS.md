# task5 served NIAH recall@2048 — side-by-side absorbed-latent diagnostic (R15)

The absorbed-latent scorer is wired SIDE-BY-SIDE into the serving recall-oracle path
(`retrieve_topk_graph_safe` → `_maybe_record_recall_oracle`), gated by `recall_oracle`; each
decode-step oracle record now carries `payload["absorbed"]` next to the table-path payload. The
selection decode consumes is unchanged (table-driven) — this is score-only.

## Live validation (DeepSeek-V3.2, TP=8, fp8 KV, eager, recall_oracle)

`serve_double_sparsity.sh` (signature_dtype=int8) + `loop7/niah_oracle_sweep.py`
(lengths 1024/4096/16384 × 20 trials = 14,640 samples):

- **244/244 smoke records carried `payload["absorbed"]`, 0 failures** — the wiring fires on the live
  graph-safe path.
- **Absorbed↔table selection agreement: 99.88% (14,623/14,640)**; absorbed recall@2048 vs the same
  run's table recall: **Δ ~0.02pp** (4096: 45.10 vs 45.08; 1024/16384: identical). The fp8-latent
  absorbed scorer reproduces the table's selection on real NIAH traffic — the declared value-affecting
  fp8-latent change is within ±0.5pp of the table it replaces (AC-5 value-affecting gate evidence).
  See `absorbed_recall_summary.json`.

## Why the absorbed-vs-FROZEN-baseline numbers in the summary are CONFOUNDED (do not read as a fail)

The frozen baseline `development/loop9/runs/20260610_m0/recall_baseline.json` (overall 64.696%) is the
**GLM-5.1 + signature_dtype=fp16 + mem 0.7** op-point (`development/profiling/runs/20260609/_env.sh`:
`glm51-fp8-channel-mask`, `"signature_dtype": "fp16"`). This validation run used
**DeepSeek-V3.2 + int8 + mem 0.8** — a DIFFERENT model AND dtype. Both the table (55.7%) and the
absorbed (55.7%) sit ~13pp below the baseline at long contexts, so the gap is the model+dtype
mismatch (int8 vs fp16 signature precision), **NOT** the absorbed scorer (which tracks the table at
99.88%). The absorbed-vs-baseline deltas in `absorbed_recall_summary.json` are therefore invalid for
the literal AC-5 gate.

## Remaining: matched-config absorbed-vs-baseline run (R16)

`run_absorbed.sh full` boots the baseline op-point (GLM-5.1 / fp16 / mem 0.7, via `_env.sh`) with the
recall oracle, runs the sweep, and post-processes both the table and `rec["absorbed"]` recall via
`absorbed_recall_summary.py`. That run produces the apples-to-apples absorbed-vs-frozen-baseline
number (expected PASS: absorbed≈table≈baseline given the 99.88% within-run agreement). Prerequisite to
confirm: GLM-5.1 routes attention through `DeepseekV2AttentionMLA` so the recall_oracle-gated bind in
`deepseek_v2.py` fires (it does for DSv3.2 — 244/244 records here).
