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

## Matched-config absorbed-vs-baseline gate (R16) — PASS

`run_absorbed.sh full` booted the baseline op-point (GLM-5.1-FP8 / fp16 / mem 0.7, via `_env.sh`) with
the recall oracle and ran the sweep + `absorbed_recall_summary.py` gate vs the frozen baseline. The
`recall_oracle`-gated bind fires on GLM-5.1 (`GlmMoeDsaForCausalLM` inherits `DeepseekV2AttentionMLA`):
smoke = 624/624 records carry `rec["absorbed"]`, 0 errors. The full matched-population gate
(18,720 records, 6,240/length, all carry `rec["absorbed"]`) **PASSES ±0.5pp** (`absorbed_recall_summary.json`):

| length | absorbed recall@2048 | baseline | Δ (pp) |
|--------|----------------------|----------|--------|
| 1024   | 100.000% | 100.0   | +0.000 |
| 4096   | 58.397%  | 58.045  | +0.352 |
| 16384  | 35.946%  | 36.042  | −0.096 |
| overall| 64.781%  | 64.696  | +0.085 |

Comparability OK (length set + per-length sample counts match the baseline exactly); `problems[]`
empty; no hard failures; table-path sanity overall 64.706% (also within tolerance). **task5's served
recall gate is closed**: the fp8-latent absorbed scorer reproduces the frozen recall baseline on the
served GLM-5.1-FP8 op-point. (The R15 DSv3.2+int8 numbers above remain documented as the confounded run
that motivated the matched-config re-run.)
