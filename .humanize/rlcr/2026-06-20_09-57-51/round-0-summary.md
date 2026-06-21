# Round 0 Summary — Loop 13 (DS-vs-DSA accuracy diagnosis)

## Verdict (the deliverable)
**Two-part root cause, with live GSM8K evidence on GLM-5.1-FP8 (8×H200 TP=8, temp 0, completion API). No fix landed (diagnosis loop).**

1. **DENSE 0.620 → H3 (downstream-of-selection slot-validity bug): the current decode slot is excluded from its own attention.** `_select_topk_indices` invalidates the current slot in the `_slot_written` bitmap before scoring (`_slot_written[layer_id, out_cache_loc] = False`) and only restores it after the KV write, so the current token scores −∞ and is dropped from its selected set (DS keeps 715/716 in dense). Each decode token cannot attend to itself.
   - **Airtight isolation:** force-including ONLY the current slot (anchor-recency b=1) recovers dense **0.620 → 0.970** (≈ DSA 0.975). b=64 → 0.960, forced-all → 0.950. One token is the entire dense gap.
   - **Scorer/perf-opts exonerated:** the fp32-EXACT reference selector (no fp8-in-register, no bf16 reduce, no radix approx, no width bucketing) scores the SAME 0.620 dense → rules out H1. Recovery to ≈DSA with the current slot rules out H0/H2 for dense.

2. **SPARSE 0.000 → a distinct secondary failure (H0/H2 family), confounded by H3.** With real pruning (selected 2048 of ~5600), the collapse persists with the current slot included (anchor b=1/b=64 sparse 0.000/0.007) and with the fp32-exact scorer (reference sparse 0.000). The channel-importance top-2048 doesn't capture the long-context tokens DSA's learned indexer does — but this is confounded by the H3 current-slot bug on every decode step and can only be cleanly characterized after the H3 fix.

**Recommendation:** fix H3 first (force-include the current decode slot / restore `_slot_written` before the selected set is consumed — small, localized), then re-measure the sparse selection ceiling. No selection/adapter fix landed this loop.

## Evidence (per-arm GSM8K)
| Arm | Dense | Sparse |
|---|---|---|
| DSA (native) | 0.975 | 0.953 |
| DSA `--disable-radix-cache` | 0.960 | 0.940 |
| production DS | 0.620 | 0.000 |
| fp32-exact reference (raw-dot) | 0.620 | 0.000 |
| forced-all dense (incl current) | 0.950 | n/a |
| anchor-recency b=64 | 0.960 | 0.007 |
| anchor-recency b=1 (current only) | 0.970 | 0.000 |

Full writeup: `development/loop13/ROOT_CAUSE.md`. Table: `evidence/evidence_table.md`. Notes: `evidence/findings.md`. Codex review: `evidence/codex_review_h3.md`.

## Files Changed
Diagnostic code (config-gated; production behavior byte-identical when fields unset):
- `python/sglang/srt/layers/attention/double_sparsity/config.py` — `selector_impl` + `forced_all_dense_control` fields, validation, parse wiring.
- `python/sglang/srt/layers/attention/double_sparsity/absorbed_latent.py` — `dequantize_resident_latent`, `absorbed_latent_score_logical_fp8` (gather-then-dequant), `reference_rawdot_select`, `apply_forced_all_dense`.
- `python/sglang/srt/models/deepseek_v2.py` — `_reference_selector_topk` (fp32 reference at the `_select_topk_indices` seam, gated to decode-with-state 3-D cuda queries, mask indexed per-layer); forced-all override; one-time reference-active log.

Harness + evidence (`development/loop13/`):
- `serve.sh` modes `dsa_noradix`/`ds_capture`/`ref`/`ds_forced_all`/`ds_anchor`; `run_gsm8k.sh` `THREADS`+`REGIME` knobs; `analyze_captures.py`.
- `ROOT_CAUSE.md`, `evidence/evidence_table.md`, `evidence/findings.md`, `evidence/codex_review_h3.md`, `evidence/meta/{run_meta.json,ds_instruments.md}` (server logs gitignored).

## Validation
- CPU unit tests (passed): `dequantize_resident_latent` round-trip; `apply_forced_all_dense` (dense→[0..s-1], sparse untouched); `reference_rawdot_select` top-k == `torch.topk`; `absorbed_latent_score_logical_fp8` == full-pool dequant EXACTLY on finite scores; config validation (defaults / new fields / bad-enum rejection).
- Live: all arms boot from the dev clone (`_env.sh` guard passed), DS genuinely active by regime (sparse selected<total dense_fallback==0; dense selected==seq_len under forced-all). Regression reproduced (gate sound).

## Remaining Items (explicitly deferred — see goal-tracker Plan Evolution Log + Explicitly Deferred)
- AC-2.2/2.3/2.4 (TP head-agg, radix/width index-equivalence micro-tests, recall-oracle): SUPERSEDED — the fp32-exact reference exonerates the whole scorer; recall-oracle is NIAH-only. `analyze_captures.py` + `ds_capture` mode built and ready.
- AC-3.2 (served cosine): deprioritized — a different scorer can't recover while H3 corrupts the downstream feed.
- AC-7.1/7.2/7.3 (no-mask ablation, full knob sweep, per-head oracle): deferred — the BAD sparse ceiling is confounded by H3; the verdict does NOT assert a clean H0. Partial: anchor-recency b=1/b=64 sweep run for the H3/sparse separation.
- AC-6 (GOOD-branch bisection): N/A — gate is BAD; cause is H3, not a perf-opt regression.

## BitLesson Delta
Action: add
Lesson ID(s): BL-20260620-ds-current-slot-exclusion
Notes: Captured (a) the root cause — DS drops the current decode slot via the `_slot_written` invalidation so each decode token can't attend to itself (dense 0.62, sparse 0.00) — and (b) the reusable diagnostic method: a fp32-EXACT reference selector exonerates the entire scorer in one shot; forced-all / anchor-recency-budget-sweep isolates downstream-of-selection and pins it to a single token (current slot, b=1→0.970); gather-then-dequant keeps the eager reference tractable; the sparse collapse stays confounded until the H3 fix.
