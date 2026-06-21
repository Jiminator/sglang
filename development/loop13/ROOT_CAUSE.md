# Loop 13 — Root cause of the DS-vs-DSA GSM8K accuracy degradation (AC-8)

**Diagnosis loop — final verdict with live, fail-closed evidence. NO selection/adapter fix is landed this loop.**

GLM-5.1-FP8, 8×H200 TP=8, page 64, fp8_e4m3 KV, seed 42, temp 0, completion API.
Dev clone `/sgl-workspace/sglang`, mask sha256 `5c89c516…`.
Dense = GSM8K 5-shot/200 (~716 tok < top_k 2048). Sparse = 24-shot/150 (~4.2–5.6k tok > 2048).
This writeup is generated from the final committed evidence package (`evidence/evidence_table.md`,
`evidence/meta/run_meta.json`, and the per-AC artifacts cited inline); `ac8_selfcheck.py` fail-closes if any
required artifact, serial cell, or citation is missing.

## Headline
**The accuracy ceiling is GOOD: with the cosine scorer and the current decode slot included, naive-DS reaches
≈ DSA in both regimes (dense 0.950, sparse 0.940 vs measured DSA 0.975/0.973).** The channel-importance
algorithm DOES transfer to GLM-5.1 MLA, and the offline mask is adequate. The production DS collapse
(dense 0.620, sparse 0.000) is **TWO regressions** layered in during the table-free optimization history —
NOT the algorithm failing to transfer (not H0) and NOT a bad mask (not H2):

1. **Dense 0.620 → H3: the current decode slot is excluded from its own selection** — the
   `_slot_written[layer_id, out_cache_loc]=False` invalidation in `_select_topk_indices` is not restored
   before the selected set is consumed. Including the current slot recovers dense **0.620 → 0.950/0.970**.
2. **Sparse 0.000 → the raw-dot `scorer_norm="off"` lock.** The Loop-11 table-free rewrite (`01e3ff238`
   deletes `TokenLabelTable`) hard-locked `scorer_norm="off"` (raw channel-dot) — the absorbed-latent
   identity only holds for the raw dot — i.e. it **dropped the Loop-7 cosine scorer**. Raw-dot collapses
   long-context selection (faithful raw-dot sparse **0.013**); the **cosine** scorer (re-materialized on a
   per-head signature) recovers sparse to **0.940 ≈ DSA**, **but only together with the H3 fix** (the two
   regressions interact — see the AC-6 2×2). Sparse needs both.

## Per-arm GSM8K evidence (AC-4; batched + serial)
Full machine-readable table: `evidence/evidence_table.md` (+ per-arm `evidence/meta/arms/*.json`). Serial =
THREADS=1, batched = 64 threads; both on the validated configs. Serial ≈ batched on every arm → the
regression is **not** batch-dependent.

| Arm | dense (b) | sparse (b) | dense (serial) | sparse (serial) | DS selected/total (dense; sparse) | Note |
|---|---|---|---|---|---|---|
| DSA (native indexer) | 0.975 | 0.973 | 0.965 | 0.947 | — | accuracy target (DS off) |
| DSA, `--disable-radix-cache` | 0.960 | 0.940 | 0.965 | 0.973 | — | radix-cache disable is output-neutral |
| **production DS** (table-free) | **0.620** | **0.000** | 0.655 | 0.013 | 334/334; **2048/3692** | the regression |
| **FAITHFUL raw-dot** (current incl, TF32 off) | **0.950** | **0.013** | 0.965 | 0.013 | 334/334; **2048/3692** | H3-clean ceiling; raw-dot collapses sparse |
| **FAITHFUL cosine** (current incl, TF32 off) | **0.940** | **0.940** | 0.965 | 0.947 | 334/334; **2048/3692** | **cosine recovers sparse 0.013→0.940 ≈ DSA** |
| cosine, current EXCLUDED (`ref_cosine_noinc`) | 0.625 | 0.313 | — | — | — | AC-6 single-variable arm (current-slot only) |
| production raw-dot, fp32 reduce (`ds_reduce_fp32`) | 0.620 | 0.000 | — | — | — | AC-6 leg 7 (reduce dtype) — = production |

`ds_selected_vs_total` is **artifact-backed** (`evidence/ac4_selected_vs_total.json`, R21 — probed from the
live server's `meta_info["double_sparsity"]`): for production_ds / ref_faithful / ref_cosine the selector
keeps **all** tokens in dense (334/334) and prunes to **2048 of 3692** in sparse, with `dense_fallback==0` —
DS is genuinely active. The reference selectors are performance-naive and exact (fp32 dequant of the resident
latent, exact absorbed channel-dot / cosine, exact full-width `torch.topk`; no fp8-in-register dequant, bf16
reduce, radix approximation, or selector-width bucketing). "Faithful" = current decode slot force-included
(dense `selected == seq_len`) + TF32 disabled (leak-free fp32).

## AC-5 decision gate (`evidence/gate_ac5.md`)
naive-DS = best(faithful raw-dot, cosine): dense best(0.950, 0.940)=**0.950** vs DSA 0.975 → 2.5 pp (within
3 pp); sparse best(0.013, **0.940**)=**0.940** vs DSA 0.973 → 3.3 pp (within 5 pp, > 0). **GATE = GOOD** →
AC-6 (single-variable bisection). AC-7 (BAD-branch no-mask/knob sweep) is **moot** — not the taken branch.

## AC-6 bisection — the scorer × current-slot 2×2 (measured)
The two culprits are **not independent**. `ref_cosine_noinc` flips exactly one variable vs faithful cosine —
`reference_include_current` true→false (the production current-slot exclusion), everything else held — giving
the full 2×2 (each cell dense / sparse):

| scorer \ current-slot | EXCLUDED (production) | INCLUDED (faithful) |
|---|---|---|
| **raw-dot** | production 0.620 / **0.000** | ref_faithful 0.950 / **0.013** |
| **cosine** | ref_cosine_noinc 0.625 / **0.313** | ref_cosine 0.940 / **0.940** |

1. **Sparse recovery to ≈0.94 requires BOTH fixes.** Cosine alone (production current-slot exclusion) reaches
   only 0.313; current-slot inclusion alone (raw-dot) reaches only 0.013 (corroborated by the `ds_anchor`
   arms — forcing recent slots back on the raw-dot path stays 0.000/0.007). The two regressions interact.
2. **Current-slot exclusion (H3) is a culprit in BOTH regimes** (under cosine it costs dense 0.940→0.625 AND
   sparse 0.940→0.313), corroborated in both regimes (`ac6_ref_cosine_noinc_corrob.json`: sparse 4992/4992
   single-swap + dense 3744/3744 add).
3. **Scorer (raw-dot → cosine)** is the other variable. Responsible change: Loop-11 rewrite `01e3ff238`
   (`config.py` hard-locks `scorer_norm="off"`).

Per-leg matrix `evidence/ac6_bisection_matrix.json`: head_agg = not-a-differing-variable (max on both;
cross-TP is AC-2.2); scorer + current-slot = **measured** (2×2); radix + width = **retired** (AC-2.3,
4992/4992); bf16-vs-fp32 reduce = **measured** (`ds_reduce_fp32` = production, median Jaccard 0.998); only
fp8-absorbed is **blocked** (no production config toggles absorbed precision — `absorbed_latent_kernel.py`
scores fp8 in-register; exact-fp32 absorbed lives only on the multi-variable reference path; a production
fp32-absorbed path = new code = a fix) and bounded second-order (raw-dot exact-fp32 0.013 vs fp8 0.000 ⇒
≤~1.3 pp).

## Corroboration (the cheap controls + faithfulness proofs)
- **AC-2.1 — H3 measured directly on the validity bitmap.** `evidence/forced_all_assertions.json`: the
  forced-all dense control's post-adapter physical slots == `req_to_token[req,0:seq_len]` on **61776/61776**
  rows across 20+ decode steps with **0** duplicate/`-1`/out-of-range/adapter-error and **0 non-current
  unwritten** — so the `logical_to_physical` adapter + selected-index path is a provable clean no-op; the
  ONLY unwritten live slot on every row is the **current decode slot** (the `_slot_written` invalidation =
  H3, observed on the bitmap, not inferred).
- **AC-4 length-cap garbage counters — clean adapter path on every served arm.**
  `evidence/ac4_garbage_counters.json` (production_ds) + `..._ref_faithful.json` + `..._ref_cosine.json`:
  real (non-current) garbage **0** in both regimes on all three (41808 dense + 37440 sparse rows each). The
  only difference is current-slot membership — production EXCLUDES it (`current_slot_unwritten=0`), the
  faithful references INCLUDE it (`=rows`) — pinning H3 from both the forced-include and scored-exclude
  sides.
- **AC-2.4 — recall-oracle@2048 (NIAH, corroboration only).** `evidence/ac2_4_recall_oracle.json`: dense
  recall@2048 = **1.0** (the selector keeps every token → the dense gap is NOT scorer-ranking, it is H3);
  sparse recall@2048 = **0.4103** (needle_worst_rank median 2524 > 2048) — the production raw-dot scorer
  ranks the needle inside the 2048 budget only ~41% of the time, corroborating the **scorer-driven** sparse
  collapse.
- **AC-3.1 — the raw-dot ceiling is the materialized-`K_label` ceiling, on captured rows.**
  `evidence/ac3_1_materialized_k_selected_index_equality.json`: on **96 dense + 96 sparse** real captured
  decode rows the absorbed raw-dot score and the materialized fp32 `K_label` score select the **identical**
  top-2048 (max abs score diff ~1e-9 = fp32 round-off) — so the "faithful raw-dot" ceiling is trustworthy
  (its number is not an absorbed-vs-materialized artifact). (Supersedes the synthetic
  `ac3_1_materialized_k.json`.)
- **AC-2.2 — head aggregation.** `evidence/head_agg_tp_semantics.json`: within-rank `head_agg="max"` matches
  on both paths; the cross-TP SUM-vs-reference-local difference is a measured second-order ≤1.3 pp, not the
  bottleneck.

## Why NOT H0 / H2 (and why the no-mask ablation is moot)
- NOT **H0** (algorithm doesn't transfer): cosine reaches ≈DSA in both regimes, so the channel-importance
  algorithm transfers — the raw-dot scorer, not the algorithm, was the failure.
- NOT **H2** (bad mask): the SAME offline channel mask reaches ≈DSA under cosine → the mask is adequate. The
  BAD-branch no-mask ablation (AC-7.1) is the response to a BAD gate; the gate is GOOD, so AC-7 is not on the
  taken branch.

## Verdict (ranked)
**H1 (a perf-optimization regression), as a ranked pair, with an H3 mechanism on the dense side:**
1. **Dense:** H3 — current decode slot excluded from its own selection (`_slot_written` not restored).
   Measured on the bitmap (AC-2.1) and from both garbage-counter sides (AC-4); GSM8K cost 0.620 → 0.950/0.970.
2. **Sparse:** the raw-dot `scorer_norm="off"` lock (Loop-11 `01e3ff238` dropped the Loop-7 cosine scorer),
   **interacting** with H3 — sparse recovery to ≈0.94 needs BOTH the cosine scorer AND current-slot inclusion
   (AC-6 2×2); recall-oracle (AC-2.4) confirms the raw-dot ranking is scorer-limited in long context.
Not H0, not H2 (cosine + the same mask reach ≈DSA). All deltas are far above the GSM8K single-run
significance bar (n=150 binomial stderr ≈ 4 pp).

## Recommendation (follow-up FIX loops — NOT this diagnosis loop)
1. **Fix H3 (small, localized):** force-include the current decode slot in its own selection/attention set,
   or restore `_slot_written` for the current slot before the selected set is consumed.
2. **Restore the cosine scorer for the table-free path:** re-materialize the per-head signature (or compute
   the cosine norms from the absorbed projection, `|K_label_h| = ||w_sel[h] @ c_kv||`) so
   `scorer_norm="cosine"` is available again — this recovers the sparse regime.
**No selection/adapter fix is landed in this diagnosis loop** — this is a verdict + recommendation only.

## Adversarial review
Round-0 Codex review (`evidence/codex_review_h3.md`) confirmed the dense H3 diagnosis and demanded the cosine
arm + a faithful, leak-free ceiling — exactly what overturned the Round-0 "sparse = confounded H0/H2"
wording. Round-1 Codex review of the GOOD gate: `evidence/codex_review_gate.md`. Every per-AC artifact above
is fail-closed at production time (the producer writes only on success) AND re-validated by `build_ledger.py`
before it is rendered into `run_meta.json`; `ac8_selfcheck.py` refuses this writeup if any is absent.

## Reusable artifacts built this loop
- Faithful, leak-free reference selectors: `selector_impl ∈ {reference_rawdot, reference_cosine}` +
  `reference_include_current` + TF32-off (the accuracy-ceiling instruments).
- Guarded, default-off, config-borne diagnostic captures: `forced_all_assert` (AC-2.1 + AC-4 garbage),
  `recall_oracle` (AC-2.4 NIAH), `materialized_k_capture` (AC-3.1 captured-row equality), plus the
  selected-vs-total probe.
- `serve.sh` modes: `dsa_noradix`, `ds_capture`, `ds_reduce_fp32`, `ref`/`ref_faithful`/`ref_cosine`/
  `ref_cosine_noinc`, `ds_forced_all`/`ds_forced_all_assert`, `ds_garbage`, `ref_faithful_garbage`/
  `ref_cosine_garbage`, `ds_recall_oracle`, `ref_faithful_matk`, `ds_anchor`; `run_gsm8k.sh` THREADS/REGIME.
- `build_ledger.py` (provenance-consistent ledger with per-artifact fail-closed gates) + the per-AC reducers
  (`ac2_1_*`, `ac2_2_*`, `verify_ac2_3.py`, `ac4_garbage_counters.py`, `niah_recall_oracle.py`,
  `ac3_1_materialized_k_equality.py`, `ac4_selected_vs_total_probe.py`, `ac6_*`) + `ac8_selfcheck.py`.
