# Round 8 Contract

Round 7 was ADVANCED. Codex now drives toward close-out. Two findings: (1) a real bug I introduced —
when `ds_reduce_fp32` switched to graph mode, `build_ledger.py` still hard-codes `--disable-cuda-graph`,
so the arm JSON contradicts the actual graph-enabled run (makes the single-variable arm look
multi-variable); (2) original-plan close-out items (AC-2.1, AC-2.2, AC-2.4, AC-3.1, AC-4 sample/garbage,
AC-8) remain. This round closes the two OFFLINE-computable items and fixes the metadata bug; the
GPU/instrumentation-heavy items are queued with an explicit per-item approach.

## Mainline Objective (exactly one)
**Advance original-plan close-out by settling the two offline-computable evidence items:** the
**AC-2.2** TP head-aggregation micro-test (numbers + a defensible conclusion) and **AC-4** per-arm GSM8K
**sample IDs/order**, on top of a corrected, self-consistent AC-4 ledger.

## Target ACs
- **AC-2.2** (primary): the head-aggregation micro-test stated with numbers + conclusion.
- **AC-4** (secondary): per-arm sample IDs/order persisted; ledger metadata correct + fail-closed.

## Blocking Side Issues (truly block AC-4/AC-6 evidence integrity)
- **`ds_reduce_fp32` ledger metadata is wrong.** `build_ledger.py:111` hard-codes
  `extra="--disable-radix-cache --disable-cuda-graph --enable-double-sparsity"`, but `serve.sh
  ds_reduce_fp32` runs in GRAPH mode (no `--disable-cuda-graph`) and the server log confirms
  `disable_cuda_graph=False`, decode `cuda graph: True`. The generated arm JSON says `cuda_graph: "off"`
  + `--disable-cuda-graph` — contradicting the run and making the single-variable arm look
  multi-variable. Fix `extra` to match serve.sh, regenerate, and add a **fail-closed consistency check**:
  for `ds_reduce_fp32`, `server_args` must NOT contain `--disable-cuda-graph`, `cuda_graph` must be
  graph-enabled, and the config must contain `"score_reduce_dtype": "fp32"`.

## Queued Side Issues (documented, OUT OF SCOPE this round — explicit approach each)
- **AC-2.1 forced-all physical-slot assertions** — needs guarded instrumentation of the
  `logical_to_physical`→`transform_index_page_table_decode` adapter to dump physical slots + `req_to_token`
  during a `ds_forced_all` run, then assert (no dup/`-1`/unwritten/out-of-range, adapter errors 0). GPU +
  new capture. Next round.
- **AC-2.4 recall-oracle@2048** — the `recall_oracle` instrument is NIAH-only (DEC); run the NIAH
  dense/sparse oracle, label as corroboration. GPU run. Next round.
- **AC-3.1 captured-row materialized fp32 `K_label`** — needs the resident latent captured alongside
  scores, then offline materialize `K_label` and compare top-2048 selection vs `absorbed_latent_score_logical`.
  GPU capture + offline compute. Next round.
- **AC-4 length-cap garbage counters** — invalid/unwritten/duplicate/out-of-range physical-slot counts
  per layer/step; same adapter instrumentation as AC-2.1. Next round.
- **AC-8 final writeup** — after AC-2.1/2.2/2.4/3.1/4 land.
- Plan-term comment cleanup; reference-mode fail-closed.

## Approach (AC-2.2, offline from validated captures)
The captured per-rank `pre_reduce_scores` are validated (`sum(pre)==post` 702/702). Compute, per
8-rank group, the served cross-TP SUM vs global-MAX vs global-MEAN selected-index sets:
- served SUM (= `reduce_token_scores`, the production reduction) vs global-MAX(per-rank scores) vs MEAN.
Report Jaccard + identical counts. Preliminary numbers: SUM-vs-MAX median Jaccard 0.67 (0/624 identical);
SUM==MEAN (scale-only). Conclusion (the AC-2.2 micro-test statement): the served head_agg="max" +
cross-TP **SUM** is **NOT** equivalent to a global max over all heads. **Exoneration as the accuracy
bottleneck:** `build_absorbed_projection` uses `num_local_heads` and the reference path does NO cross-TP
reduce (verified — no `reduce_token_scores`/all-reduce in `_reference_selector_topk`), so production
(cross-TP SUM) and the reference (per-rank-local) use DIFFERENT head aggregation, yet the GOOD ceiling
holds under cosine on BOTH (cosine recovers; raw-dot collapses under both: production-SUM sparse 0.000 ≈
reference-local sparse 0.013). So the cross-TP aggregation is not the accuracy driver — consistent with
AC-6 (scorer + current-slot are the culprits). Write fail-closed `evidence/head_agg_tp_semantics.json`.

## Approach (AC-4 sample IDs/order, offline)
Re-derive the exact GSM8K example IDs/order each arm used from the stock `run_eval` gsm8k loader
(fixed seed-42 slice) for the dense (5-shot/200) and sparse (24-shot/150) configs, and persist them so
every arm row is reproducible. Wire into the ledger; do not fabricate.

## Concrete Success Criteria
1. `build_ledger.py` `ds_reduce_fp32` extra == `serve.sh` (no `--disable-cuda-graph`); regenerated arm
   JSON/table/run_meta show `cuda_graph` graph-enabled + `--disable-radix-cache --enable-double-sparsity`.
   A fail-closed check asserts the ds_reduce_fp32 args/cuda_graph/config invariants (verified it fires).
2. `evidence/head_agg_tp_semantics.json`: served SUM vs global-MAX vs MEAN selected-set Jaccard +
   identical counts on the validated 8-rank groups; the AC-2.2 conclusion stated with the exoneration
   reasoning + the verifying facts (`num_local_heads`, no reference cross-TP reduce, cosine recovery).
   Fail-closed on zero groups / `sum(pre)!=post`.
3. `evidence/gsm8k_sample_ids.json` (or per-arm field): the dense + sparse seq of example IDs/order,
   re-derived from the stock loader; referenced from the ledger.
4. ROOT_CAUSE.md / findings.md note AC-2.2 settled (cross-TP SUM exonerated) and the AC-4 additions.
   Tests pass; provenance consistent. Commit; round-8-summary with BitLesson Delta + Goal Tracker Update
   Request. No selection/adapter fix; if any GPU run is needed it is one TP=8 at a time (this round is
   CPU-only). No exit by lying / editing loop state / cancel-rlcr-loop.
