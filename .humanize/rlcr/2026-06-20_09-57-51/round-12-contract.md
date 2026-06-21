# Round 12 Contract

Round 11 was ADVANCED (the reference-arm behavior fix landed). Codex found the same class of bug,
narrower: `ds_selector_behavior_for()` only branches on `selector_impl`, so the `ds_forced_all` arm —
which has `forced_all_dense_control=true` and at runtime calls `apply_forced_all_dense()` to REPLACE the
dense scored top-k with the logical sweep `[0..seq_len-1]` (`deepseek_v2.py:2631-2645`,
`absorbed_latent.py:501-527`) — still renders as plain `production · [5120] · bf16 · blocked/radix`. This
is the last behavior-surface provenance item; Codex says fix it (CPU) before the GPU close-out.

## Mainline Objective (exactly one)
**Make the `ds_forced_all` selector-behavior surface reflect the forced-all dense override**, not generic
production top-k — and guard it, so the AC-4 table is behavior-accurate for every DS arm.

## Target ACs
- **AC-4 / AC-2.1** (primary): the forced-all downstream-isolation control renders its actual selected-set
  behavior (forced `[0..seq_len-1]` for dense), guarded.
- **AC-6 / AC-8** (secondary): the AC-4 table the writeup relies on is now behavior-accurate for all arms.

## Blocking Side Issues (this CPU bug — it IS the mainline)
- **`ds_selector_behavior_for()` ignores `forced_all_dense_control`.** Branch on
  `eff["forced_all_dense_control"]` BEFORE the generic production case. For `ds_forced_all` render: path
  `forced-all dense diagnostic (production scoring then dense override)`; selector width `full live dense
  rows (seq_len<=top_k)`; score_reduce `not used for the final dense selected set`; topk `forced
  [0..seq_len-1] after production scoring`; scoring `production pre-override only`; scorer/head_agg as
  pre-override context. Add a fail-closed assertion: if `forced_all_dense_control=true`,
  `ds_selector_behavior.topk` must contain `forced` and must NOT be plain `blocked/radix`.

## Queued Side Issues (documented, OUT OF SCOPE — GPU/instrumentation close-out, next rounds)
- AC-2.1 forced-all physical-slot assertions (`forced_all_assertions.json`) + AC-4 length-cap garbage
  counters — guarded `logical_to_physical`→`transform_index_page_table_decode` adapter instrumentation +
  GPU run (shared physical-slot boundary). (Note: this is the *physical-slot* AC-2.1; the forced-all
  *behavior surface* here is the CPU prerequisite.)
- AC-3.1 captured-row materialized fp32 `K_label` selected-index equality — resident-latent capture + offline.
- AC-2.4 recall-oracle@2048 (NIAH-only) — GPU run, labeled corroboration.
- AC-4 remaining serial cells (DSA-radix serial, production DS sparse serial); selected-vs-total gaps.
- AC-8 final writeup (after the above).
- Plan-term comment cleanup; reference-mode fail-closed.

## Concrete Success Criteria
1. `ds_selector_behavior_for()` branches on `forced_all_dense_control` first; `ds_forced_all.json`
   `ds_selector_behavior` shows the forced dense override (topk contains `forced [0..seq_len-1]`, score_reduce
   `not used for the final dense selected set`), NOT `blocked/radix`/`bf16` as the used top-k/reduce.
2. `evidence_table.md` shows `ds_forced_all` as the forced-all dense override; `production_ds` and
   `ds_reduce_fp32` STILL render production top-k (`[5120] · bf16`/`fp32` · blocked/radix); reference arms
   STILL render `full · none · exact torch.topk`.
3. A fail-closed assertion rejects any `forced_all_dense_control=true` arm whose `ds_selector_behavior.topk`
   is plain `blocked/radix` (no `forced`) — verified it fires when re-broken. The existing reference-arm
   and effective-key/cuda-graph/DS-config assertions still hold.
4. Tests pass; provenance consistent. Commit; round-12-summary with BitLesson Delta + Goal Tracker Update
   Request. CPU-only this round; no selection/adapter fix; no exit by lying / editing loop state /
   cancel-rlcr-loop.
