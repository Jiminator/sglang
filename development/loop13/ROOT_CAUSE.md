# Loop 13 — Root cause of the DS-vs-DSA GSM8K accuracy degradation

**Diagnosis loop — verdict with live evidence. No fix is landed this loop.**

GLM-5.1-FP8, 8×H200 TP=8, page 64, fp8_e4m3 KV, seed 42, temp 0, completion API.
Dev clone `/sgl-workspace/sglang` @ git `180f6dd6d`, mask sha256 `5c89c516…`.
Dense = GSM8K 5-shot/200 (~716 tok < top_k 2048). Sparse = 24-shot/150 (~4.2–5.6k tok > 2048).

## Per-arm GSM8K evidence

| Arm | Dense | Sparse | Note |
|---|---|---|---|
| DSA (native indexer) | 0.975 | 0.953 | accuracy target |
| DSA, `--disable-radix-cache` | 0.960 | 0.940 | radix-cache disable is output-neutral |
| production DS (table-free) | **0.620** | **0.000** | the regression |
| naive-DS raw-dot, **fp32-exact reference** | **0.620** | **0.000** | exact scorer == production |
| DS forced-all dense (incl current slot) | **0.950** | n/a | dense recovers |
| DS anchor-recency b=64 (incl current+recent) | **0.960** | **0.007** | dense recovers, sparse does NOT |
| DS anchor-recency b=1 (current-token ONLY) | **0.970** | 0.000 | airtight: ONE token (current slot) recovers dense to ≈DSA; sparse unaffected |

Reference selector = performance-naive and algorithmically exact: dequantize the resident
latent to fp32, exact absorbed channel-dot, exact full-width `torch.topk` — no fp8-in-register
dequant, no bf16 cross-TP reduce, no approximate radix top-k, no selector-width bucketing.
Served `selector_impl="reference_rawdot"`. DS genuinely active on sparse (selected 2048 < total,
dense_fallback 0). The fp32 absorbed score equals a materialized fp32 K_label score by the
absorbed identity (exact algebra; CPU unit test confirms reference top-k == `torch.topk`).

## Verdict

### Primary cause — DENSE 0.620: H3 (downstream-of-selection slot-validity bug)
In the dense regime seq ≈ 716 < top_k 2048, so selection is essentially a no-op: DS keeps
**715 of 716** tokens (sparsity_rate ≈ 0.0014). The single dropped token is the **current
decode slot**: `_select_topk_indices` invalidates it in the `_slot_written` bitmap
(`_slot_written[layer_id, out_cache_loc] = False`) *before* scoring — so a reused physical KV
slot's stale latent can't be selected — and the companion restore only happens after the KV
write. Within the same decode step the current token's own slot therefore scores −∞ and is
excluded from its selected attention set, so **each decode token cannot attend to itself.**

Evidence it is THIS and not the scorer / a perf optimization / the algorithm:
- The fp32-**exact** reference scorer scores the **same** 0.620 dense — so fp8-in-register
  scoring, bf16 reduce, approximate radix top-k, and selector-width bucketing are all
  **exonerated** (rules out H1 for dense).
- Forcing the current slot back in recovers dense to **≈ DSA 0.975** — and the recovery is
  airtight to a SINGLE token: anchor-recency **b=1 (the current slot ONLY) → 0.970**, b=64 →
  0.960, forced-all → 0.950. One token (the current decode slot) is the entire dense gap, so
  the channel-importance selection itself transfers fine in dense (rules out H0/H2 for dense).

This is a **downstream-of-selection / slot-validity** bug (H3), localized to the current-slot
invalidation in `_select_topk_indices`.

### Secondary — SPARSE 0.000: an additional selection-quality failure (H0/H2), confounded by H3
At long context with **real** pruning (selected 2048 of ~5600), the collapse persists even
when the current slot is included (anchor-recency b=64 sparse 0.007) and even with the
**fp32-exact** scorer (reference sparse 0.000). So beyond the current-slot bug, the
channel-importance top-2048 does not capture the tokens GSM8K needs for long-context reasoning
that DSA's learned indexer captures (0.953). This is the H0/H2 family — but it is **confounded
by the H3 current-slot bug on every decode step** and cannot be cleanly characterized until H3
is fixed.

## Adversarial review (Codex, gpt-5.5 xhigh — `evidence/codex_review_h3.md`)
Codex rated the verdict "partly" sound: the dense diagnosis is "very strong" and ruled H1/H0/H2
out for dense, but asked for (a) a current-**only** rescue to make the current-slot claim
airtight, and (b) a sparse recency sweep to separate H3 from a coexisting H0 in sparse. Both
were run: anchor-recency b=64 (dense recovers, sparse does not) and b=1 (current-only) — see
the table. The two-part verdict above reflects Codex's refinement.

## Recommendation (follow-up loop — NOT this loop)
1. **Fix H3 (small, localized):** force-include the current decode slot in its own selection/
   attention set, or restore `_slot_written` for the current slot before the selected set is
   consumed. Expected to recover dense to ≈ DSA.
2. **Re-measure the sparse selection ceiling AFTER the H3 fix** to cleanly decide whether the
   channel-importance algorithm transfers to long context (H0) or the mask needs recalibration
   (H2). The reference selector built here is the reusable instrument for that.
3. No selection/adapter fix is landed in this diagnosis loop.

## Reusable artifacts built this loop
- `selector_impl="reference_rawdot"` — the fp32-exact reference selector (the accuracy-ceiling
  instrument).
- `forced_all_dense_control` — the dense downstream-isolation control.
- `serve.sh` modes: `dsa_noradix`, `ds_capture`, `ref`, `ds_forced_all`, `ds_anchor`.
- `run_gsm8k.sh` `THREADS`/`REGIME` knobs; `analyze_captures.py` (TP head-agg + selected-index
  equivalence over capture dumps).
