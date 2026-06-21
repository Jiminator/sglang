# Loop 13 — Root-cause findings (live, in progress)

GLM-5.1-FP8, 8×H200 TP=8, page 64, fp8_e4m3 KV, seed 42, temp 0, completion API.
Git sha 180f6dd6d, mask sha256 5c89c516… . See `evidence_table.md` for the full table.

## AC-1 — pinned baselines (reproduced)
| arm | dense (5sh/200) | sparse (24sh/150) |
|---|---|---|
| DSA (native) batched | 0.975 | 0.973 |
| DSA serial | 0.965 | 0.947 |
| DSA-radix-off batched | 0.960 | 0.940 |
| production DS batched | **0.620** | **0.000** |
| production DS serial (dense) | 0.655 | — |

- Regression reproduced. `--disable-radix-cache` is output-neutral (DSA-radix-off ≈ DSA), so the radix *cache* is not the cause.

## AC-3 — fp32 raw-dot reference selector (accuracy ceiling)
The reference selector is performance-naive and algorithmically exact: it dequantizes the
resident latent to fp32 and scores the exact absorbed channel-dot, then takes an exact
full-width `torch.topk` — **no** fp8-in-register dequant, **no** bf16 cross-TP reduce,
**no** approximate radix top-k, **no** selector-width bucketing. Served via `selector_impl="reference_rawdot"`,
eager. DS genuinely active on the sparse regime (selected 2048 < total, dense_fallback 0).

| arm | dense | sparse |
|---|---|---|
| naive-DS raw-dot (fp32 exact) batched | **0.620** | **0.000** |

### Headline: the exact scorer does NOT recover dense.
naive-DS raw-dot dense (0.620) == production DS dense (0.620) AND naive-DS raw-dot sparse (0.000) == production DS sparse (0.000), vs DSA 0.975/0.953.
The exact scorer recovers NEITHER regime.

This **exonerates the scorer and every selection-side perf optimization** as the cause of
the *dense* degradation, because:
1. In the dense regime seq≈716 < top_k 2048, so selection is (essentially) a no-op — DS keeps
   **715 of 716** tokens (`sparsity_rate≈0.0014`; it drops ~1 token, almost certainly the
   current decode slot invalidated by `_slot_written` before scoring).
2. The fp32-exact scorer keeps the same ~all tokens and still scores 0.620 — identical to
   the fp8/bf16/radix production scorer.

→ Since DS keeps (almost) all live tokens in dense yet scores 0.620 vs DSA's 0.975 on the
**same tokens through the same `flash_mla_with_kvcache` decode kernel**, the dense degradation
is **downstream of selection (H3)** — the `logical_to_physical` → `transform_index_page_table_decode`
index adapter, KV-slot validity, or the kernel feed — NOT the scorer (H1 ruled out for dense)
and NOT a pure algorithm-doesn't-transfer story for the dense regime.

### Decisive next control (AC-2.1)
Forced-all dense control (`forced_all_dense_control=true`): force logical `[0..seq_len-1]`
(all tokens incl. the current slot), bypassing scoring+validity.
- If forced-all dense **recovers toward 0.975** → the dropped current-token / selection-domain
  validity is the bug.
- If forced-all dense **stays ≈0.620** with the adapter producing the natural dense page table →
  the bug is the adapter/slot-validity/kernel-feed path even when fed all tokens.

(Status: pending — runs next after the reference sparse arm completes.)

## AC-2.1 — Forced-all dense control: H3 LOCALIZED
Force logical [0..seq_len-1] (all tokens incl. the current decode slot) for seq<=top_k,
via `forced_all_dense_control=true`. The dense request now reports selected==total
(716/716, vs production 715/716).

| arm | dense |
|---|---|
| production DS | 0.620 |
| DS forced-all (incl current slot) | **0.950** |
| DSA (native) | 0.975 |

**Forcing the current decode slot back into the selected set recovers dense GSM8K from
0.620 to 0.950 (≈ DSA).** 

### Root cause (H3, slot-validity)
The DS selection path drops exactly one token in dense — the current decode slot. The
`_slot_written[layer_id, out_cache_loc] = False` invalidation (deepseek_v2._select_topk_indices)
marks the current slot invalid BEFORE scoring so a reused slot's stale latent can't be
selected; the companion restore happens only after the KV write, so for THIS decode step
the current token's own slot scores -inf and is excluded from its selected attention set.
Each decode token therefore cannot attend to itself, which degrades generation
(dense 0.62 vs 0.975) and — at long context, compounded with real pruning — collapses it
(sparse 0.000). Force-including the current slot recovers dense to ≈ DSA.

This is a DOWNSTREAM-of-selection / slot-validity bug (H3), NOT:
- a scorer / perf-optimization bug (H1): the fp32-exact reference scorer also scores 0.620/0.000;
- an algorithm-transfer failure (H0) or bad mask (H2): with the current slot included, the
  exact channel-dot selection reaches ≈ DSA in dense.

The fix (force-include the current/recent slot, or restore _slot_written before selection)
is a FOLLOW-UP loop — this diagnosis loop lands no fix.

## AC-2.2 (refinement) — recency-anchor sweep: dense vs sparse diverge
Codex adversarial review (evidence/codex_review_h3.md) flagged that forced-all bypasses
validity for the whole dense row and that sparse (real pruning) may have a coexisting H0.
Tested `anchor_mode=recency` (force-include recent slots incl. current, within the 2048 budget):

| arm | dense | sparse |
|---|---|---|
| production DS | 0.620 | 0.000 |
| anchor recency b=64 | **0.960** | **0.007** |
| DSA | 0.975 | 0.953 |

→ Including the current/recent slots **recovers dense (0.960) but NOT sparse (0.007)**.
So the verdict is TWO-PART:
1. **Dense 0.620 = H3 (current decode slot excluded from its own attention).** Fully recoverable
   by including the current slot (forced-all 0.950, anchor-recency 0.960).
2. **Sparse 0.000 = an ADDITIONAL failure beyond the current-slot bug.** With real pruning
   (selected 2048 of ~5600) the long-context selection still collapses even with the current
   slot included and even with the fp32-EXACT scorer (reference sparse 0.000). This points to
   the channel-importance top-2048 not capturing the tokens needed for long-context reasoning
   (H0 / H2 family), confounded by the H3 current-slot bug on every decode step. The sparse
   selection ceiling can only be cleanly measured after the H3 fix.

## AC-2.1 (airtight) — recency-anchor sweep b=1 vs b=64
| anchor budget | dense | sparse |
|---|---|---|
| b=1 (current slot ONLY) | **0.970** | 0.000 |
| b=64 (current + 63 recent) | 0.960 | 0.007 |
| DSA | 0.975 | 0.953 |

Including ONLY the single current decode slot recovers dense to 0.970 (~DSA) — the dense gap
is exactly one token. Neither b=1 nor b=64 recovers sparse -> the sparse collapse is a distinct
secondary failure (long-context selection quality), not the current-slot bug. Airtight per
Codex's requested current-only isolation + sparse sweep.

## ROUND 1 — verdict FLIPPED: ceiling is GOOD; sparse = the raw-dot scorer_norm=off lock
Built faithful (current-slot-included, H3-clean) + leak-free (TF32-off) reference selectors,
and implemented the served COSINE reference (materialized per-head signature, normalize after
mask-channel gather). Codex Round-0 demanded these and they overturn the Round-0 sparse wording.

| arm (faithful, TF32 off) | dense | sparse |
|---|---|---|
| raw-dot (current incl) | 0.950 | 0.013 |
| **cosine (current incl)** | 0.940 | **0.940** |
| DSA | 0.975 | 0.953 |

- **cosine RECOVERS sparse 0.013 → 0.940 ≈ DSA** (DS active: selected 2048<5610, no fallback).
- AC-5 gate recomputed best(raw,cosine): dense 0.950 (2.5pp), sparse 0.940 (1.3pp) → **GOOD**.
- → GOOD branch (AC-6 bisection). Two culprits, single-variable:
  1. sparse = raw-dot scorer_norm="off" lock (Loop-11 table-free dropped the Loop-7 cosine scorer);
     raw-dot 0.013 vs cosine 0.940, only normalization differs. Cost ~92.7pp.
  2. dense = H3 current-slot exclusion. Cost ~33pp.
- NOT H0 (cosine transfers) and NOT H2 (same mask works under cosine) → the no-mask ablation (AC-7,
  a BAD-branch action) is moot on the GOOD branch.

## ROUND 5 — AC-6 bisection refines R1: the two culprits INTERACT (scorer × current-slot 2×2)
The R1 "sparse = just the scorer lock" was incomplete. R5 ran one clean single-variable arm
(`ref_cosine_noinc`: cosine with `reference_include_current=false`, the ONLY change vs faithful cosine)
→ dense **0.940→0.625**, sparse **0.940→0.313**. Full 2×2 (dense / sparse):

| scorer \ current-slot | EXCLUDED (production) | INCLUDED (faithful) |
|---|---|---|
| raw-dot | production 0.620 / 0.000 | ref_faithful 0.950 / 0.013 |
| cosine | ref_cosine_noinc 0.625 / 0.313 | ref_cosine 0.940 / 0.940 |

- Sparse ≈0.94 needs **BOTH** cosine scorer AND current-slot inclusion; neither alone (cosine+excl
  0.313, rawdot+incl 0.013). Corroborated by `ds_anchor` (current-slot forced back on raw-dot → still
  0.000/0.007). The two regressions interact.
- Current-slot exclusion (H3) is a culprit in **BOTH** regimes (cosine: dense 0.940→0.625, sparse
  0.940→0.313), not dense-only. The faithful cosine 0.940 sparse ceiling used the non-production
  current-slot inclusion; the production-path cosine ceiling (current excluded) is **0.313** sparse.
- Full per-leg AC-6 matrix in `evidence/ac6_bisection_matrix.json` (R6, completed R7): scorer +
  current-slot MEASURED; radix + width RETIRED (AC-2.3, 4992/4992); bf16-vs-fp32 score-reduce MEASURED
  (R7) via the runnable `score_reduce_dtype="fp32"` route — `ds_reduce_fp32` = production 0.620/0.000,
  near-selection-neutral (median Jaccard 0.998, `ac6_score_reduce_fp32_corrob.json`), NOT a culprit;
  head_agg NOT-a-differing-variable (max on both paths; AC-2.2 covers cross-TP); only fp8-absorbed
  BLOCKED (no production config toggles absorbed precision; exact-fp32 absorbed only on the
  multi-variable reference path) + bounded second-order (raw-dot exact-fp32 0.013 vs fp8 0.000 ≤~1.3pp).
  The current-slot leg is corroborated in BOTH regimes (sparse 4992/4992 swap + dense 3744/3744 add):
  `evidence/ac6_ref_cosine_noinc_corrob.json`
  (4992/4992 — the include flag swaps exactly the current decode slot; it is -inf-masked in every capture).

## AC-2.3 RESOLVED (Round 5, pruning-valid) — radix == torch.topk, width [5120] == full
On **4992** real captured sparse GLM-5.1 score rows (median seq_len **4280**, 2048 of ~4280 pruned),
`verify_ac2_3.py` (`evidence/ac2_3_radix_width_equivalence.json`) shows the production blocked/radix
algorithm == exact `select_topk_sequence_order` (**4992/4992**) and selector-width [5120] == full
(**4992/4992**). The verifier fails (exit 2) if `pruning_rows==0`, so it cannot pass on smoke captures.
The radix and selector-width AC-2.3 suspects are RETIRED on the actual sparse workload. (Superseding
the Round-4 over-claim below, which ran on seq_len=13 smoke rows.)

### Why the Round-4 AC-2.3 claim was invalid (history; superseded by the RESOLVED section above)
The Round-4 method is right (run BOTH the production blocked/radix algorithm and exact
select_topk_sequence_order on the SAME captured score row; no score-vs-selection alignment needed),
but its captures were INVALID for the sparse regime: all 624 committed score rows are seq_len=13 while
top_k=2048, so both methods select every position — pruning is never exercised and the width-[5120]
check is vacuous (w=min(5120,13)=13). A select-all sanity check cannot retire the radix or
selector-width suspects for the sparse DS workload (selected≈2048 of ~5.6k). The Round-3 81/546 was a
separate score-vs-selection step-misalignment artifact. R5 fixed this by capturing the SPARSE regime
(median seq_len 4280) and hardening verify_ac2_3.py to record the seq_len distribution + a pruning_rows
count and FAIL if pruning_rows==0 — so it cannot pass on smoke captures again.
