# AC-5 decision gate (Round 1 — recomputed from a valid best-of raw/cosine FAITHFUL ceiling)


> Baseline: DSA batched = 0.975/0.973 (measured, AC-1 reproduction). The plan's original-session
> sparse number was 0.953; reproduced here as 0.973. The gate uses the measured batched comparator.

naive-DS ceiling = best(faithful-raw-dot, cosine), both FAITHFUL (current decode slot
force-included → H3-clean) and leak-free (TF32 disabled), exact fp32.

| regime | faithful raw-dot | cosine | best (naive-DS) | DSA | gap | threshold | result |
|---|---|---|---|---|---|---|---|
| dense (5sh/200) | 0.950 | 0.940 | **0.950** | 0.975 | 2.5 pp | within 3 pp | PASS |
| sparse (24sh/150) | 0.013 | **0.940** | **0.940** | 0.973 | 3.3 pp | within 5 pp + >0 | PASS |

## GATE = GOOD
The accuracy ceiling is GOOD: with the cosine scorer + the current decode slot included,
naive-DS reaches ≈ DSA in BOTH regimes. The algorithm transfers; the production collapse is
caused by regressions, not by the algorithm/mask failing (rules out a clean H0/H2 ceiling).

GOOD → AC-6 (single-variable bisection). The two culprits are already isolated by these arms:
1. **Sparse culprit — the raw-dot `scorer_norm="off"` lock.** Single-variable: faithful raw-dot
   sparse 0.013 vs faithful COSINE sparse 0.940 (only the scorer normalization differs). The
   table-free rewrite (Loop 11, `01e3ff238` deletes TokenLabelTable) hard-locked `scorer_norm="off"`
   because the absorbed-latent identity only holds for the raw dot — i.e. it DROPPED the Loop-7
   cosine scorer (which Loop 7 measured lifted 16K NIAH recall 5%→40%). Cost: ~92.7 pp sparse.
2. **Dense culprit — the current decode slot exclusion (H3).** Single-variable: production DS dense
   0.620 vs current-slot-included 0.970 (anchor b=1) / 0.950 (forced-all). The `_slot_written`
   invalidation drops the current slot from its own attention. Cost: ~33 pp dense.

## Single-variable airtightness (Codex Round-1 MUST_DO #1)
The faithful raw-dot arm scores via the absorbed-identity path; the cosine arm via a materialized
per-head signature. To make NORMALIZATION the sole variable, the cosine code path was run with
the division removed (`normalize=False`, the "materialized-raw" control). PROVEN OFFLINE (CPU):
its scores equal the absorbed raw-dot scores to max |Δ| = 4.8e-6 (rel 4.4e-7) AND produce
**bit-identical top-k selection**. Identical selection ⇒ identical model output ⇒ identical GSM8K,
so the materialized-signature path does not change the raw selection. Therefore the ONLY variable
between sparse 0.013 (raw-dot) and 0.940 (cosine) is the cosine normalization — the
`scorer_norm="off"` lock is definitively the sparse culprit.

## Exact counts (Codex MUST_DO #3)
sparse (n=150): cosine 141/150 (0.940), DSA 146/150 (0.973 measured batched), faithful raw-dot 2/150 (0.013).
dense (n=200): faithful raw-dot 190/200 (0.950), cosine 188/200 (0.940), DSA 195/200 (0.975).
The cosine-vs-rawdot sparse delta (141 vs 2 of 150) is unambiguous on a single run. The dense pass
margin is thin (~5–7 examples vs the 3pp threshold) — recorded, not oversold.

## Deferred to FIX loops / next round (Codex MUST_DO #2, #4)
- Production-style cosine control (cosine scores through the production graph-safe path, same TP
  aggregation / reduce / top-k) — a deployability check for the FIX, not the ceiling diagnosis.
- Cosine without current-slot inclusion — optional clean H3-cost accounting for sparse.
