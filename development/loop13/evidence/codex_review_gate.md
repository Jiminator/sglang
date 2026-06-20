VERDICT_SOUND: partly

KEY_RISKS

- GOOD gate: arithmetically valid under the stated best-of(raw, cosine) rule. Sparse `0.940` is a legitimate active-pruning ceiling if `selected=2048<5610`, `dense_fallback=0`, `impl=reference_cosine`, and no errors are confirmed.
- Dense pass is real but thin: `0.950` vs `0.975` on 200 items is only a 5-example gap, with a 3pp threshold allowing about 6 examples. It passes, but do not oversell statistical strength.
- “Algorithm transfers” is sound for the DS mask plus cosine-scored selector family. It is not sound for the current table-free raw-dot production implementation.
- Sparse attribution is strong but not airtight: raw-dot uses absorbed-identity scoring, cosine uses a materialized per-head signature path. Algebra says the raw dots are equivalent, but implementation path equivalence still needs to be proven.
- Production collapse still includes implementation deltas absent from the references: graph-safe path, bf16 reduce, radix/top-k path, selector width, and possibly TP aggregation semantics. The reference result is a ceiling, not a production patch validation.

MUST_DO_EXPERIMENTS

1. Materialized-raw control: run the exact cosine code path but select on `dots` before dividing by norms. Same `K_label`, `Q_label`, same slot mask, same include-current, same top-k. It should match absorbed raw-dot scores/selections and stay near `0.013` sparse. This makes normalization the only variable.
2. Production-style cosine control: same candidate validity, same TP aggregation/reduce semantics, same top-k path, but cosine scores. If this stays near `0.940`, the `scorer_norm="off"` lock is definitively the sparse production blocker.
3. Repeat with exact counts and at least one rerun/larger sample. Sparse likely maps to roughly `141/150` vs DSA `143/150`; good, but small deltas need count-level reporting.
4. Optional: cosine without current-slot force-include to quantify H3 interaction on sparse. Not needed for the scorer claim, but useful for clean cost accounting.

ALTERNATIVE_EXPLANATIONS

- The cosine arm may be benefiting from the materialized-signature implementation path, not only normalization, until materialized-raw fails identically.
- A per-rank/local-head reference selection could differ from production’s synchronized selection semantics; verify TP aggregation before treating it as deployable.
- H3 current-slot exclusion may still contribute to sparse production `0.000`, though H3-clean raw-dot at `0.013` shows it is not the main sparse culprit.
- The cosine norm formula is mathematically right: `K_label_h[t] = absorbed_w_sel[h] @ c_kv[t]`, `Q_label_h = w_c * q_{S_h}`, norm after gather. Minor caveat: use clamped norms rather than product-plus-eps if near-zero norms exist.
- No-mask ablation is no longer required to retire “mask/algorithm does not transfer” for GSM8K. Cosine with the existing mask already reaches DSA. No-mask is now only a headroom/generalization study.
