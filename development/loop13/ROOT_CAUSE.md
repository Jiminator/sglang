# Loop 13 — Root cause of the DS-vs-DSA GSM8K accuracy degradation

**Diagnosis loop — verdict with live evidence. No fix is landed this loop.**

GLM-5.1-FP8, 8×H200 TP=8, page 64, fp8_e4m3 KV, seed 42, temp 0, completion API.
Dev clone `/sgl-workspace/sglang` @ git `180f6dd6d`, mask sha256 `5c89c516…`.
Dense = GSM8K 5-shot/200 (~716 tok < top_k 2048). Sparse = 24-shot/150 (~4.2–5.6k tok > 2048).

## Headline (Round 1)
**The accuracy ceiling is GOOD: with the cosine scorer and the current decode slot included,
naive-DS reaches ≈ DSA in both regimes (dense 0.950, sparse 0.940 vs measured DSA 0.975/0.973).**
The channel-importance algorithm DOES transfer to GLM-5.1 MLA. The production DS collapse is **two
regressions** layered in during the table-free optimization history — NOT the algorithm failing
to transfer (not H0) and NOT a bad mask (not H2: the same mask reaches ≈DSA under cosine):

> Baseline note: DSA batched here measures 0.975/0.973 (the plan's original-session sparse number
> was 0.953; reproduced as 0.973). The gate uses the consistent **measured batched** comparator.
> Scope note: the sparse attribution below is established at the **reference-ceiling** level
> (cosine vs raw-dot, with the materialized-raw selection-equality proof). The full
> **production-path** one-variable bisection (head_agg / fp8-absorbed / bf16-reduce / radix-topk /
> selector-width arms; production-style cosine) is **pending** (next round) — the remaining
> production opts are shown second-order (production raw-dot 0.000 ≈ exact raw-dot 0.013).

1. **Dense 0.620 → H3: the current decode slot is excluded from its own attention** (the
   `_slot_written` invalidation in `_select_topk_indices` is not restored before the selected set
   is consumed). Including the current slot recovers dense **0.620 → 0.950/0.970**.
2. **Sparse 0.000 → the raw-dot `scorer_norm="off"` lock.** The table-free rewrite (Loop 11,
   `01e3ff238` deletes `TokenLabelTable`) hard-locked `scorer_norm="off"` (raw channel-dot) because
   the absorbed-latent identity only holds for the raw dot — i.e. it **dropped the Loop-7 cosine
   scorer** (Loop 7 measured cosine lifting 16K NIAH recall 5%→40%). The raw-dot scorer collapses
   long-context selection (faithful raw-dot sparse **0.013**); the **cosine** scorer, re-materialized
   here on a per-head signature, recovers sparse to **0.940 ≈ DSA**.

## Per-arm GSM8K evidence

| Arm | Dense | Sparse | Note |
|---|---|---|---|
| DSA (native indexer) | 0.975 | 0.973 | accuracy target (measured batched) |
| DSA, `--disable-radix-cache` | 0.960 | 0.940 | radix-cache disable is output-neutral |
| production DS (table-free) | **0.620** | **0.000** | the regression |
| reference raw-dot, H3-CONTAMINATED (drops current slot) | 0.620 | 0.000 | scorer-isolation control: exact fp32 scorer == production under the same slot-validity bug → exonerates fp8/bf16/radix/width opts |
| forced-all dense (incl current) | 0.950 | n/a | H3 dense fix |
| anchor-recency b=1 (current slot ONLY) | 0.970 | 0.000 | airtight H3: ONE token recovers dense |
| **FAITHFUL raw-dot** (current incl, TF32 off) | **0.950** | **0.013** | H3-clean ceiling; raw-dot collapses sparse |
| **FAITHFUL cosine** (current incl, TF32 off) | **0.940** | **0.940** | **cosine recovers sparse 0.013→0.940 ≈ DSA**; DS active (2048<5610, no fallback) |

The reference selectors are performance-naive and exact (fp32 dequant of the resident latent,
exact absorbed channel-dot / cosine, exact full-width `torch.topk`; no fp8-in-register dequant,
bf16 reduce, radix approximation, or selector-width bucketing). "Faithful" = current decode slot
force-included (H3-clean: dense reports `selected == seq_len`) + TF32 disabled (leak-free fp32).
The cosine arm materializes the per-head signature: `|K_label_h[t]| = ||absorbed_w_sel[h] @ c_kv[t]||`
(`absorbed_w_sel[h]` are the mask-channel rows of the per-head K-noPE up-projection), `|Q_label_h| =
||w_c ⊙ q_{S_h}||`, normalize AFTER the mask-channel gather (the Loop-7 lever).

## AC-5 decision gate (recomputed — see `evidence/gate_ac5.md`)
naive-DS = best(faithful raw-dot, cosine): dense best(0.950, 0.940)=0.950 vs DSA 0.975 → 2.5 pp
(within 3 pp); sparse best(0.013, **0.940**)=0.940 vs DSA 0.973 → 3.3 pp (within 5 pp, > 0).
**GATE = GOOD** → AC-6 (single-variable bisection).

## AC-6 bisection — reference-ceiling result (production-path arms pending, next round)
The two culprits are named from the reference-ceiling single-variable controls below; the full
production-path one-variable bisection (production-style cosine; head_agg / fp8-absorbed / bf16-reduce
/ radix-topk / selector-width arms, each corroborated) is **pending**. Until then the sparse culprit
is the strong, single-variable-supported **candidate**, not a closed production-path attribution.
1. **Sparse: the raw-dot `scorer_norm="off"` lock.** Single-variable control: faithful raw-dot
   sparse 0.013 vs faithful **cosine** sparse 0.940 (identical setup; ONLY the scorer normalization
   differs). Cost ≈ **92.7 pp** sparse. Responsible change: Loop 11 table-free rewrite
   (`01e3ff238`, deletes `TokenLabelTable`; `config.py` hard-locks `scorer_norm="off"`).
2. **Dense: the current decode slot exclusion (H3).** Single-variable: production DS dense 0.620 vs
   current-slot-included 0.950 (forced-all) / 0.970 (anchor b=1). Cost ≈ **33 pp** dense.
   Mechanism: `_slot_written[layer_id, out_cache_loc] = False` before scoring, not restored before
   the selected set is consumed.

Both deltas are far above the GSM8K single-run significance bar (n=150 binomial stderr ≈ 4 pp), so
they are unambiguous on a single run (per the significance convention).

## Why NOT H0 / H2 (and why the no-mask ablation is moot)
- NOT H0 (algorithm doesn't transfer): cosine reaches ≈DSA in both regimes, so the
  channel-importance algorithm transfers — the raw-dot scorer, not the algorithm, was the failure.
- NOT H2 (bad mask): the SAME offline channel mask reaches ≈DSA under cosine, so the mask is
  adequate. The BAD-branch no-mask ablation (AC-7.1) is the response to a BAD gate; the gate is
  GOOD, so AC-7 is not on the taken branch.

## Adversarial review
Round-0 Codex review (`evidence/codex_review_h3.md`) confirmed the dense H3 diagnosis and demanded
the cosine arm + a faithful, leak-free ceiling — exactly what overturned the Round-0 "sparse =
confounded H0/H2" wording. Round-1 Codex review of the GOOD gate: `evidence/codex_review_gate.md`.

## Recommendation (follow-up FIX loops — NOT this diagnosis loop)
1. **Fix H3 (small, localized):** force-include the current decode slot in its own selection/
   attention set, or restore `_slot_written` for the current slot before the selected set is used.
2. **Restore the cosine scorer for the table-free path:** re-materialize the per-head signature (or
   compute the cosine norms from the absorbed projection as done here, `|K_label_h|=||w_sel[h]@c_kv||`)
   so `scorer_norm="cosine"` is available again — this recovers the sparse regime.
No selection/adapter fix is landed in this diagnosis loop.

## Reusable artifacts built this loop
- `selector_impl ∈ {reference_rawdot, reference_cosine}` + `reference_include_current` + TF32-off —
  the faithful, leak-free reference selectors (the accuracy-ceiling instruments).
- `forced_all_dense_control` — the dense downstream-isolation control.
- `serve.sh` modes: `dsa_noradix`, `ds_capture`, `ref`, `ref_faithful`, `ref_cosine`,
  `ds_forced_all`, `ds_anchor`. `run_gsm8k.sh` `THREADS`/`REGIME` knobs; `analyze_captures.py`.
- Per-arm metadata JSON under `evidence/meta/arms/`.
