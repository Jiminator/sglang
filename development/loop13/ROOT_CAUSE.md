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
> Scope note (updated Round 6): the AC-6 per-leg bisection matrix is complete
> (`evidence/ac6_bisection_matrix.json`). scorer + current-slot are **measured** (the 2×2 below; the
> current-slot leg corroborated on 4992 captured rows, `ac6_ref_cosine_noinc_corrob.json`) — sparse
> ≈0.94 needs BOTH; current-slot exclusion (H3) hurts BOTH regimes. radix + width are **retired**
> (AC-2.3, 4992/4992). head_agg is **not a reference→production difference** (max on both paths;
> AC-2.2 covers cross-TP). fp8-absorbed + bf16-reduce are **blocked** with a specific code citation
> (the production absorbed_latent_kernel.py is raw-dot-only; config.py:110/170 reject cosine; a cosine
> production kernel = a fix) and bounded second-order (raw-dot exact-fp32 0.013 vs fp8+bf16 0.000).

1. **Dense 0.620 → H3: the current decode slot is excluded from its own attention** (the
   `_slot_written` invalidation in `_select_topk_indices` is not restored before the selected set
   is consumed). Including the current slot recovers dense **0.620 → 0.950/0.970**.
2. **Sparse 0.000 → the raw-dot `scorer_norm="off"` lock.** The table-free rewrite (Loop 11,
   `01e3ff238` deletes `TokenLabelTable`) hard-locked `scorer_norm="off"` (raw channel-dot) because
   the absorbed-latent identity only holds for the raw dot — i.e. it **dropped the Loop-7 cosine
   scorer** (Loop 7 measured cosine lifting 16K NIAH recall 5%→40%). The raw-dot scorer collapses
   long-context selection (faithful raw-dot sparse **0.013**); the **cosine** scorer, re-materialized
   here on a per-head signature, recovers sparse to **0.940 ≈ DSA** — **but only together with the H3
   fix below** (R5 single-variable bisection: cosine with the production current-slot exclusion reaches
   only sparse **0.313**; the two regressions interact — see the AC-6 2×2). Sparse needs both fixes.

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
| **cosine, current EXCLUDED** (`ref_cosine_noinc`, R5) | **0.625** | **0.313** | AC-6 single-variable arm: ONLY current-slot flipped vs faithful cosine → both regimes drop; sparse needs BOTH fixes |

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

## AC-6 bisection — the scorer × current-slot 2×2 (measured, Round 5)
The two culprits are **not independent**. Round 5 closed the single-variable bisection across the
two reference→production variables that have clean config toggles (scorer normalization; current-slot
inclusion). The new arm `ref_cosine_noinc` flips **exactly one** variable vs the faithful cosine
ceiling — `reference_include_current` true→false (the production current-slot exclusion) — everything
else (cosine scorer, `head_agg=max`, exact fp32, TF32-off) held fixed. Result: dense **0.940→0.625**
(= production dense 0.620) and sparse **0.940→0.313**. Combined with the existing arms this gives the
full 2×2:

| scorer \ current-slot | EXCLUDED (production) | INCLUDED (faithful) |
|---|---|---|
| **raw-dot** | production 0.620 / **0.000** | ref_faithful 0.950 / **0.013** |
| **cosine** | ref_cosine_noinc 0.625 / **0.313** | ref_cosine 0.940 / **0.940** |

(each cell dense / sparse). Reading the sparse column:
1. **Sparse recovery to ≈0.94 requires BOTH fixes.** Cosine alone with the production current-slot
   exclusion reaches only **0.313**; current-slot inclusion alone under raw-dot reaches only **0.013**
   (corroborated by the `ds_anchor` arms — forcing the recent slots back on the raw-dot path stays
   0.000/0.007). Only cosine **and** current-slot inclusion together reach 0.940. The two regressions
   **interact**; neither is individually sufficient for sparse.
2. **Current-slot exclusion (H3) is a culprit in BOTH regimes**, not dense-only as the
   reference-ceiling framing implied. Under cosine it costs dense 0.940→0.625 **and** sparse
   0.940→0.313. The faithful cosine 0.940 sparse ceiling benefited from the (non-production)
   current-slot inclusion; the production-path cosine ceiling (current excluded) is **0.313** sparse.
3. **Scorer (raw-dot → cosine)** is the other variable: holding current-slot included it is worth
   sparse 0.013→0.940; holding it excluded, 0.000→0.313. Responsible change: Loop 11 table-free
   rewrite (`01e3ff238`, deletes `TokenLabelTable`; `config.py` hard-locks `scorer_norm="off"`).
   Current-slot mechanism: `_slot_written[layer_id, out_cache_loc]=False` before scoring, not restored
   before the selected set is consumed.

All deltas are far above the GSM8K single-run significance bar (n=150 binomial stderr ≈ 4 pp).

**Per-leg AC-6 bisection matrix** (`evidence/ac6_bisection_matrix.json`, generated) — every
reference→production variable is measured, retired, not-a-difference, or carries an explicit per-leg
blocker (no blanket "out of scope"):

| leg | variable | verdict | evidence |
|---|---|---|---|
| 1 | head_agg (within-rank) | not-a-differing-variable | production AND reference both use `head_agg="max"`; the cross-TP sum-of-max question is AC-2.2 (separate) |
| 2 | scorer (raw-dot ↔ cosine) | **measured** | 2×2 + `test_reference_selectors.py` (materialized-raw == absorbed-raw selection) |
| 3 | current-slot (incl ↔ excl) | **measured** | `ref_cosine_noinc` 0.625/0.313 + `ac6_ref_cosine_noinc_corrob.json` (4992/4992 single-swap) |
| 4 | radix top-k (exact ↔ blocked) | retired | `ac2_3_radix_width_equivalence.json` 4992/4992 |
| 5 | selector width ([5120] ↔ full) | retired | `ac2_3_radix_width_equivalence.json` 4992/4992 |
| 6 | fp8-absorbed (fp32 ↔ fp8) | **blocked** | code path below + second-order bound |
| 7 | bf16-reduce (fp32 ↔ bf16) | **blocked** | code path below + second-order bound |

Legs 6–7 **blocker (specific, not blanket):** the fp8-absorbed and bf16-reduce variables live ONLY in
the production absorbed-latent Triton scoring kernel
(`absorbed_latent_kernel.py`, called from `deepseek_v2.py:_select_topk_indices` ~2588/2602), which
implements **only** `scorer_norm="off"` (raw channel-dot); `config.py:110` `_ALLOWED_SCORER_NORM=("off",)`
and the validation at `config.py:170` hard-reject `scorer_norm="cosine"`. The reference cosine path
computes exact fp32 and does not route through that kernel, so there is **no config toggle** to test
fp8/reduce under cosine — doing so needs a new production-path cosine kernel = a selection-path code
change = a **fix**, forbidden this loop. They are bounded **second-order**: on the raw-dot path, where
exact-fp32 (`ref_faithful`) and fp8+bf16 (production) can be compared, sparse 0.013 vs 0.000 ⇒ fp8/reduce
contribute ≤~1.3 pp beyond the scorer/current-slot effects.

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
  `ref_cosine_noinc` (the AC-6 current-slot single-variable arm), `ds_forced_all`, `ds_anchor`.
  `run_gsm8k.sh` `THREADS`/`REGIME` knobs; `analyze_captures.py`.
- `verify_ac2_3.py` (pruning-valid radix/width equivalence, fail-closed on zero pruning rows),
  `ac6_corrob_ref_cosine_noinc.py` (current-slot selection-swap corroboration on captured rows),
  `ac6_bisection_matrix.py` (the per-leg AC-6 matrix generator).
- Per-arm metadata JSON under `evidence/meta/arms/` (with AC-6 leg/corroboration + measured-source
  provenance); `build_ledger.py` asserts provenance consistency + fails closed on an uncorroborated
  AC-6 arm.
