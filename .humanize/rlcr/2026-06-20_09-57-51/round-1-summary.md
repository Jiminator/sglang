# Round 1 Summary — Loop 13 (DS-vs-DSA accuracy diagnosis)

## Headline: the verdict FLIPPED — the ceiling is GOOD
Round-0 left a confounded "sparse = H0/H2" wording. Codex's Round-0 review correctly demanded a
served cosine arm + a faithful, leak-free ceiling. Building those overturned the conclusion:

| arm (faithful: current slot incl, TF32 off, exact fp32) | dense | sparse |
|---|---|---|
| raw-dot | 0.950 | 0.013 |
| **cosine** | 0.940 | **0.940** |
| DSA | 0.975 | 0.953 |

**The cosine scorer recovers sparse 0.013 → 0.940 ≈ DSA.** AC-5 gate recomputed from
best(raw,cosine): dense 0.950 (2.5 pp), sparse 0.940 (1.3 pp) → **GOOD**. The channel-importance
algorithm DOES transfer to GLM-5.1 MLA. The production collapse is **two regressions**:
1. **Dense (0.620) = H3**: the current decode slot is excluded from its own attention (`_slot_written`). Including it → 0.950/0.970. Cost ~33 pp.
2. **Sparse (0.000) = the `scorer_norm="off"` raw-dot lock**: the Loop-11 table-free rewrite (`01e3ff238`, deletes `TokenLabelTable`) dropped the Loop-7 cosine scorer. Single-variable: raw-dot sparse 0.013 vs cosine 0.940. Cost ~92.7 pp.

NOT H0 (cosine transfers) and NOT H2 (same mask reaches ≈DSA under cosine) → the BAD-branch
no-mask ablation (AC-7) is moot; the GOOD branch (AC-6 bisection) is the taken branch and is answered.

## Mainline objective (round-1-contract.md): DONE
Make the reference a valid AC-5 gate input (cosine + faithful + leak-free) and recompute the gate.

## Work Completed (diagnostic code; production unchanged when flags unset)
- `reference_cosine` selector: cosine on a materialized per-head signature
  (`|K_label_h|=||absorbed_w_sel[h]@c_kv||`, `|Q_label_h|=||w_c⊙q_{S_h}||`, normalize after gather).
  A `normalize=False` mode gives the materialized-raw single-variable control. (AC-3.2)
- `reference_include_current` config flag: force-include the current decode slot → H3-clean ceiling
  (dense reports `selected==seq_len`). (AC-3.3)
- TF32 disabled (`allow_tf32=False` + cuDNN) in the reference path → leak-free fp32. (AC-3.4)
- `serve.sh` modes `ref_faithful` and `ref_cosine`; per-arm metadata JSON under `evidence/meta/arms/`.

ACs addressed: **AC-3.2** (cosine served, dense 0.940/sparse 0.940, DS active by regime),
**AC-3.3** (faithful dense `selected==714==seq_len`; sparse `selected 2048<5610`, `dense_fallback 0`),
**AC-3.4** (TF32 disabled), **AC-5** (gate GOOD, valid best-of), **AC-6** (two culprits single-variable,
costs + responsible change; cosine-vs-rawdot sparse delta 141 vs 2 of 150 is unambiguous).

## Files Changed
- Code: `config.py`, `absorbed_latent.py`, `deepseek_v2.py` (cosine, include_current, TF32-off, normalize control).
- Harness/tests: `serve.sh` (ref_faithful/ref_cosine), `development/loop13/test_reference_selectors.py`.
- Evidence/writeup: `ROOT_CAUSE.md` (rewritten), `evidence/{gate_ac5.md, evidence_table.md, findings.md, codex_review_gate.md, meta/arms/*.json}`.
Two atomic commits (`fea920c06`, `62ad64346`); tree clean; one TP=8 server at a time; GPUs idle.

## Validation
- CPU: `python3 development/loop13/test_reference_selectors.py` → 5/5 pass, including the decisive
  `test_materialized_raw_equals_absorbed_raw` (cosine path `normalize=False` is selection-equal to the
  absorbed raw-dot, max |Δ| 4.8e-6, bit-identical top-k) — proving NORMALIZATION is the sole
  cosine-vs-rawdot variable (Codex Round-1 MUST_DO #1, addressed offline → conclusive).
- Live: ref_faithful + ref_cosine boot from the dev clone (guard passed), DS active by regime,
  0 selector errors; cosine sparse 0.940 verified with selected 2048<5610, dense_fallback 0,
  impl=reference_cosine confirmed.
- Codex gate verification: `evidence/codex_review_gate.md` — GOOD gate arithmetically valid, cosine
  norm formula correct, no-mask ablation no longer required.

## Remaining Items (queued / next round — not verdict-changing)
- AC-2 capture artifacts (`ds_capture`, `cheap_controls.json`, forced-all physical-slot-assertion JSON) + the full per-arm AC-4 ledger with length-cap garbage-rate columns.
- Production-style cosine deployability control (Codex MUST_DO #2) — a FIX-loop concern.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260620-ds-rawdot-scorer-lock
- Notes: Captured that the DS sparse collapse is the `scorer_norm="off"` raw-dot lock (the table-free
  rewrite dropped the Loop-7 cosine scorer), recoverable by a cosine scorer whose per-head signature
  norms are computable from the bind-time absorbed projection (no TokenLabelTable rebuild needed); and
  the diagnostic method (toggleable raw-dot/cosine reference + materialized-raw selection-equality
  proof) that turns "does the algorithm transfer?" into a one-variable GSM8K experiment and makes the
  no-mask ablation moot once best-of(raw,cosine) is GOOD.
