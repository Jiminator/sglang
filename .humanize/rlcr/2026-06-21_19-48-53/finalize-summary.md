# Finalize Phase Summary — loop13 diagnosis

Code review passed (`codex review --base loop13-base`, no blocking findings). This finalize pass reviewed
the recently-added code for functionality-equivalent simplification.

## Note on the simplifier agent
The `code-simplifier:code-simplifier` plugin is **not installed** in this environment (not in the available
agent registry), so I performed the simplification review manually under the same hard constraints
(functionality-equivalent only; no production-code changes; no behavior/exit-code/field changes to the
fail-closed reducers).

## What was simplified
- **`development/loop13/build_ledger.py`**: removed a **dead `import glob`** (the module reads per-arm `.out`
  scores via `score_from_out` + `os.path`/`open`; `glob` was never used). One line.

That is the only change. The other in-scope diagnostic reducers
(`ac4_garbage_counters.py`, `ac3_1_materialized_k_equality.py`, `niah_recall_oracle.py`,
`ac4_selected_vs_total_probe.py`, `ac8_selfcheck.py`, `ac2_1_forced_all_assertions.py`) were scanned (AST
unused-import/name check + manual read) and are **already lean — no change**. The
`from __future__ import annotations` directives flagged by a naive unused-import check are intentional and
were kept.

## Scope I deliberately did NOT touch
- **Production code** (`python/sglang/srt/layers/attention/double_sparsity/*`, `models/deepseek_v2.py`): the
  guarded, default-off DS instrumentation was adversarially verified **byte-identical-when-off** across 22
  rounds; churning the decode hot path to "simplify" would risk the exact property the loop spent those
  rounds proving. Out of scope, untouched.
- Committed evidence (`evidence/**`), `serve.sh`, `run_gsm8k.sh`: unchanged.

This is the right call per the project doctrine (surgical changes, no churn): the diagnosis package was
already reviewed and accepted; a verified fail-closed evidence pipeline earns a high bar for any change.

## Files modified during Finalize
- `development/loop13/build_ledger.py` (−1 line: dead import).
- `development/loop13/evidence/evidence_table.md` + `evidence/meta/*` — regenerated **provenance only** (the
  generator-blob hash refreshes because build_ledger.py's content changed; no data values changed). Committed
  together so build_ledger's self-consistency check (committed per-arm blob == generator blob) stays green.

## Confirmation tests still pass
- `build_ledger.py` → provenance consistent.
- `ac8_selfcheck.py` → "AC-8 PACKAGE COMPLETE".
- Full CPU reducer suite (`ac3_1_materialized_k_equality`, `ac4_garbage_counters`, `ac2_1_forced_all_assertions`,
  `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`, `ac6_score_reduce_corrob`, `ac2_2_head_agg`,
  `ac4_sample_ids`, `verify_ac2_3 .sglang_ds_scorecap_sparse`) — all exit 0.
- `test_reference_selectors.py` — 5/5 pass (production decode path byte-identical when the diagnostic flags
  are off).

## Refactoring decisions
Conservative by design. The only safe, genuine win was the dead-import removal; everything else was left
as-is because it is already clean and/or is verified production/evidence that must not change. Committed as
`46a5a00c5`.
