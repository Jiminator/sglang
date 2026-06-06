# Loop 2 coverage ledger (no silent skips)

Every axis is either measured (gate + profile) or closed with a cited launch/runtime outcome.

## Infeasible / non-launchable axes (closed with citation)

| axis / knob | attempt | outcome (taxonomy) | citation |
|---|---|---|---|
| EAGLE tree `--speculative-eagle-topk 2` (draft8/steps3) | launched on combo base | **startup-reject** (~15s, before weight load) | `ValueError` at `python/sglang/srt/arg_groups/speculative_hook.py:388` (`_handle_eagle_family`): "speculative_eagle_topk > 1 with page_size > 1 ... only supported for the 'flashinfer' backend." DSA forces `page_size=64` + `attention_backend=dsa`. Server log also notes the v2→v1 fallback ("spec v2 topk > 1 currently requires page_size == 1") preceding the hard reject. |

EAGLE tree (topk>1) is therefore **infeasible flags-only on this DSA path** — it cannot be benchmarked without an out-of-scope attention-backend swap. The incumbent topk=1 verify/draft cost is instead characterized from the `combo_baseline` profile (see `profiling/combo_baseline.md`): EAGLE adds ~2× MoE launches; verify/draft is a major, non-negligible share of decode cost at conc 64 (consistent with loop 1's "lighter EAGLE regresses").

## DSA prefill × decode cross-product (bf16) — taxonomy

Base = combo (EAGLE steps3/topk1/draft4, mem0.85, mrr64, chunked-prefill 4096, lpm) + `--dsa-prefill-backend P --dsa-decode-backend D`.
Expected (source): `decode=flashmla_auto` → first-request runtime failure (no `auto` branch in decode dispatch, `dsa_backend.py:1726`); `flashmla_kv` under bf16 → launchable, quantizes whole cache (`dsa_backend.py:1846-1848`), expected slow. Owner decision DEC-3: **no pruning** — every launchable cell fully gate-benchmarked AND profiled.

| prefill \ decode | flashmla_sparse | flashmla_kv | flashmla_auto | fa3 |
|---|---|---|---|---|
| flashmla_sparse | pending | pending | expect runtime-fail | **done (combo_baseline = 24.08 TPS)** |
| flashmla_kv | pending | pending | expect runtime-fail | pending |
| flashmla_auto | pending | pending | expect runtime-fail | pending |
| fa3 | pending | pending | expect runtime-fail | pending |

Results are appended to `sweep_table.md` (gate) and per-cell profiles to `profiling/<tag>.md`. This ledger is updated as cells complete.
