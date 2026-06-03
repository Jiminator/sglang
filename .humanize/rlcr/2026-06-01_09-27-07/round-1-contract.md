# Round 1 Contract

## Mainline Objective
Land the **production-shaped AC-3 Tier-2.B selector** — primarily a **length-conditional hybrid scorer** (raw channel-dot for context ≤ threshold, cosine for longer), config-gated and **default-off byte-identical** — and **measure** it at 4K + 16K to show it recovers 4K's ~75% recall *while keeping* 16K's ~40%. Add the remaining independent AC-3 scorer flags (head-aggregation, anchor-budget) as config-gated variants with CPU + TP-determinism tests, and fix the two scorer-related Blocking Side Issues so the result is trustworthy.

## Target ACs (1–2)
- **AC-3** (primary): non-learned, flag-gated selector uplift with non-regression (within-budget parity, default byte-identical, TP cross-rank determinism).
- **AC-6** (premise): no Tier-1 regression (default path unchanged; new scorer is opt-in).

## Blocking Side Issues In Scope (must fix this round)
- **B1 — graph-capture silently raw-scores a non-off scorer.** `cuda_graph.py::capture_decode_step` uses the graph-safe raw scorer whenever scratch exists, ignoring `selector.config.scorer_norm`. Fix: **fail fast** when a non-`off` scorer would enter the raw graph-safe path (the graph-safe Triton scorer-norm port is queued; until then, a non-off scorer must run eager, not silently raw under capture).
- **B2 — serve op-point default mismatch.** `serve_double_sparsity.sh` defaults `fp16`/`mem 0.6`, not the Loop-7 `int8`/`mem 0.7`. Fix: add a fail-fast `LOOP7_MEASUREMENT=1` mode that pins `int8` + `mem 0.7` and logs the effective `double_sparsity_config`, so measurement runs cannot silently use the old regime.

## Queued Side Issues Out Of Scope (justified)
- **Oracle fail-closed + 64K oracle re-run** (Codex gap #1 / blocking #2): this is M0/AC-1 *diagnostic* hardening, a separate workstream; it does **not** block the hybrid measurement (which uses served NIAH recall, not the oracle). High-priority next round.
- **Full measurement matrix** (task12: DSA same-node, MMLU re-anchor, dense-DS, N≥50 for binding 16K, every-variant, all-length): measure-first — prove the hybrid recovers 4K this round; the binding matrix is the next round once the landed scorer is chosen.
- **Graph-safe Triton scorer-norm port** (production perf landing): the hybrid runs on the eager fallback (DEC-6 research path) for measurement; B1's fail-fast is the interim safety. Port is queued.
- **Tier-2.A / AC-4** (task13–17), **M4 consolidation / AC-6 perf** (task19–20), **plan-marker code cleanup**: separate milestones; queued.

## Round Success Criteria
- `scorer_norm` config field extended to `{off, cosine, hybrid}` (+ a hybrid token-length threshold field); independent `head_agg` and `anchor_budget` config fields added; all reject bad values; **default off ⇒ byte-identical** (unit-tested).
- Hybrid implemented per-request by `seq_len` (raw ≤ threshold, cosine above) in both the eager logical scorer and the physical scorer; head-aggregation + anchor-budget implemented + unit-tested.
- **B1 fixed**: graph capture fails fast (clear error) for a non-off scorer rather than silently raw-scoring; **B2 fixed**: `LOOP7_MEASUREMENT=1` pins int8/mem0.7 + logs the config.
- Parameterized TP=8-shaped cross-rank determinism test covering each scorer flag (CPU multiprocess-shaped harness; full TP=8 run is a measurement-time check).
- **Measured on 8×H200**: hybrid NIAH at 4K + 16K (N=20) shows 4K ≈ 75% (recovered vs cosine's 25%) AND 16K ≈ 40% (kept). Artifact + CIs committed.
- All DS unit tests pass; the round summary records what was/wasn't measured (no binding AC-3 closure claim without the full matrix).
