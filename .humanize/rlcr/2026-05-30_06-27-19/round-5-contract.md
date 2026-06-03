# Round 5 Contract

## Mainline Objective (exactly one)
**Make the AC-4 evidence durable and complete** so task5/AC-4 is verifiable (it gates AC-5). The work (mem-fraction lift + no-gen-OOM at 0.7) is done and PASSED; this round closes the *acceptance-artifact* gaps Codex's R4 review found — it is an evidence/packaging round (no production code), with focused re-boots only to capture durable numbers.

## Target ACs (1–2)
- **AC-4** (`coding`, hardware-run) — complete the durable acceptance artifact.

## Blocking Side Issues in Scope (Codex R4 review)
1. **Full HBM budget missing the torch reserved/allocated residual + named small components.** The committed budget has NVML + torch-avail + the big named tensors, but not per-fraction torch reserved/allocated, nor explicit `written` / score-scratch / FlashMLA-metadata accounting. Fix: assemble a complete per-fraction budget — `torch_total` (139.80 GiB), `torch_used = total − avail`, named components (weights / KV / table+scales / cuda-graph pool / `written` / score-scratch computed analytically), **NVML total + NVML per-process** (the external torch-reserved proxy, since the serve HTTP API doesn't expose `torch.cuda.memory_reserved/allocated` per rank), and a **clearly-labeled residual bucket** = NVML_used − Σ(named). The budget must close.
2. **No-gen-OOM / no-growth proof not durably tracked.** `nvml_timeseries_0.7.csv` is gitignored (`*.csv`); no tracked stress request/result + server-log excerpt. Fix: capture the NVML series as a tracked **.txt**, save the sustained-stress client request/result log (tracked), and a server-log excerpt confirming no generation-time OOM during the stress.

## Queued / Out of Scope
- **Trailing whitespace** in `memfraction_sweep_int8/mf_{0.6,0.7,0.8}.txt` (Codex queued; `git diff --check` fails). Cheap — fix it this round as branch hygiene.
- **AC-5 client-SLO benchmark** — the *next* round, gated on AC-4 verification. Not started this round.
- AC-6 hardware proof, AC-7/AC-8/AC-9, gated AC-10 — later. No FlashMLA decode-assert changes (AC-3.3).

## Round Success Criteria
1. A tracked AC-4 addendum (`runs/20260530_dsv32_loop6/ac4_hbm_budget_addendum.md`) gives, per fraction (0.6/0.7/0.8): `torch_total`, `torch_used`(≈reserved), NVML total + NVML per-process, named components incl. `written`+score-scratch, and a labeled residual bucket — the budget closes (named + residual = NVML used) and is not only named tensors.
2. Durable, tracked no-OOM proof at 0.7: NVML time-series **.txt** (plateau, no monotonic growth), a stress request/result **.txt** (sustained 4096-ISL × conc traffic, all-OK), and a server-log excerpt with **no generation-time OOM**.
3. `mf_*.txt` trailing whitespace removed; `git diff --check` passes for the new artifacts.
4. Servers killed cleanly; commit + push to `jimmy`; `round-5-summary.md` with BitLesson Delta; tracker updated (task5/AC-4 → done with the addendum).

## Out-of-Scope Guards
- No production code change (evidence round). fp16 stays the launcher default.
- Re-boots are for durable measurement only; the operating point (0.7, int8) and the AC-4 verdict are unchanged.
